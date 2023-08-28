import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    crop,
)

from datasets.transforms.ToColorMap import ToColorMap
from utils import get_head_mask, get_label_map


class GazeFollow(Dataset):
    def __init__(self, data_dir, labels_path, input_size=224, output_size=64, is_test_set=False):
        self.data_dir = data_dir
        self.input_size = input_size
        self.output_size = output_size
        self.is_test_set = is_test_set
        self.head_bbox_overflow_coeff = 0.1  # Will increase/decrease the bbox of the head by this value (%)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.depth_transform = transforms.Compose(
            [ToColorMap(plt.get_cmap("magma")), transforms.Resize((input_size, input_size)), transforms.ToTensor()]
        )

        column_names = [
            "path",
            "idx",
            "body_bbox_x",
            "body_bbox_y",
            "body_bbox_w",
            "body_bbox_h",
            "eye_x",
            "eye_y",
            "gaze_x",
            "gaze_y",
            "bbox_x_min",
            "bbox_y_min",
            "bbox_x_max",
            "bbox_y_max",
        ]
        if is_test_set is False:
            column_names.append("inout")

        df = pd.read_csv(labels_path, sep=",", names=column_names, usecols=column_names, index_col=False)

        if is_test_set:
            df = df[
                ["path", "eye_x", "eye_y", "gaze_x", "gaze_y", "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max"]
            ].groupby(["path", "eye_x"])
            self.keys = list(df.groups.keys())
            self.X = df
            self.length = len(self.keys)
        else:
            df = df[df["inout"] != -1]  # only use "in" or "out "gaze (-1 is invalid, 0 is out gaze)
            df.reset_index(inplace=True)

            self.X = df["path"]
            self.y = df[
                ["bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max", "eye_x", "eye_y", "gaze_x", "gaze_y", "inout"]
            ]
            self.length = len(df)

    def __getitem__(self, index):
        if self.is_test_set:
            return self.__get_test_item__(index)
        else:
            return self.__get_train_item__(index)

    def __len__(self):
        return self.length

    def __get_train_item__(self, index):
        path = self.X.iloc[index]
        x_min, y_min, x_max, y_max, eye_x, eye_y, gaze_x, gaze_y, gaze_inside = self.y.iloc[index]

        # Expand face bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert("RGB")
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        # Data augmentation
        # Jitter (expansion-only) bounding box size
        if np.random.random_sample() <= 0.5:
            self.head_bbox_overflow_coeff = np.random.random_sample() * 0.2
            x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
            x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        # Random Crop
        if np.random.random_sample() <= 0.5:
            # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
            crop_x_min = np.min([gaze_x * width, x_min, x_max])
            crop_y_min = np.min([gaze_y * height, y_min, y_max])
            crop_x_max = np.max([gaze_x * width, x_min, x_max])
            crop_y_max = np.max([gaze_y * height, y_min, y_max])

            # Randomly select a random top left corner
            if crop_x_min >= 0:
                crop_x_min = np.random.uniform(0, crop_x_min)
            if crop_y_min >= 0:
                crop_y_min = np.random.uniform(0, crop_y_min)

            # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
            crop_width_min = crop_x_max - crop_x_min
            crop_height_min = crop_y_max - crop_y_min
            crop_width_max = width - crop_x_min
            crop_height_max = height - crop_y_min
            # Randomly select a width and a height
            crop_width = np.random.uniform(crop_width_min, crop_width_max)
            crop_height = np.random.uniform(crop_height_min, crop_height_max)

            # Crop it
            img = crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

            # Record the crop's (x, y) offset
            offset_x, offset_y = crop_x_min, crop_y_min

            # Convert coordinates into the cropped frame
            x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
            gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), (gaze_y * height - offset_y) / float(
                crop_height
            )

            width, height = crop_width, crop_height

        # Random flip
        if np.random.random_sample() <= 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            x_max_2 = width - x_min
            x_min_2 = width - x_max
            x_max = x_max_2
            x_min = x_min_2
            gaze_x = 1 - gaze_x

        # Random color change
        if np.random.random_sample() <= 0.5:
            img = adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
            img = adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
            img = adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Load depth image
        depth_path = path.replace("train", "depth").replace("test2", "depth2")
        depth = Image.open(os.path.join(self.data_dir, depth_path))
        depth = depth.convert("L")

        # Apply transformation to images...
        if self.image_transform is not None:
            img = self.image_transform(img)
            face = self.image_transform(face)

        # ... and depth
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        # Generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)
        gaze_heatmap = get_label_map(
            gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size], 3, pdf="Gaussian"
        )

        gaze_coords = (gaze_x, gaze_y)
        eye_coords = (eye_x, eye_y)

        return (
            img,
            depth,
            face,
            head,
            gaze_heatmap,
            torch.FloatTensor([eye_coords]),
            torch.FloatTensor([gaze_coords]),
            torch.IntTensor([bool(gaze_inside)]),
            torch.IntTensor([width, height]),
            path,
        )

    def __get_test_item__(self, index):
        eye_coords = []
        gaze_coords = []
        gaze_inside = []
        for _, row in self.X.get_group(self.keys[index]).iterrows():
            path = row["path"]
            x_min = row["bbox_x_min"]
            y_min = row["bbox_y_min"]
            x_max = row["bbox_x_max"]
            y_max = row["bbox_y_max"]
            gaze_x = row["gaze_x"]
            gaze_y = row["gaze_y"]
            eye_x = row["eye_x"]
            eye_y = row["eye_y"]
            # All ground truth gaze are stacked up
            eye_coords.append([eye_x, eye_y])
            gaze_coords.append([gaze_x, gaze_y])
            gaze_inside.append(True)

        for _ in range(len(gaze_coords), 20):
            # Pad dummy gaze to match size for batch processing
            eye_coords.append([-1, -1])
            gaze_coords.append([-1, -1])
            gaze_inside.append(False)
        eye_coords = torch.FloatTensor(eye_coords)
        gaze_coords = torch.FloatTensor(gaze_coords)
        gaze_inside = torch.IntTensor(gaze_inside)

        # Expand face bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert("RGB")
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Load depth image
        depth_path = path.replace("train", "depth").replace("test2", "depth2")
        depth = Image.open(os.path.join(self.data_dir, depth_path))
        depth = depth.convert("L")

        # Apply transformation to images...
        if self.image_transform is not None:
            img = self.image_transform(img)
            face = self.image_transform(face)

        # ... and depth
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        # Generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)
        num_valid = 0
        for gaze_x, gaze_y in gaze_coords:
            if gaze_x == -1:
                continue

            num_valid += 1
            gaze_heatmap = get_label_map(
                gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size], 3, pdf="Gaussian"
            )
        gaze_heatmap /= num_valid

        return (
            img,
            depth,
            face,
            head,
            gaze_heatmap,
            eye_coords,
            gaze_coords,
            gaze_inside[0],
            torch.IntTensor([width, height]),
            path,
        )

    def get_head_coords(self, path):
        if not self.is_test_set:
            raise NotImplementedError("This method is not implemented for training set")

        # NOTE: this is not 100% accurate. I should also condition by eye_x
        # However, for the application of this method it should be enough
        key_index = next((key for key in self.keys if key[0] == path), -1)
        if key_index == -1:
            raise RuntimeError("Path not found")

        for _, row in self.X.get_group(key_index).iterrows():
            x_min = row["bbox_x_min"]
            y_min = row["bbox_y_min"]
            x_max = row["bbox_x_max"]
            y_max = row["bbox_y_max"]

        # Expand face bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        return x_min, y_min, x_max, y_max
