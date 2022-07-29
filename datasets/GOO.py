import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
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


class GOO(Dataset):
    def __init__(self, data_dir, labels_path, input_size=224, output_size=64, is_test_set=False):
        """Access to the GOO dataset.

        Args:
            data_dir (str): path to dataset's root directory
            labels_path (str): path to test/train .pickle file
            input_size (int, optional): Size of scene/depth/head. Defaults to 224.
            output_size (int, optional): Size of output heatmap. Defaults to 64.
            is_test_set (bool, optional): Prepares dataset for train/test. Defaults to False.
        """

        self.data_dir = data_dir
        self.input_size = input_size
        self.output_size = output_size
        self.is_test_set = is_test_set
        self.head_bbox_overflow_coeff = 0.1
        self.scene_transform = self.head_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.depth_transform = transforms.Compose(
            [ToColorMap(plt.get_cmap("magma")), transforms.Resize((input_size, input_size)), transforms.ToTensor()]
        )

        # Corrupted images
        bad_images = ["15/cam1/cam00001_img00530.jpg"]

        with open(os.path.join(data_dir, labels_path), "rb") as file:
            self.data = pickle.load(file)
            # Remove bad filenames
            self.data[:] = [item for item in self.data if item["filename"].replace("\\", "/") not in bad_images]
            self.length = len(self.data)

    def __getitem__(self, index):
        if self.is_test_set:
            return self.__get_test_item__(index)
        else:
            return self.__get_train_item__(index)

    def __len__(self):
        return self.length

    def __get_test_item__(self, index):
        item = self.data[index]

        path = item["filename"].replace("\\", "/")
        scene = Image.open(os.path.join(self.data_dir, path))
        scene = scene.convert("RGB")
        width, height = scene.size

        # Load depth image
        depth = Image.open(os.path.join(self.data_dir + "_depth", path))
        depth = depth.convert("L")

        eye_x, eye_y = map(float, [item["hx"] / 640, item["hy"] / 480])
        gaze_x, gaze_y = map(float, [item["gaze_cx"] / 640, item["gaze_cy"] / 480])
        x_min = (eye_x - 0.15) * width
        y_min = (eye_y - 0.15) * height
        x_max = (eye_x + 0.15) * width
        y_max = (eye_y + 0.15) * height
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max < 0:
            x_max = 0
        if y_max < 0:
            y_max = 0

        # Expand head bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        # Create head mask
        mask = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)

        # Crop the head
        head = scene.crop(map(int, [x_min, y_min, x_max, y_max]))

        # Apply transformations to scene, head, and depth if available
        if self.scene_transform is not None:
            scene = self.scene_transform(scene)

        if self.head_transform is not None:
            head = self.head_transform(head)

        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        # Generate the heat map used for prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)
        gaze_heatmap = get_label_map(
            gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size], 3, pdf="Gaussian"
        )

        return (
            scene,
            depth,
            head,
            mask,
            gaze_heatmap,
            torch.FloatTensor([[eye_x, eye_y]]),
            torch.FloatTensor([[gaze_x, gaze_y]]),
            torch.IntTensor([True]),  # gaze_inside
            torch.IntTensor([width, height]),
            path,
        )

    def __get_train_item__(self, index):
        item = self.data[index]

        path = item["filename"].replace("\\", "/")
        scene = Image.open(os.path.join(self.data_dir, path))
        scene = scene.convert("RGB")
        width, height = scene.size

        # Load depth image
        depth = Image.open(os.path.join(self.data_dir + "_depth", path))
        depth = depth.convert("L")

        eye_x, eye_y = map(float, [item["hx"] / 640, item["hy"] / 480])
        gaze_x, gaze_y = map(float, [item["gaze_cx"] / 640, item["gaze_cy"] / 480])
        x_min = (eye_x - 0.15) * width
        y_min = (eye_y - 0.15) * height
        x_max = (eye_x + 0.15) * width
        y_max = (eye_y + 0.15) * height
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max < 0:
            x_max = 0
        if y_max < 0:
            y_max = 0

        # Expand head bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        # Data augmentation
        # Jitter (expansion-only) bounding box size
        if np.random.random_sample() <= 0.5:
            k = np.random.random_sample() * 0.2
            x_min -= k * abs(x_max - x_min)
            y_min -= k * abs(y_max - y_min)
            x_max += k * abs(x_max - x_min)
            y_max += k * abs(y_max - y_min)

        # Random Crop
        if np.random.random_sample() <= 0.5:
            # Calculate the minimum valid range of the crop that doesn't exclude the head and the gaze target
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
            scene = crop(scene, crop_y_min, crop_x_min, crop_height, crop_width)

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
            scene = scene.transpose(Image.FLIP_LEFT_RIGHT)
            x_max_2 = width - x_min
            x_min_2 = width - x_max
            x_max = x_max_2
            x_min = x_min_2
            gaze_x = 1 - gaze_x

        # Random color change
        if np.random.random_sample() <= 0.5:
            scene = adjust_brightness(scene, brightness_factor=np.random.uniform(0.5, 1.5))
            scene = adjust_contrast(scene, contrast_factor=np.random.uniform(0.5, 1.5))
            scene = adjust_saturation(scene, saturation_factor=np.random.uniform(0, 1.5))

        # Create head mask
        mask = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)

        # Crop the head
        head = scene.crop(map(int, [x_min, y_min, x_max, y_max]))

        # Apply transformations to scene, head, and depth if available
        if self.scene_transform is not None:
            scene = self.scene_transform(scene)

        if self.head_transform is not None:
            head = self.head_transform(head)

        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        # Generate the heat map used for prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)
        gaze_heatmap = get_label_map(
            gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size], 3, pdf="Gaussian"
        )

        eye_coords = (eye_x, eye_y)
        gaze_coords = (gaze_x, gaze_y)

        return (
            scene,
            depth,
            head,
            mask,
            gaze_heatmap,
            torch.FloatTensor([eye_coords]),
            torch.FloatTensor([gaze_coords]),
            torch.IntTensor([True]),  # gaze_inside
            torch.IntTensor([width, height]),
            path,
        )

    def get_head_coords(self, path):
        if not self.is_test_set:
            raise NotImplementedError("This method is not implemented for training set")

        item = None
        for d in self.data:
            if d["filename"].replace("\\", "/") == path:
                item = d

        if item is None:
            raise RuntimeError("Path not found")

        path = item["filename"].replace("\\", "/")
        scene = Image.open(os.path.join(self.data_dir, path))
        scene = scene.convert("RGB")
        width, height = scene.size

        eye_x, eye_y = map(float, [item["hx"] / 640, item["hy"] / 480])
        x_min = (eye_x - 0.15) * width
        y_min = (eye_y - 0.15) * height
        x_max = (eye_x + 0.15) * width
        y_max = (eye_y + 0.15) * height
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max < 0:
            x_max = 0
        if y_max < 0:
            y_max = 0

        # Expand face bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        return x_min, y_min, x_max, y_max
