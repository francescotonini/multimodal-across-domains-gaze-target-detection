import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import Decoder, DomainClassifier, Encoder, ModalityReverse, ResNet


class SpatialDepthLateFusion(nn.Module):
    def __init__(
        self,
        scene_layers=[3, 4, 6, 3, 2],
        depth_layers=[3, 4, 6, 3, 2],
        face_layers=[3, 4, 6, 3, 2],
        scene_inplanes=64,
        depth_inplanes=64,
        face_inplanes=64,
        has_adv_da=False,
        has_multimodal_da=False,
    ):
        super(SpatialDepthLateFusion, self).__init__()

        # Common
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.has_adv_da = has_adv_da
        if self.has_adv_da:
            self.adv_da = DomainClassifier()

        self.has_multimodal_da = has_multimodal_da
        if self.has_multimodal_da:
            self.multimodal_da = ModalityReverse()

        self.scene_backbone = ResNet(in_channels=4, layers=scene_layers, inplanes=scene_inplanes)

        self.depth_backbone = ResNet(in_channels=4, layers=depth_layers, inplanes=depth_inplanes)

        self.face_backbone = ResNet(in_channels=3, layers=face_layers, inplanes=face_inplanes)

        # Attention
        self.attn = nn.Linear(1808, 1 * 7 * 7)

        # Encoding for scene saliency
        self.scene_encoder = Encoder()

        # Encoding for depth saliency
        self.depth_encoder = Encoder()

        # Decoding
        self.decoder = Decoder()

    def forward(self, images, depths, head, face, alpha=None):
        # Scene feature map
        scene_feat = self.scene_backbone(torch.cat((images, head), dim=1))

        # Depth feature map
        depth_feat = self.depth_backbone(torch.cat((depths, head), dim=1))

        # Face feature map
        face_feat = self.face_backbone(face)
        # Reduce feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)

        # Reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)

        # Attention layer
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2)
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        # Scene feature map * attention
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat)

        # Depth feature map * attention
        attn_applied_depth_feat = torch.mul(attn_weights, depth_feat)

        # Scene encode
        scene_encoding = self.scene_encoder(torch.cat((attn_applied_scene_feat, face_feat), 1))

        # Depth encoding
        depth_encoding = self.depth_encoder(torch.cat((attn_applied_depth_feat, face_feat), 1))

        # Deconv by scene and depth encoding summation
        x = self.decoder(scene_encoding + depth_encoding)

        # Adversarial DA on head branch
        label = None
        if self.has_adv_da:
            label = self.adv_da(face_feat, alpha)

        # Multimodal DA on rgb/depth branches
        rgb_rec = None
        depth_rec = None
        if self.has_multimodal_da:
            rgb_rec, depth_rec = self.multimodal_da(scene_feat, depth_feat)

        return x, label, rgb_rec, depth_rec
