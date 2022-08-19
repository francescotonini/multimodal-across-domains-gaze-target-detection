import torch

from utils import get_memory_format

from .SpatialDepthLateFusion import SpatialDepthLateFusion


def get_model(config, device=torch.device("cuda")):
    model = SpatialDepthLateFusion(has_adv_da=config.head_da, has_multimodal_da=config.rgb_depth_da).to(
        device, memory_format=get_memory_format(config)
    )

    modules = []
    if config.freeze_scene:
        modules += [model.scene_backbone]

    if config.freeze_face:
        modules += [model.face_backbone]

    if config.freeze_depth:
        modules += [model.depth_backbone]

    for module in modules:
        for layer in module.children():
            for param in layer.parameters():
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params}")
    print(f"Total trainable params: {total_trainable_params}")

    return model


def load_pretrained(model, pretrained_dict):
    if model is None:
        return model

    if pretrained_dict is None:
        print("Pretraining is None")

        return model

    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    print(model.load_state_dict(model_dict, strict=True))

    return model
