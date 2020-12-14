
def get_model(model_type):
    if model_type.lower() == "deeplabv3p":
        from src.deeplab.networks import DeepLabV3pXc
        return DeepLabV3pXc
    elif model_type.lower() == "unet":
        from src.unet.networks import UNet
        return UNet
    elif model_type.lower() in ["attentionunet", "attunet"]:
        from src.unet.networks import AttentionUNet
        return AttentionUNet
