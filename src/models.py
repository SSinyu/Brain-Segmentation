
def get_model(model_type):
    if model_type == "deeplabv3p":
        from src.deeplab.networks import DeepLabV3pXc
        return DeepLabV3pXc
        
