
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor, InterpolationMode


def get_norm(model_name):
    mobileNet_transform = Compose([
        Resize(232, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    s3d_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.43216, 0.394666, 0.37645],
                  std=[0.22803, 0.22145, 0.216989]),
    ])

    vivit_transform = Compose([
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5]),
    ])

    if model_name == 'single_frame' or model_name == 'early_fusion' or model_name == 'late_fusion' or model_name == 'cnn_lstm':
        return mobileNet_transform
    elif model_name == 's3d':
        return s3d_transform
    elif model_name == 'vivit':
        return vivit_transform
