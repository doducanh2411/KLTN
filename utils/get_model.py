from models.cnn_lstm import CNNLSTM
from models.early_fusion import EarlyFusion
from models.late_fusion import LateFusion
from models.s3d_cnn import S3D
from models.single_frame import SingleFrame
from models.vivit import ViViT


def get_model(model_name, num_classes, num_frames=None, image_size=None):
    if model_name == "s3d":
        return S3D(num_classes)
    elif model_name == "cnn_lstm":
        return CNNLSTM(num_classes)
    elif model_name == "early_fusion":
        return EarlyFusion(num_classes, num_input_channels=num_frames)
    elif model_name == "late_fusion":
        return LateFusion(num_classes)
    elif model_name == "single_frame":
        return SingleFrame(num_classes)
    elif model_name == "vivit":
        return ViViT(num_classes, image_size, num_frames)
    else:
        raise ValueError(f"Model {model_name} not found")
