from models.base_models import SingleFrame, EarlyFusion, LateFusion, CNNLSTM, S3D, ViViT


def get_model(model_name, num_classes, num_frames=None):
    if model_name == 'single_frame':
        return SingleFrame(num_classes)
    elif model_name == 'early_fusion':
        return EarlyFusion(num_classes)
    elif model_name == 'late_fusion':
        return LateFusion(num_classes)
    elif model_name == 'cnn_lstm':
        return CNNLSTM(num_classes)
    elif model_name == 's3d':
        return S3D(num_classes)
    elif model_name == 'vivit':
        return ViViT(num_classes)
