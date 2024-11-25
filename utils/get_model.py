from models.base_models import SingleFrame, EarlyFusion, LateFusion, CNNLSTM, S3D, ViViT, Swin, Timesformer, VideoMAE
from models.mutimodal_models import MultiModalViViT, MultiModalS3D, MultiModalCNNLSTM
from models.attention_multimodal_models import AttentionMultiModalCNNLSTM, AttentionMultiModalEarlyFusion, AttentionMultiModalLateFusion, AttentionMultiModalS3D, AttentionMultiModalViViT
from models.multimodal_text_audio_models import MultiModalTextAudioCNNLSTM, MultiModalTextAudioS3D, MultiModalLateFusionTextAudio, MultiModalTextAudioViViT, MultiModalTextAudioEarlyFusion


def get_model(model_name, num_classes, num_frames=None):
    if model_name == 'single_frame':
        return SingleFrame(num_classes)
    elif model_name == 'early_fusion':
        return EarlyFusion(num_classes, num_input_channels=num_frames)
    elif model_name == 'late_fusion':
        return LateFusion(num_classes)
    elif model_name == 'cnn_lstm':
        return CNNLSTM(num_classes)
    elif model_name == 's3d':
        return S3D(num_classes)
    elif model_name == 'vivit':
        return ViViT(num_classes)
    elif model_name == 'multimodal_vivit':
        return MultiModalViViT(num_classes)
    elif model_name == 'multimodal_s3d':
        return MultiModalS3D(num_classes)
    elif model_name == 'multimodal_cnn_lstm':
        return MultiModalCNNLSTM(num_classes)
    elif model_name == 'swin':
        return Swin(num_classes)
    elif model_name == 'timesformer':
        return Timesformer(num_classes)
    elif model_name == 'videomae':
        return VideoMAE(num_classes)
    elif model_name == 'attention_cnn_lstm':
        return AttentionMultiModalCNNLSTM(num_classes)
    elif model_name == 'attention_early_fusion':
        return AttentionMultiModalEarlyFusion(num_classes)
    elif model_name == 'attention_late_fusion':
        return AttentionMultiModalLateFusion(num_classes)
    elif model_name == 'attention_s3d':
        return AttentionMultiModalS3D(num_classes)
    elif model_name == 'attention_vivit':
        return AttentionMultiModalViViT(num_classes)
    elif model_name == 'multimodal_text_audio_cnn_lstm':
        return MultiModalTextAudioCNNLSTM(num_classes)
    elif model_name == 'multimodal_text_audio_s3d':
        return MultiModalTextAudioS3D(num_classes)
    elif model_name == 'multimodal_text_audio_late_fusion':
        return MultiModalLateFusionTextAudio(num_classes)
    elif model_name == 'multimodal_text_audio_vivit':
        return MultiModalTextAudioViViT(num_classes)
    elif model_name == 'multimodal_text_audio_early_fusion':
        return MultiModalTextAudioEarlyFusion(num_classes)
