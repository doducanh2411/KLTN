import os
import gc
import json
import torch
from PIL import Image
from decord import VideoReader
from models.captioning_model.spacetimegpt import SpaceTimeGPT
from utils.get_norm import get_norm


def read_frames(video_path, num_frames, transform=None):
    vr = VideoReader(video_path)
    video_frames_count = len(vr)

    if video_frames_count >= num_frames:
        skip_frames_window = max(video_frames_count // num_frames, 1)
        frame_indices = [i * skip_frames_window for i in range(num_frames)]
    else:
        frame_indices = list(range(video_frames_count))
        pad_start = (num_frames - video_frames_count) // 2
        frame_indices = [0] * pad_start + frame_indices + \
                        [video_frames_count - 1] * \
            (num_frames - len(frame_indices))

    frames_list = [vr[idx].asnumpy() for idx in frame_indices]

    if transform is not None:
        frames_list = [transform(Image.fromarray(frame))
                       for frame in frames_list]

    # Ensure the output always has exactly num_frames
    if len(frames_list) > num_frames:
        frames_list = frames_list[:num_frames]
    elif len(frames_list) < num_frames:
        padding_needed = num_frames - len(frames_list)
        frames_list += [frames_list[-1]] * padding_needed

    video = torch.stack(frames_list)
    video = video.unsqueeze(0)
    del vr
    gc.collect()
    return video


def make_caption(root_dir):
    model = SpaceTimeGPT()
    captions = {}
    transform = get_norm(model_name='s3d')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _, class_name in enumerate(os.listdir(root_dir)):
        class_folder = os.path.join(root_dir, class_name)
        if os.path.isdir(class_folder):
            for video_file in os.listdir(class_folder):
                video_path = os.path.join(class_folder, video_file)
                video_features = read_frames(
                    video_path, 32, transform).to(device)
                caption = model(video_features)
                captions[video_path] = caption
                print(f"Video Path: {video_path}, Caption: {caption}")

    if not os.path.exists('caption'):
        os.makedirs('caption')

    dir_name = os.path.basename(os.path.normpath(root_dir))
    output_file = os.path.join('caption', f"{dir_name}_caption.json")
    with open(output_file, 'w') as f:
        json.dump(captions, f, indent=4)
