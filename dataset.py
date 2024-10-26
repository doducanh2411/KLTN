import os
import torch
import gc
import json
from torch.utils.data import Dataset
from decord import VideoReader
from PIL import Image


class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=32, transform=None, num_classes=4, captions=None):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Path to the directory containing subdirectories (classes).
            num_frames (int, optional): Number of frames to be used. Defaults to 32.
            transform (callable, optional): A function/transform to apply to each frame. Defaults to None.
            num_classes (int, optional): Number of classes for classification. Defaults to 4.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.transform = transform

        self.video_files = []
        self.labels = []
        self.captions = {}

        if captions is not None and os.path.exists(captions):
            with open(captions, 'r') as f:
                self.captions = json.load(f)

        for label, class_name in enumerate(os.listdir(root_dir)):
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                for video_file in os.listdir(class_folder):
                    self.video_files.append(
                        os.path.join(class_folder, video_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.video_files)

    def _read_frames(self, video_path):
        vr = VideoReader(video_path)
        video_frames_count = len(vr)

        if video_frames_count >= self.num_frames:
            skip_frames_window = max(video_frames_count // self.num_frames, 1)
            frame_indices = [
                i * skip_frames_window for i in range(self.num_frames)]
        else:
            frame_indices = list(range(video_frames_count))
            pad_start = (self.num_frames - video_frames_count) // 2
            frame_indices = [0] * pad_start + frame_indices + \
                [video_frames_count - 1] * \
                (self.num_frames - len(frame_indices))

        frames_list = [vr[idx].asnumpy() for idx in frame_indices]

        if self.transform is not None:
            frames_list = [self.transform(Image.fromarray(frame))
                           for frame in frames_list]

        # Ensure the output always has exactly num_frames
        if len(frames_list) > self.num_frames:
            frames_list = frames_list[:self.num_frames]
        elif len(frames_list) < self.num_frames:
            padding_needed = self.num_frames - len(frames_list)
            frames_list += [frames_list[-1]] * padding_needed

        video = torch.stack(frames_list)

        del vr
        gc.collect()
        return video

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        video = self._read_frames(video_path)
        caption = self.captions.get(video_path, "")

        return video, label, caption
