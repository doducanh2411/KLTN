import os
import cv2
import torch
import numpy as np
import gc
from torch.utils.data import Dataset
from decord import VideoReader


class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=900, target_size=112, num_classes=4):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Path to the directory containing subdirectories (classes).
            num_frames (int, optional): Number of frames to be used. Defaults to 900.
            target_size (int, optional): Desired size of the output images. Defaults to 112.
            num_classes (int, optional): Number of classes for classification. Defaults to 4.
        """

        self.root_dir = root_dir
        self.num_frames = num_frames
        self.target_size = (target_size, target_size)
        self.num_classes = num_classes

        self.video_files = []
        self.labels = []

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
        frames_list = []

        if video_frames_count < self.num_frames:
            padding = self.num_frames - video_frames_count
            pad_start = padding // 2
            pad_end = padding - pad_start

            for frame in vr:
                frame = frame.asnumpy()
                resized_frame = cv2.resize(frame, self.target_size)
                frames_list.append(resized_frame)

            first_frame = frames_list[0]
            last_frame = frames_list[-1]

            padding_start_frames = [first_frame for _ in range(pad_start)]
            padding_end_frames = [last_frame for _ in range(pad_end)]

            frames_list = padding_start_frames + frames_list + padding_end_frames
        elif video_frames_count == self.num_frames:
            for frame in vr:
                frame = frame.asnumpy()
                resized_frame = cv2.resize(frame, self.target_size)
                frames_list.append(resized_frame)
        else:
            skip_frames_window = max(
                int(video_frames_count / self.num_frames), 1)

            for frame_counter in range(self.num_frames):
                frame_pos = frame_counter * skip_frames_window
                frame = vr[frame_pos].asnumpy()
                resized_frame = cv2.resize(frame, self.target_size)
                frames_list.append(resized_frame)

        if len(frames_list) > self.num_frames:
            frames_list = frames_list[:self.num_frames]
        elif len(frames_list) < self.num_frames:
            padding_needed = self.num_frames - len(frames_list)
            frames_list += [frames_list[-1]] * padding_needed

        del vr
        video = torch.tensor(np.array(frames_list), dtype=torch.float32)
        del frames_list
        gc.collect()
        return video

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        video = self._read_frames(video_path)
        return video, label
