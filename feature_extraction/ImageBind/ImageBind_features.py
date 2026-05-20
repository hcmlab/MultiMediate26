import sys
sys.path.append('/data/stars/user/adhawan/ImageBind-main/imagebind/') # to link to updated dataloader
import data

import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import pandas as pd
import torch.nn as nn
import os
import numpy as np
import pathlib
from pathlib import Path


OUTPUT_DIR = '/data/stars/user/adhawan/features_extracted/ImageBind/mpiigi/'
base_path = Path('/data/stars/user/mbalazia/datasets/mpiigi/')
model = imagebind_model.imagebind_huge(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model.to(device)

model.eval()



# Define video formats to search for
video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}  # Add more formats if required

# Initialize a list to store the paths to video files
video_path = []

# Recursively traverse the subdirectories and collect the video files
for video in base_path.rglob('*'):
    if video.is_file() and video.suffix.lower() in video_extensions:
        # Use resolve() to get the absolute path
        video_path.append(str(video.resolve()))
print("Total file number", len(video_path))


# Load data
for i in range(len(video_path)):
    inputs = {
        # ModalityType.TEXT: data.load_and_transform_text(text_list, device), #uncomment if required
        ModalityType.VISION: data.load_and_transform_video_data([video_path[i]], device, win_size=1)
        # ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device), #uncomment if required
    }

    with torch.no_grad():

        T = inputs['vision'].shape[1]  # Number of frames (ex. 6)
        frame_embeddings = []

        for t in range(T):
            # Extract single frame (keeping batch dimension)
            frame = inputs['vision'][:, t, :, 0, :, :]  # Shape: [1, 3, 2, 224, 224]

            # Pass the frame through the model
            embeddings = model({'vision': frame})

            # Extract vision embeddings
            feature = embeddings[ModalityType.VISION].to('cpu')  # Shape: [1, 1024]

            # Append to list
            frame_embeddings.append(feature.squeeze(0))  # Remove batch dim -> [1024]

        # Stack embeddings into a single tensor [T, 1024]
        frame_embeddings = torch.stack(frame_embeddings)

        # features = embeddings[ModalityType.VISION].to('cpu')
        directory, file_name = os.path.split(video_path[i])

        # Extract the parent directory (recording07)
        parent_directory = os.path.basename(directory)

        # Extract the base name of the file (subjectPos1.video)
        base_name = os.path.splitext(file_name)[0]

        # Combine the parent directory and base name
        formatted_name = f"{parent_directory}_{base_name}"

        save_path = (OUTPUT_DIR) + formatted_name +'.npy'
        np.save(save_path, frame_embeddings.numpy())  # Convert Tensor to NumPy array
        torch.cuda.empty_cache()
        print('saved file: ', save_path)



# original implementation
#  from torch.utils.data import DataLoader, Dataset
#
# # Define a custom dataset for video processing
# class VideoDataset(Dataset):
#     def __init__(self, video_paths):
#         self.video_paths = video_paths
#
#     def __len__(self):
#         return len(self.video_paths)
#
#     def __getitem__(self, idx):
#         return self.video_paths[idx]

# Function to process and save embeddings
# def process_and_save_embeddings(video_paths, model, device, output_dir):
#     dataset = VideoDataset(video_paths)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Adjust batch_size as needed
#
#     for video_path in dataloader:
#         for i in range(len(video_path)):
#             inputs = {
#                 # ModalityType.TEXT: data.load_and_transform_text(text_list, device),
#                 ModalityType.VISION: data.load_and_transform_video_data([video_path[i]], device)
#                 # ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
#             }
#
#             # Ensure inputs are on the correct device
#
#             with torch.no_grad():
#                 embeddings = model(inputs)
#
#
#                 features = embeddings[ModalityType.VISION].to('cpu')
#                 directory, file_name = os.path.split(video_path[i])
#                 print('I am here: ', features)
#                 exit()
#                 # Extract the parent directory (recording07)
#                 parent_directory = os.path.basename(directory)
#
#                 # Extract the base name of the file (subjectPos1.video)
#                 base_name = os.path.splitext(file_name)[0]
#
#                 # Combine the parent directory and base name
#                 formatted_name = f"{base_name}"
#
#                 save_path = (OUTPUT_DIR) + formatted_name + '.npy'
#                 np.save(save_path, features.numpy())  # Convert Tensor to NumPy array
#                 torch.cuda.empty_cache()
#                 print('saved file: ', save_path)
#
# process_and_save_embeddings(video_path, model, device, OUTPUT_DIR)

# print(
#     "Audio x Text: ",
#     torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
# )
# print(
#     "Vision x Audio: ",
#     torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
# )

# Expected output:
#
# Vision x Text:
# tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
#         [3.3836e-05, 9.9994e-01, 2.4118e-05],
#         [4.7997e-05, 1.3496e-02, 9.8646e-01]])
#
# Audio x Text:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
#
# Vision x Audio:
# tensor([[0.8070, 0.1088, 0.0842],
#         [0.1036, 0.7884, 0.1079],
#         [0.0018, 0.0022, 0.9960]])
