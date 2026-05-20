import sys

from moviepy.video.io.VideoFileClip import VideoFileClip

sys.path.append('.')
sys.path.append('..')
import glob
import json
import os.path
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import argparse
from tqdm import tqdm
from torch.return_types import mode
from torchvision.io import read_video
from models.model_factory import model_factory
from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_video_chunk(video_path, start_time, end_time, output_format="TCHW"):
    print(start_time, end_time)

    # print(read_video(video_path, start_pts=start_time, end_pts=end_time-1, pts_unit='sec', output_format=output_format)[0].shape)

    return read_video(video_path, start_pts=start_time, end_pts=end_time, pts_unit='sec', output_format=output_format)


def get_video_info(filename):
    clip = VideoFileClip(filename)
    duration = clip.duration
    fps = clip.fps
    return duration, fps


def extract_video_features_by_chunks(model_type, video_path, out_file, model_weights=None):
    # @todo handle model weights
    # adjust chunk_duration_sec, win_size and stride
    model_init, weights = model_factory[model_type]
    print(f"Processing video: {video_path}")

    model = model_init(weights=None)
    model.head = nn.Linear(in_features=model.head.in_features, out_features=13)
    checkpoint = torch.load(model_weights)
    new_checkpoint = {}
    for key, value in checkpoint.items():
        key = key.replace('module.', '')
        new_checkpoint[key] = value
    model.load_state_dict(new_checkpoint)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)
    features = dict()

    def hook(module, input, output):
        features['avg_pool'] = output

    # @todo : fix make this more clean
    if model_type == 'mvit_v2_s':
        layer = model.module.norm
    else:
        if hasattr(model, 'module'):
            layer = model.module.avgpool
        else:
            layer = model.avgpool
        # layer = model.module.avgpool
    layer.register_forward_hook(hook)

    model.eval()
    preprocess = weights.transforms()

    duration, fps = get_video_info(video_path)
    chunk_duration_sec = 10
    video_features = []
    x = 0
    # total frames - number of frames in the video
    total_frames = duration * fps
    #half_win_size = win_size // 2
    print(total_frames, fps, duration)

    with torch.no_grad():
        current_start = 0
        end_of_video = False
        leftover_frames = None
        print(current_start)

        while not end_of_video:
            print("inside while loop")
            current_end = current_start + chunk_duration_sec
            if x == 1:
                print("EOV")
                end_of_video = True

            if end_of_video:
                break
            
            print(f"Processing chunk from {current_start} to {current_end} seconds")

            with open('log.txt', 'a') as log_file:
                log_file.write(f"Reading video from {current_start} to {current_end} seconds")
                print(f"Reading video from {current_start} to {current_end} seconds")

            # Load current chunk.
            current_end = min(current_end, duration)
            if current_end <= current_start:
                current_end = current_start + chunk_duration_sec  # Ensure end time is always after start time
            current_chunk = read_video_chunk(video_path, current_start, current_end)[0]
            if leftover_frames is not None:
                current_chunk = torch.cat((leftover_frames, current_chunk), dim=0)

            print(f"Read chunk of shape: {current_chunk.shape}, dtype: {current_chunk.dtype}")
            print(f"Chunk has {current_chunk.shape[0]} frames")


            # If we are in the beginning of the video, prepend zeros
            # if current_start * fps < win_size // 2:
            #     padding_frames = torch.zeros(
            #         (half_win_size, current_chunk.shape[1], current_chunk.shape[2], current_chunk.shape[3]),
            #         dtype=current_chunk.dtype)
            #     current_chunk = torch.cat((padding_frames, current_chunk), dim=0)

            # If we are at the end of the video, append padding
            if current_end >= duration:
                x = 1
                actual_frames = current_chunk.shape[0]
                expected_frames = int(chunk_duration_sec * fps)
                if actual_frames < expected_frames:
                    padding_size = expected_frames - actual_frames
                    padding_frames = torch.zeros(
                        (padding_size, current_chunk.shape[1], current_chunk.shape[2], current_chunk.shape[3]),
                        dtype=current_chunk.dtype
                    )
                    current_chunk = torch.cat((current_chunk, padding_frames), dim=0)

            # chunk processing.
            chunk_len = current_chunk.shape[0]
            win_size = 32 #adjust
            stride = 16 #adjust
            loop_entered = False
            print(f"Entering chunk processing loop with win_size={win_size}, stride={stride}")
            print(f"Loop range: {list(range(0, chunk_len - win_size + 1, stride))}")
            for i in range(0, chunk_len - win_size + 1, stride):
                
                start_frame = i
                end_frame = i + win_size

                if end_frame > chunk_len:
                    print(f"Skipping clip {start_frame}:{end_frame} — not enough frames")
                    continue

                loop_entered = True
                clip = current_chunk[start_frame:end_frame, :, :, :]

                clip_batch = preprocess(clip).unsqueeze(0)
                clip_batch = clip_batch.to(device)
                _ = model(clip_batch)
                feat = features['avg_pool'].squeeze()
                video_features.append(feat.unsqueeze(0))

            if not loop_entered:
                print(f"No valid clips for chunk {current_start}-{current_end}, len={chunk_len}, win={win_size}")
            if chunk_len >= (win_size - stride):
                leftover_frames = current_chunk[-(win_size - stride):]
            else:
                leftover_frames = None
            current_start = current_end
    
    if not video_features:
        print(f"No features extracted for {video_path}. Skipping.")
        return

    video_features = torch.cat(video_features, dim=0)

    video_features = video_features.detach().cpu().numpy()
    print(total_frames, video_features.shape)
    np.save(out_file, video_features)


def extract_features(root_dir, model_type, out_dir, window_size, stride, weights='', ext='.mp4'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    config_dict = {
        'model_type': model_type,
        'window_size': window_size,
        'stride': stride,
        'weights': str(weights or '')
    }

    with open(os.path.join(out_dir, 'config.json'), 'w') as fp:
        json.dump(config_dict, fp)
    print('written')
    video_files = glob.glob(os.path.join(root_dir, '*' + ext))
    print(video_files)
    for video_file in tqdm(video_files):
        out_file = os.path.join(out_dir, os.path.basename(os.path.dirname(video_file)) + '_' + os.path.basename(
            video_file).replace(ext, '.npy'))
        print(f'save to {out_file}')
        if os.path.exists(out_file):
            continue
        extract_video_features_by_chunks(model_type, video_file, out_file, model_weights=weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from videos.')

    parser.add_argument('root_dir', type=str, help='Root directory containing video files.')
    parser.add_argument('model_type', type=str, help='Type of model to use for feature extraction.')
    parser.add_argument('out_dir', type=str, help='Output directory to save extracted features.')
    parser.add_argument('--window_size', type=int, default=1, help='Window size for feature extraction.')
    parser.add_argument('--stride', type=int, default=1, help='Stride for feature extraction.')
    parser.add_argument('--ext', type=str, default='.mp4', help='Video file extension. Default is .mp4')
    parser.add_argument('--weights', type=str, default='', help='Model weights file path.')

    args = parser.parse_args()
    extract_features(args.root_dir, args.model_type, args.out_dir, args.window_size, args.stride, args.weights,
                     args.ext)



