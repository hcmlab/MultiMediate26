import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from torchvision import transforms
import random
import torch
from timm.models import create_model
from torchvision import transforms
import models  # noqa: F401
import numpy as np
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


def to_normalized_float_tensor(vid):
    vid_tensor = torch.from_numpy(vid) if isinstance(vid, np.ndarray) else vid
    # print(vid_tensor.shape)
    vid_tensor = vid_tensor.unsqueeze(0)  # Add an additional dimension to axis=0
    if vid_tensor.ndim == 4:  # Ensure the input has 4 dimensions
        return vid_tensor.permute(3, 0, 1, 2).to(torch.float32) / 255
    else:
        raise ValueError(f"Expected input with 4 dimensions, but got {vid_tensor.ndim} dimensions")



def resize(vid, size, interpolation='bilinear'):
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='/data/stars/user/smajhi/AD/feature_extraction/RGB_feature/VidMAEV2/Features/Blemore')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

    
    depth_anything = create_model(
        'vit_giant_patch14_224',
        img_size=224,
        pretrained=False,
        num_classes=710,
        all_frames=16,
        tubelet_size=2,
        drop_path_rate=0.3,
        use_mean_pooling=True)
    ckpt = torch.load('/data/stars/user/smajhi/AD/feature_extraction/RGB_feature/VidMAEV2/VideoMAEv2/ckpt/vit_g_hybrid_pt_1200e_k710_ft.pth', map_location=torch.device('cuda'))
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
    depth_anything.load_state_dict(ckpt)
    depth_anything.eval()
    depth_anything.cuda()

    transform = transforms.Compose([ToFloatTensorInZeroOne(),Resize((224, 224))])    



    # Dinov2Model.from_pretrained("facebook/dinov2-base")
    # depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    # depth_anything.load_state_dict(torch.load(f'/data/stars/user/smajhi/AD/feature_extraction/RGB_feature/Depth-Anything-V2/checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    # depth_anything = depth_anything.to(DEVICE).eval()
    
    org_video_path ='/data/stars/user/mbalazia/datasets/'
    filenames = [args.video_path]
    '''
    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)
    '''
    os.makedirs(args.outdir, exist_ok=True)
    
    margin_width = 50
    # cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    ALL_Feature = []    
    print(filenames)
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(org_video_path+filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {frame_count}")
        
        if args.pred_only: 
            output_width = frame_width
        else: 
            output_width = frame_width * 2 + margin_width

        mod_outdir = os.path.join(args.outdir, os.path.dirname(filename))

        if not os.path.exists(mod_outdir):
                os.makedirs(mod_outdir)

        
        output_path = os.path.join(mod_outdir, os.path.splitext(os.path.basename(filename))[0] + '.npy')
        #out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        cnt = 0
        frames = []
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            frames.append(raw_frame)
            if len(frames) == 16:  # Process every 16 consecutive frames
                print(f"Processing frames {cnt}-{cnt+15}")
                # Preprocess frames
                preprocessed_frames = torch.stack([transform(frame) for frame in frames]).unsqueeze(0).to(DEVICE)
                # Extract features
                preprocessed_frames = preprocessed_frames.squeeze(3)
                preprocessed_frames = preprocessed_frames.permute(0, 2, 1, 3, 4)
                # preprocessed_frames = preprocessed_frames.unsqueeze(0)
                print(preprocessed_frames.shape)
                Feature = depth_anything.forward_features(preprocessed_frames)#(preprocessed_frames).last_hidden_state[:, :, :]
                Feature_cpu = Feature.cpu().detach().numpy()
                print(Feature_cpu.shape)
                ALL_Feature.append(Feature_cpu)
                frames = []  # Clear the frame buffer
                cnt += 16
        # Process remaining frames if any
        if frames:
            print(f"Processing remaining frames {cnt-len(frames)}-{cnt-16}")
            preprocessed_frames = torch.stack([transform(frame) for frame in frames]).unsqueeze(0).to(DEVICE)
            preprocessed_frames = preprocessed_frames.squeeze(3)
            preprocessed_frames = preprocessed_frames.permute(0, 2, 1, 3, 4)
            # Stack zeros in axis=2 to make it 16
            if preprocessed_frames.shape[2] < 16:
                padding = 16 - preprocessed_frames.shape[2]
                zeros = torch.zeros(
                    (preprocessed_frames.shape[0], preprocessed_frames.shape[1], padding, preprocessed_frames.shape[3], preprocessed_frames.shape[4]),
                    device=preprocessed_frames.device
                )
                preprocessed_frames = torch.cat((preprocessed_frames, zeros), dim=2)
            Feature = depth_anything.forward_features(preprocessed_frames)
            Feature_cpu = Feature.cpu().detach().numpy()
            print(Feature_cpu.shape)
            ALL_Feature.append(Feature_cpu)



    ALL_Feature = np.asarray(ALL_Feature)
    np.save(output_path, ALL_Feature)
