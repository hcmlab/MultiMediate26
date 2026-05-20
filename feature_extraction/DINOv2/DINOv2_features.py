import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from transformers import AutoImageProcessor, Dinov2Model
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=3)#10
# pca_features = pca.fit_transform(features)

# from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='/data/stars/user/smajhi/AD/feature_extraction/RGB_feature/PCA_DinoV2/Features/Blemore')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    

    depth_anything = Dinov2Model.from_pretrained("facebook/dinov2-base")
    # depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    # depth_anything.load_state_dict(torch.load(f'/data/stars/user/smajhi/AD/feature_extraction/RGB_feature/Depth-Anything-V2/checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    org_video_path = '/data/stars/user/mbalazia/datasets/' #'/data/stars/user/mbalazia/noxi/'
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
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    ALL_Feature = []    
    print(filenames)
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(org_video_path+filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        
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
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            if cnt %16 ==0:
                print(cnt)
                pixel_values = torch.tensor(raw_frame).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                batch_size, _, height, width = pixel_values.shape
                Feature = depth_anything(pixel_values).last_hidden_state[:, :, :] #.infer_image(raw_frame, args.input_size)
                # print(Feature.shape)
                Feature_cpu = Feature.cpu().detach().numpy()
                Feature_cpu = np.squeeze(Feature_cpu, axis=0)
                Feature_cpu = Feature_cpu.transpose(1, 0)
                Feature_PCA = pca.fit_transform(Feature_cpu)
                Feature_PCA = np.expand_dims(Feature_PCA, axis=0)
                print(Feature_PCA.shape)                
                ALL_Feature.append(Feature_PCA)
                # exit()
                # break
            cnt+=1
    ALL_Feature = np.asarray(ALL_Feature)
    np.save(output_path, ALL_Feature)
