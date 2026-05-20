import os, argparse, cv2, torch, random
import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
parser.add_argument('--features_path')
args = parser.parse_args()

batch_size = 2048

def features_path_part(features_path, p):
	return features_path.replace('.pt', '_'+str(p).zfill(3)+'.pt')

def extract_features_batch(batch):
	with torch.no_grad():
		from transformers import AutoProcessor, CLIPModel
		processor_clip = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
		model_clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
		inputs_clip = processor_clip(images=batch, return_tensors='pt')
		return model_clip.get_image_features(**inputs_clip)
	print('Unknown model:', args.model, flush=True)
	exit()

video_names = [video_name for video_name in os.listdir(args.data_path) if video_name.endswith('.mp4')]
random.shuffle(video_names)
for video_name in video_names:
	#if video_name not in ['000XX-V1c.mp4','000XX-V1p.mp4']: ##########################################
	#	continue #####################################################################################
	video_path = os.path.join(args.data_path, video_name)
	features_path = os.path.join(args.features_path, video_name.replace('.mp4', '.pt'))
	if os.path.isfile(features_path):
		print('Features already exist:', features_path, flush=True)
		continue
	if not os.path.isfile(video_path):
		print('Video does not exist:', video_path, flush=True)
		continue
	if not os.path.isdir(os.path.dirname(features_path)):
		os.makedirs(os.path.dirname(features_path))
	print('Processing:', video_path, flush=True)
	open(features_path, 'w').close()
	frames = []
	k = 0
	video = cv2.VideoCapture(video_path)
	for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))), mininterval=0):
		success, frame = video.read()
		if not success:
			break
		frames.append(np.array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
		if (i+1)%batch_size==0:
			features = extract_features_batch(frames)
			torch.save(features, features_path_part(features_path, k))
			frames = []
			k += 1
	if frames!=[]:
		features = extract_features_batch(frames)
		torch.save(features, features_path_part(features_path, k))
	video.release()
	concatenated_tensor = torch.load(features_path_part(features_path, 0))
	torch.save(concatenated_tensor, features_path)
	for j in range(k):
		tensor_j = torch.load(features_path_part(features_path, j+1))
		concatenated_tensor = torch.cat((concatenated_tensor, tensor_j), dim=0)
		torch.save(concatenated_tensor, features_path)
	for j in range(k+1):
		os.remove(features_path_part(features_path, j))
	print('Done:', features_path, torch.load(features_path).shape, flush=True)
