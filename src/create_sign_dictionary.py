import glob
from video2numpy import video2numpy
from transformers import VideoMAEImageProcessor, VideoMAEModel
import numpy as np
import torch
import os
import argparse
import pickle


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Path to the VideoMAE model.")
    parser.add_argument("-v", "--videos", required=True, help="Path to the isolated sign videos.")
    parser.add_argument("-f", "--frames", required=True, help="Path to the frame files in numpy format.")
    parser.add_argument("-o", "--output", required=True, help="Output file to store the dictionary.")
    return parser

def compute_video_embeddings(video):
    sign_features=[]
    for i in range(0,video.shape[0],16):
        try:
            inputs = processor(list(video[i:i+16]), return_tensors="pt")
            inputs = inputs.to('cuda:0')
            with torch.no_grad():
                outputs = model(**inputs)

            sign_features.append(torch.mean(outputs.last_hidden_state, dim=1))
        except:
            pass
    return sign_features


def create_sign_dict(model, processor, video_path, frames_path, take_every=1, device='cuda:0'):
    sign_dict = {}
    for filename in os.listdir(video_path):
        sign = filename.split('.mp4')[0]

        # Convert video to numpy array file
        video = glob.glob(os.path.join(video_path,filename))
        video2numpy(video, frames_path, take_every)

        #Load numpy video and transpose to match model input
        frames_filename = filename.replace('.mp4','.npy')
        video = np.load(os.path.join(frames_path, frames_filename), encoding='bytes')
        video = np.transpose(video, (0, 3, 1, 2))

        embeddings = compute_video_embeddings(video)
        sign_dict[sign] = embeddings

    return sign_dict       


def save_sign_dict(sign_dict, out_dir):
    with open(out_dir, 'wb') as f: 
        pickle.dump(sign_dict,f)


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
 
    processor = VideoMAEImageProcessor.from_pretrained(args.model, device='cuda:0')
    model = VideoMAEModel.from_pretrained(args.model)
    model = model.to(device)

    sign_dict = create_sign_dict(model, processor, args.videos, args.frames, device=device)
    save_sign_dict(sign_dict, args.output)
 






