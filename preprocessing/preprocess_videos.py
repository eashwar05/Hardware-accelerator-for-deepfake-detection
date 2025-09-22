# data_preprocessing/preprocess_videos.py (Updated for FaceForensics++)

import os
import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import argparse

def process_videos(input_dir, output_dir, frames_to_sample=50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=False, device=device, post_process=False, min_face_size=40)
    os.makedirs(output_dir, exist_ok=True)
    video_files = [os.path.join(root, file) for root, _, files in os.walk(input_dir) for file in files if file.endswith('.mp4')]

    for video_path in tqdm(video_files, desc="Processing Videos"):
        try:
            # --- MODIFICATION FOR FACEFORENSICS++ STRUCTURE ---
            # Get the name of the folder containing the video (e.g., 'original', 'Deepfakes')
            parent_folder = os.path.basename(os.path.dirname(video_path))
            
            # Determine if the video is real or fake based on its parent folder
            if parent_folder == 'original':
                label_dir_name = "real"
            else:
                label_dir_name = "fake"
            # --- END OF MODIFICATION ---

            video_output_dir = os.path.join(output_dir, label_dir_name)
            os.makedirs(video_output_dir, exist_ok=True)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < frames_to_sample: continue

            frame_indices = torch.linspace(0, frame_count - 1, frames_to_sample).long()
            saved_frame_count = 0
            for i in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i.item())
                ret, frame = cap.read()
                if not ret: continue

                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                boxes, _ = mtcnn.detect(frame_pil)
                
                if boxes is not None:
                    box = boxes[0]
                    margin_x, margin_y = (box[2] - box[0]) * 0.2, (box[3] - box[1]) * 0.2
                    box = [max(0, box[0] - margin_x), max(0, box[1] - margin_y),
                           min(frame_pil.width, box[2] + margin_x), min(frame_pil.height, box[3] + margin_y)]
                    
                    face = frame_pil.crop(box).resize((384, 384), Image.LANCZOS)
                    face.save(os.path.join(video_output_dir, f"{video_name}_{parent_folder}_frame_{saved_frame_count}.png"))
                    saved_frame_count += 1
            cap.release()
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess video datasets for deepfake detection.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the root directory of the FaceForensics++ dataset.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the directory where processed faces will be saved.")
    parser.add_argument('--num_frames', type=int, default=50, help="Number of frames to sample from each video.")
    args = parser.parse_args()
    process_videos(args.input_dir, args.output_dir, args.num_frames)