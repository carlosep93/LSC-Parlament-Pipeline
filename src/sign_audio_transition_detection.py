import os
import cv2
import torch
import pympi
import numpy as np
from pathlib import Path
from video2numpy import video2numpy
from pose_format import Pose
from pose_format.utils.holistic import load_holistic
from pose_format.utils.generic import pose_normalization_info, pose_hide_legs, normalize_hands_3d
from sign_language_segmentation.src.utils.probs_to_segments import probs_to_segments

class CustomSignSegmentation:
    def __init__(self, video_path: str, model_path: str):
        self.video_path = Path(video_path)
        self.model_path = Path(model_path)
        
        # Derive paths
        self.output_dir = self.video_path.parent
        self.base_name = self.video_path.stem
        self.frames_dir = self.output_dir / "frames"
        self.frames_file = self.frames_dir / f"{self.base_name}.npy"
        self.pose_file = self.output_dir / f"{self.base_name}.pose"
        self.eaf_file = self.output_dir / f"{self.base_name}.eaf"
        
        # Create directories
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video properties
        self.fps = self._get_video_fps()

    def _get_video_fps(self):
        cap = cv2.VideoCapture(str(self.video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps

    def ensure_frames(self):
        if not self.frames_file.exists():
            frames = video2numpy([str(self.video_path)], str(self.frames_dir), 1)
        else:
            frames = np.load(self.frames_file)
        return np.load(str(self.frames_file), encoding='bytes')

    def ensure_pose(self):
        if not self.pose_file.exists():
            cap = cv2.VideoCapture(str(self.video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            ## TO CHECK: Load the frames directly from numpy 
            def frame_generator(cap):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cap.release()
        
            frames = frame_generator(cap)
            pose = load_holistic(
                frames, 
                fps=self.fps, 
                width=width, 
                height=height,
                progress=False, #Enable to true to get progress bar
                additional_holistic_config={'model_complexity': 1}
            )

            with open(str(self.pose_file), "wb") as f:
                pose.write(f)

    def process_pose(self, pose, optical_flow=False):
        pose = pose.get_components(["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])
        normalization_info = pose_normalization_info(pose.header)
        pose = pose.normalize(normalization_info)
        pose_hide_legs(pose)
        normalize_hands_3d(pose)
        
        if optical_flow:
            self._add_optical_flow(pose)
        return pose

    def _add_optical_flow(self, pose):
        from pose_format.numpy.representation.distance import DistanceRepresentation
        from pose_format.utils.optical_flow import OpticalFlowCalculator
        
        calculator = OpticalFlowCalculator(fps=pose.body.fps, distance=DistanceRepresentation())
        flow = calculator(pose.body.data)
        flow = np.expand_dims(flow, axis=-1)
        flow = np.concatenate([np.zeros((1, *flow.shape[1:]), dtype=flow.dtype), flow], axis=0)
        pose.body.data = np.concatenate([pose.body.data, flow], axis=-1).astype(np.float32)

    def predict(self, model, pose):
        """
        Predict with proper shape handling
        """
        with torch.no_grad():
            torch_body = pose.body.torch()
            pose_data = torch_body.data.tensor
            
            # Debug original shape
            print(f"Original pose data shape: {pose_data.shape}")
            
            # Reshape to expected dimensions:
            # [batch_size=1, sequence_length, num_features]
            if len(pose_data.shape) == 4:
                # If shape is [frames, components, joints, dims]
                batch_size = 1
                seq_length = pose_data.shape[0]
                features = pose_data.shape[1] * pose_data.shape[2] * pose_data.shape[3]
                pose_data = pose_data.reshape(batch_size, seq_length, features)
            
            # Ensure we have 225 features as expected by model
            if pose_data.shape[-1] != 225:
                pose_data = pose_data[..., :225]
            
            # Debug final shape
            print(f"Reshaped pose data: {pose_data.shape}")
            
            return model(pose_data)

    def generate_segments(self):
        # Read and process pose
        with open(str(self.pose_file), "rb") as f:
            pose = Pose.read(f.read())
            if 'E4' in str(self.model_path):
                pose = self.process_pose(pose, optical_flow=True)
            else:
                pose = self.process_pose(pose)

        # Load model and predict
        model = torch.jit.load(str(self.model_path))
        model.eval()
        probs = self.predict(model, pose)

        # Generate segments
        sign_segments = probs_to_segments(probs["sign"], 50, 50, threshold_likeliest=True)
        sentence_segments = probs_to_segments(probs["sentence"], 90, 90, threshold_likeliest=True)

        # Create ELAN file
        self._create_elan_file(sign_segments, sentence_segments, pose.body.fps)
        
        return sign_segments, sentence_segments

    def _create_elan_file(self, sign_segments, sentence_segments, fps):
        eaf = pympi.Elan.Eaf(author="sign-language-processing/transcription")
        
        # Add video
        mimetype = "video/mp4" if str(self.video_path).endswith(".mp4") else None
        eaf.add_linked_file(str(self.video_path), mimetype=mimetype)
        
        # Add pose file
        eaf.add_linked_file(str(self.pose_file), mimetype="application/pose")

        # Add segments
        for tier_id, segments in {"SIGN": sign_segments, "SENTENCE": sentence_segments}.items():
            eaf.add_tier(tier_id)
            for segment in segments:
                start_ms = int(segment["start"] / fps * 1000)
                end_ms = int(segment["end"] / fps * 1000)
                eaf.add_annotation(tier_id, start_ms, end_ms)

        eaf.to_file(str(self.eaf_file))

    def compute_silences(self):
        eaf = pympi.Elan.Eaf(file_path=str(self.eaf_file))
        sign_annotations = eaf.get_annotation_data_for_tier('SENTENCE')
        segments = [(start / 1000.0, end / 1000.0) for start, end, _ in sign_annotations]
        
        segments_with_silences = []
        if segments:
            segments_with_silences.append((segments[0][0], segments[0][1], 0.0))
            for i in range(1, len(segments)):
                silence = segments[i][0] - segments[i-1][1]
                segments_with_silences.append((segments[i][0], segments[i][1], silence))
        
        return segments_with_silences

    def process(self):
        """Main processing pipeline"""
        frames = self.ensure_frames()
        self.ensure_pose()
        sign_segments, sentence_segments = self.generate_segments()
        silences = self.compute_silences()
        return list(silences)

    
    
    