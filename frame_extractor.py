import cv2
import os 
from pathlib import Path


class FrameExtractor:
    """Extract Frames based on a sampling frequency from video"""
    def __init__(self,vid_path : Path , 
                 output_dir : Path, 
                 sampling_frequency : int = 1) -> None:
        self.vid_path = vid_path
        self.output_dir = output_dir
        self.sampling_frequency = sampling_frequency


    def extract_frames(self):
        os.makedirs(self.output_dir, exist_ok=True)
        video = cv2.VideoCapture(self.vid_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        target_frame_index = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if frame_count == target_frame_index:
                output_path = os.path.join(self.output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(output_path, frame)
                target_frame_index += int(fps * self.sampling_frequency)
            frame_count += 1
        video.release()
        cv2.destroyAllWindows()
    