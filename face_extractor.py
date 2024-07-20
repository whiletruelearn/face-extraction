from deepface import DeepFace
from pathlib import Path
from docarray import DocumentArray
from PIL import Image
from dataclasses import dataclass
import numpy as np 
import cv2
import uuid 
import glob
import json 
import os 

@dataclass
class BoundingBox:
    image_path : Path
    left: float
    top: float
    right: float
    bottom: float


class FaceExtractor:

    """ Extract Faces from images with their bounding box information"""
    def __init__(self,images_directory : Path,
                 output_directory : Path,
                 confidence_threshold : float = 0.999, 
                 lap_var_threshold : int = 11 ,
                 algorithm : str = "mtcnn",
                 crop_size : tuple = (256,256)):
        
        self.images_dir = images_directory
        self.out_dir = output_directory
        self.conf_thrshld = confidence_threshold
        self.lap_var_thrshld = lap_var_threshold 
        self.algorithm = algorithm
        self.crop_size = crop_size
        self.frame_stats = {}
        
    def _get_image_paths(self):
        return glob.glob(f"{self.images_dir}/*.jpg")
    
    
    def _is_blur_image(self, cropped_image):

        gray = cv2.cvtColor(np.asarray(cropped_image), cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return True if lap_var < self.lap_var_thrshld else False 
    
    @staticmethod
    def _get_bbox(cords,image_path):
        bbox = BoundingBox(image_path= image_path,
                           left = cords["x"],
                           top=cords["y"],
                           right=cords["x"] + cords["w"],
                           bottom=cords["y"] + cords["h"])
        
        return bbox

    def _extract_face(self, image_path : Path):
        try:
            faces = DeepFace.extract_faces(image_path,detector_backend=self.algorithm)
            self.frame_stats[image_path] = len(faces)
            for face in faces:
                face_coords = face["facial_area"]
                confidence = face["confidence"]
                if confidence >= self.conf_thrshld:
                    image = Image.open(image_path)
                    bbox = self._get_bbox(face_coords,image_path)
                    cords = [bbox.left,
                                bbox.top,
                                bbox.right, 
                                bbox.bottom]
                    cropped_image = image.crop(cords)
                    cropped_image = cropped_image.resize(self.crop_size)
                    if not self._is_blur_image(cropped_image):
                        uid = uuid.uuid1().hex[:6]
                        cropped_image.save(f"{self.out_dir}/face_{uid}.jpg")
                        with open(f"{self.out_dir}/face_{uid}.json",'w') as f:
                            f.write(json.dumps(bbox.__dict__))
            return True
        except Exception:
            print(f"Error in processing {image_path}")
            return False

    def extract_faces(self):
        os.makedirs(self.out_dir,exist_ok=True)
        image_paths = self._get_image_paths()
        extract_status = [self._extract_face(image_path) for image_path in image_paths]
        success = sum(extract_status)
        failure = len(image_paths) - success
        print(f"Succeeded: {success} ; Failure: {failure}")

    def display_faces(self):
        img_arr = DocumentArray.from_files(f"{self.out_dir}/*.jpg")
        img_arr.plot_image_sprites()