from fastapi import FastAPI, UploadFile, File
from dataclasses import dataclass
from deepface import DeepFace
from face_extractor import FaceExtractor

import glob
import cv2
import numpy as np 
import pickle
import tempfile
import json 

with open("cluster_mediods.pkl","rb") as f:
    medoids = pickle.load(f)

with open("cluster_ids.pkl","rb") as f:
    cluster_ids = pickle.load(f)

def _get_embedding(image_path):
    return DeepFace.represent(image_path,
                                  enforce_detection=False,
                                  model_name="Facenet512")[0]["embedding"]


def find_cluster_id(face_image_path):
    query_vector = _get_embedding(face_image_path)
    distances = np.linalg.norm(medoids - query_vector, axis=1)
    most_similar_index = np.argmin(distances)
    return cluster_ids[most_similar_index]


@dataclass
class BoundingBox:
    cluster_id : int
    left: float
    top: float
    right: float
    bottom: float


app = FastAPI()

@app.post("/get_people_from_image")
async def get_people_from_image(image_file: UploadFile = File(...)):
    results = []
    temp_dir = tempfile.mkdtemp()
    frame_img_path = f"{temp_dir}/temp_image.jpg"
    with open(frame_img_path, "wb") as temp_image:
        temp_image.write(await image_file.read())
    
    faces_dir = f"{temp_dir}/faces"
    face_extractor = FaceExtractor(images_directory=temp_dir,
                                   output_directory=faces_dir,
                                   confidence_threshold=0.90,
                                   lap_var_threshold=2)
    face_extractor.extract_faces()

    json_files = glob.glob(f"{faces_dir}/*.json")
    for json_file in json_files:
        with open(json_file,'r') as f:
            bbox_info =  json.load(f)
        image_path = json_file.replace(".json",".jpg")
        cluster_id = find_cluster_id(image_path)
        bbox_info["cluster_id"] = int(cluster_id)
        results.append(bbox_info)

    print(faces_dir)
    print(results)
    return {"result": results}


