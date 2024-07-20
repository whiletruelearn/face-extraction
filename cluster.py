from pathlib import Path
from sklearn.cluster import HDBSCAN 
from docarray import DocumentArray
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from deepface import DeepFace

import pandas as pd
import hashlib
import shutil 
import os
import numpy as np

class Clustering:
    def __init__(self,faces_directory: Path,
                 cluster_output_dir : Path,
                 embedding_model : str = "Facenet512") -> None:
        self.faces_dir = faces_directory
        self.cluster_out_dir = cluster_output_dir
        self.embedding_model = embedding_model 
        self.vec_to_image_path = {}
        self.mediods = None
        self.cluster_ids = None
        self.metrics = {}
        self.cluster_stats = None

    def _get_embeddings(self, image_path):
        vec =  DeepFace.represent(image_path,
                                  enforce_detection=False,
                                  model_name=self.embedding_model)[0]["embedding"]
        vec_hash = self._get_vector_hash(vec)
        self.vec_to_image_path[vec_hash] = image_path
        return vec 

    def _get_vector_hash(self,vec):
       
        vector_str = str(vec)
        hash_object = hashlib.sha256()
        hash_object.update(vector_str.encode('utf-8'))
        hash_value = hash_object.hexdigest()
        return hash_value
    

    def _get_faces(self):
        faces_arr = DocumentArray.from_files(f"{self.faces_dir}/*.jpg")
        return faces_arr
    
    def _calculate_metrics(self,X,labels):
        self.metrics["silhouette_avg"] = silhouette_score(X, labels)
        self.metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
      
    def cluster_faces(self):

        os.makedirs(self.cluster_out_dir,exist_ok=True)
        faces_arr = self._get_faces()
        embeddings = []
        face_paths = []
        

        for face in faces_arr:
            face_paths.append(face.uri)
            embeddings.append(self._get_embeddings(face.uri))

        df = pd.DataFrame(embeddings)
        X = df.values
        target = pd.DataFrame(face_paths,columns=["image_paths"])

        hdbscan = HDBSCAN(store_centers="medoid")
        clusters = hdbscan.fit_predict(X)
        target["cluster_ids"] = clusters
        
        self._calculate_metrics(X,clusters)
        self.mediods = hdbscan.medoids_
        self.cluster_ids = np.unique(clusters)
        
        target_grp = target.groupby("cluster_ids")
        self.cluster_stats = target_grp.size().sort_values(ascending=False)
        print(f"Cluster representation \n {self.cluster_stats}")

        for cluster_id in self.cluster_ids:
            target_group = target_grp.get_group(cluster_id)
            dir_path = f"{self.cluster_out_dir}/cluster_{cluster_id}"
            os.makedirs(dir_path,exist_ok=True)
            img_paths = target_group["image_paths"].unique().tolist()
            for img_pth in img_paths:
                filename = img_pth.split("/")[-1]
                shutil.copy(f"{img_pth}",f"{dir_path}/{filename}")

    def find_medoid_image(self):
        for medoid,cluster_id in zip(self.mediods,self.cluster_ids):
            vec_hash = self._get_vector_hash(medoid.tolist())
            image_path = self.vec_to_image_path[vec_hash]
            print(f"Medoid for cluster {cluster_id}")
            darr = DocumentArray.from_files(image_path)
            darr.plot_image_sprites()


    def visualize_clusters(self):

        for cluster_id in self.cluster_ids:
            if cluster_id > 0:
                image_wildcard = f"{self.cluster_out_dir}/cluster_{cluster_id}/*.jpg"
                d_arr = DocumentArray.from_files(image_wildcard)
                print(f"####### Cluster {cluster_id} ##########")
                d_arr.plot_image_sprites()
