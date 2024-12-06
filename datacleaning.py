import os
import hashlib
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.io import read_image 
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from mtcnn import MTCNN

detector = MTCNN()
input_folder = 'dataset_2'
output_folder = 'dataset_2_train'

os.makedirs(output_folder, exist_ok=True)

def crop_face(image_path, output_path):
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB (MTCNN expects RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)

    # Check if any faces were detected
    if not faces:
        print(f"No faces detected in {image_path}")
        return
    
    for face in faces:
        x, y, w, h = face['box']
        # Crop the face from the image
        face_crop = image[y:y+h, x:x+w]
        # Resize the face to the desired size
        resized_face = cv2.resize(face_crop, (224, 224))
        # Save the cropped face to the output path
        cv2.imwrite(output_path, resized_face)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        crop_face(input_path, output_path)

print("Face cropping complete!")


def preprocess_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        return print('error: no folder found')
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):
            img = cv2.imread(file_path)

            if img is None:
                print(f"Skipping non-image file: {file_name}")
                continue
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            resized_img = cv2.resize(rgb_img, (224, 224))

            r, g, b = cv2.split(resized_img)
            r_denoised = cv2.fastNlMeansDenoising(r, None, 10, 7, 21)
            g_denoised = cv2.fastNlMeansDenoising(g, None, 10, 7, 21)
            b_denoised = cv2.fastNlMeansDenoising(b, None, 10, 7, 21)
            denoised_img = cv2.merge([r_denoised, g_denoised, b_denoised])

            r, g, b = cv2.split(rgb_img)
            r_contrast = cv2.equalizeHist(r)
            g_contrast = cv2.equalizeHist(g)
            b_contrast = cv2.equalizeHist(b)
            contrast_img = cv2.merge([r_contrast, g_contrast, b_contrast])

            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, cv2.cvtColor(contrast_img, cv2.COLOR_RGB2BGR))
        print(f"Processed and saved: {file_path}")

folder_name_complex_list = ['jewel','class5','deepak']
for name in folder_name_complex_list:
    input_folder = rf"dataset_2_train\{name}"
    output_folder = rf"dataset_2_train\{name}"
    preprocess_images(input_folder, output_folder)