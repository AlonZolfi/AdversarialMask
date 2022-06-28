from facenet_pytorch import MTCNN
import os
from PIL import Image

from shutil import copyfile
from collections import Counter

from tqdm import tqdm
import cv2
import numpy as np
from skimage import transform as trans
from pathlib import Path
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def face_crop_raw_images_batched(input_path, output_path):
    tform = trans.SimilarityTransform()
    mtcnn = MTCNN(image_size=112, device=device)
    arcface_src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)
    batch_size = 128
    for folder_name in tqdm(os.listdir(input_path)):
        Path(os.path.join(output_path, folder_name)).mkdir(parents=True, exist_ok=True)
        all_images_path = os.listdir(os.path.join(input_path, folder_name))
        chunks = (len(all_images_path) - 1) // batch_size + 1
        for i in range(chunks):
            batched_images_path = all_images_path[i*batch_size: (i+1)*batch_size]
            images = []
            for image_path in batched_images_path:
                images.append(cv2.cvtColor(cv2.imread(os.path.join(input_path, folder_name, image_path)), cv2.COLOR_BGR2RGB))
            _, _, points = mtcnn.detect(images, landmarks=True)
            for j, point in enumerate(points):
                if point is not None:
                    p = point[0]
                    tform.estimate(p, arcface_src)
                    M = tform.params[0:2, :]
                    image = cv2.imread(os.path.join(input_path, folder_name, batched_images_path[j]))
                    cut = cv2.warpAffine(image, M, (112, 112))
                    cv2.imwrite(os.path.join(output_path, folder_name, batched_images_path[j]), cut)
                    
                    
face_crop_raw_images_batched('../datasets/CASIA-WebFace', '../datasets/CASIA-WebFace_aligned')