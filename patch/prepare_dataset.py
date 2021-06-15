from facenet_pytorch import MTCNN
import os
from PIL import Image
from pathlib import Path

from shutil import copyfile


def face_crop_raw_images(input_path, output_path):
    mtcnn = MTCNN(image_size=112)
    for folder_name in os.listdir(input_path):
        Path(os.path.join(output_path, folder_name)).mkdir(parents=True, exist_ok=True)
        for image_path in os.listdir(os.path.join(input_path, folder_name)):
            img = Image.open(os.path.join(input_path, folder_name, image_path))
            mtcnn(img, save_path=os.path.join(output_path, folder_name, image_path))


def strip_lfw(input_path, output_path):
    for folder_name in os.listdir(input_path):
        for image_path in os.listdir(os.path.join(input_path, folder_name)):
            copyfile(os.path.join(input_path, folder_name, image_path), os.path.join(output_path, image_path))


face_crop_raw_images('../datasets/celebA', '../datasets/celebA_strip')
