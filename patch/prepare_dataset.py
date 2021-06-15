from facenet_pytorch import MTCNN
import os
from PIL import Image
from pathlib import Path

from shutil import copyfile
from collections import Counter

from tqdm import tqdm

def face_crop_raw_images(input_path, output_path):
    mtcnn = MTCNN()
    for folder_name in os.listdir(input_path):
        # Path(os.path.join(output_path, folder_name)).mkdir(parents=True, exist_ok=True)
        for image_path in os.listdir(os.path.join(input_path, folder_name)):
            img = Image.open(os.path.join(input_path, folder_name, image_path))
            mtcnn(img, save_path=os.path.join(output_path, image_path), return_prob=True)


def strip_lfw(input_path, output_path):
    for folder_name in os.listdir(input_path):
        for image_path in os.listdir(os.path.join(input_path, folder_name)):
            copyfile(os.path.join(input_path, folder_name, image_path), os.path.join(output_path, image_path))


# face_crop_raw_images('../datasets/lfw', '../datasets/lfw_cropped')
# strip_lfw('../datasets/lfw', '../datasets/lfw_strip')

def create_celeb_folders(root_path):
    lab_dict = {}
    with open(os.path.join(root_path, 'identity_celebA.txt'), 'r') as lab_file:
        for line in tqdm(lab_file):
            [image_name, celeb_lab] = line.split()
            lab_dict[image_name.replace('jpg', 'png')] = celeb_lab
            Path(os.path.join(root_path, celeb_lab)).mkdir(parents=True, exist_ok=True)

    value, count = Counter(lab_dict.values()).most_common(1)[0]
    print('Most common celeb: {} with {} images'.format(value, count))

    target_folder = '../datasets/celebA'
    for image in tqdm(os.listdir(os.path.join(root_path, 'img_align_celeba_png'))):
        copyfile(os.path.join(root_path, 'img_align_celeba_png', image), os.path.join(target_folder, lab_dict[image], image))


create_celeb_folders('../datasets/celebA')
