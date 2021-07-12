from facenet_pytorch import MTCNN
import os
from PIL import Image
from pathlib import Path

from shutil import copyfile
from collections import Counter

from tqdm import tqdm
import cv2
import numpy as np
from skimage import transform as trans
from pathlib import Path


def face_crop_raw_images_from_lab_cameras(input_path, output_path):
    tform = trans.SimilarityTransform()
    mtcnn = MTCNN()
    arcface_src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)
    p = Path(input_path)
    for image_path in p.glob('**/*.jpg'):
        img = cv2.imread(os.path.join(input_path, image_path.name))
        # mtcnn(img, save_path=os.path.join(output_path, folder_name, image_path))
        boxes, _, points = mtcnn.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), landmarks=True)
        if boxes is None:
            print('Face not found {}'.format(image_path.name))
            continue
        tform.estimate(points[0], arcface_src)
        M = tform.params[0:2, :]
        cut = cv2.warpAffine(img, M, (112, 112))

        cv2.imwrite(os.path.join(output_path, image_path.name), cut)


face_crop_raw_images_from_lab_cameras('../data/gmail', '../datasets/persons/alon/train_new')


def face_crop_raw_images(input_path, output_path):
    tform = trans.SimilarityTransform()
    mtcnn = MTCNN()
    arcface_src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)
    for folder_name in os.listdir(input_path):
        Path(os.path.join(output_path, folder_name)).mkdir(parents=True, exist_ok=True)
        for image_path in os.listdir(os.path.join(input_path, folder_name)):
            img = cv2.imread(os.path.join(input_path, folder_name, image_path))
            # mtcnn(img, save_path=os.path.join(output_path, folder_name, image_path))
            boxes, _, points = mtcnn.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), landmarks=True)
            if points is None:
                continue
            tform.estimate(points[0], arcface_src)
            M = tform.params[0:2, :]
            cut = cv2.warpAffine(img, M, (112, 112))
            cv2.imwrite(os.path.join(output_path, folder_name, image_path), cut)


def strip_lfw(input_path, output_path):
    for folder_name in os.listdir(input_path):
        for image_path in os.listdir(os.path.join(input_path, folder_name)):
            copyfile(os.path.join(input_path, folder_name, image_path), os.path.join(output_path, image_path))


# face_crop_raw_images('../datasets/celebA', '../datasets/celebA_stripa')

# face_crop_raw_images('../datasets/lfw.csv', '../datasets/lfw_cropped')
# strip_lfw('../datasets/lfw.csv', '../datasets/lfw_strip')

def create_celeb_folders(root_path):
    lab_dict = {}
    with open(os.path.join(root_path, 'identity_celebA.txt'), 'r') as lab_file:
        for line in tqdm(lab_file):
            [image_name, celeb_lab] = line.split()
            lab_dict[image_name.replace('jpg', 'png')] = celeb_lab
            # Path(os.path.join(root_path, celeb_lab)).mkdir(parents=True, exist_ok=True)

    print(Counter(lab_dict.values()).most_common(30))
    # print('Most common celeb: {} with {} images'.format(value, count))

    # target_folder = '../datasets/celebA'
    # for image in tqdm(os.listdir(os.path.join(root_path, 'img_align_celeba_png'))):
    #     copyfile(os.path.join(root_path, 'img_align_celeba_png', image), os.path.join(target_folder, lab_dict[image], image))


# create_celeb_folders('../../Datasets/CelebA')
