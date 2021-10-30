import random

import torch
import pandas as pd
import itertools
import glob
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN
import numpy as np
import cv2
from skimage import transform as trans
from utils import load_embedder
from PIL import Image
from torch.nn import CosineSimilarity
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn import metrics
import torch
from torchvision import transforms
from nn_modules import LandmarkExtractor, FaceXZooProjector
from PIL import Image
import os
from pathlib import Path
import utils
from config import BaseConfiguration


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_all_pairs(root_path, output_path):
    imgs_full_path = glob.glob(root_path + '/**/*.jpg', recursive=True) + \
                     glob.glob(root_path + '/**/*.png', recursive=True)
    pairs_l = list(itertools.combinations(imgs_full_path, 2))
    pairs_df = pd.DataFrame(pairs_l, columns=['path1', 'path2'])
    pairs_df['same'] = pairs_df.apply(lambda x: os.path.dirname(x['path1']) == os.path.dirname(x['path2']), axis=1)
    pairs_df.to_csv(output_path + '.csv', index=False)


# get_all_pairs('datasets/lfw', 'lfw_pairs')

def raw_to_df(pairs_path, img_root_dir):
    df = pd.DataFrame(columns=['path1', 'path2', 'same'])
    with open(pairs_path, 'r') as f, open('../threshold/digital/bad_images.txt', 'r') as bad:
        bad_images = bad.read().splitlines()
        for line in f:
            new_row = None
            splitted_line = line.strip('\n').split('\t')
            if len(splitted_line) == 3:
                image_1 = img_root_dir + splitted_line[0] + '/' + splitted_line[0] + greater_than_ten(splitted_line[1]) + '.jpg'
                image_2 = img_root_dir + splitted_line[0] + '/' + splitted_line[0] + greater_than_ten(splitted_line[2]) + '.jpg'
                new_row = pd.Series([image_1, image_2, True], index=df.columns)

            elif len(splitted_line) == 4:
                image_1 = img_root_dir + splitted_line[0] + '/' + splitted_line[0] + greater_than_ten(splitted_line[1]) + '.jpg'
                image_2 = img_root_dir + splitted_line[2] + '/' + splitted_line[2] + greater_than_ten(splitted_line[3]) + '.jpg'
                new_row = pd.Series([image_1, image_2, False], index=df.columns)

            if new_row is not None:
                if new_row[0] not in bad_images and new_row[1] not in bad_images:
                    df = df.append(new_row, ignore_index=True)
    return df


def greater_than_ten(x):
    num = int(x)
    str_returned = ''
    if num >= 100:
        str_returned = '_0' + x
    elif num >= 10:
        str_returned = '_00' + x
    else:
        str_returned = '_000' + x

    return str_returned


def stratified_sample_from_df(pairs_df, N):
    return pairs_df.groupby('same', group_keys=False).apply(
        lambda x: x.sample(int(np.rint(N * len(x) / len(pairs_df))))).sample(frac=1).reset_index(drop=True)


def equal_sample_from_df(pairs_df, N):
    return pairs_df.groupby('same', group_keys=False).apply(lambda x: x.sample(frac=N / x.shape[0])).reset_index(
        drop=True)


@torch.no_grad()
def calc_optimal_threshold(pairs_path, img_root_dir):
    calc_new = False
    cfg = BaseConfiguration()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    face_landmark_detector = utils.get_landmark_detector(cfg, device)
    location_extractor = LandmarkExtractor(device, face_landmark_detector, cfg.img_size).to(device)
    fxz_projector = FaceXZooProjector(device, cfg.img_size, cfg.patch_size).to(device)
    mask_t = utils.load_mask(cfg, os.path.join('../data/masks/blue.png'), device)
    if calc_new:
        pairs_df = raw_to_df(pairs_path, img_root_dir)
        embedder = load_embedder(['resnet100_arcface'], device)
        loader = DataLoader(dataset=PairsDataset(pairs_df), shuffle=False, batch_size=64)
        sims = pd.Series(dtype=np.float64)
        cos_sim = CosineSimilarity()
        for img_batch1, img_batch2 in tqdm(loader):
            if random.randint(0, 10) > 5:
                img_batch1 = utils.apply_mask(location_extractor, fxz_projector, img_batch1, mask_t[:, :3], mask_t[:, 3], is_3d=True)
            emb1 = embedder['resnet100_arcface'](img_batch1)
            emb2 = embedder['resnet100_arcface'](img_batch2)
            sim = cos_sim(emb1, emb2)
            sims = sims.append(pd.Series(sim.cpu().numpy()), ignore_index=True)
        pairs_df['similarity'] = sims.values
        pairs_df.to_csv('pairs_with_sim.csv', index=False)
        return
    else:
        pairs_df = pd.read_csv('pairs_with_sim.csv')
    y_true = pairs_df.same.values.astype(int)
    # preds = (pairs_df.similarity.values + 1) / 2
    preds = pairs_df.similarity.values
    # import warnings
    # warnings.filterwarnings("ignore")
    #
    # only_ones = pairs_df[pairs_df['same']==0]
    # for threshold in np.arange(-1, 1, 0.01):
    #     only_ones['decision'] = np.where(only_ones['similarity'] > threshold, 1, 0)
    #     cnt = len(only_ones[(only_ones['decision'] == 1) & (only_ones['same']==0)])
    #     far = cnt / len(only_ones) * 100
    #     print(threshold, far)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, preds)
    auc_score = metrics.roc_auc_score(y_true, preds)
    plot_roc(tpr, fpr, auc_score)
    threshold_idx = np.argmin(np.square(fpr - 0.01))
    threshold = thresholds[threshold_idx]
    print('threshold: ', threshold)
    calc_final_results(pairs_df, threshold)


def calc_final_results(df, threshold):
    df["prediction"] = False  # init
    idx = df[df.similarity >= threshold].index
    df.loc[idx, 'prediction'] = True
    cm = confusion_matrix(df.same.values.astype(bool), df.prediction.values)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    f1 = 2 * (precision * recall) / (precision + recall)
    print('precision: {}, recall: {}, accuracy: {}, f1: {}'.format(precision, recall, accuracy, f1))


def plot_roc(tpr, fpr, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('roc.png')


class PairsDataset(Dataset):
    def __init__(self, pairs_df) -> None:
        super().__init__()
        self.pairs_df = pairs_df
        self.tform = trans.SimilarityTransform()
        self.mtcnn = MTCNN()
        self.arcface_src = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32)

    def __getitem__(self, index):
        image1 = self.get_image_tensor(index, col_name='path1')
        image2 = self.get_image_tensor(index, col_name='path2')
        return image1, image2

    def __len__(self):
        return self.pairs_df.shape[0]

    def get_image_tensor(self, index, col_name):
        img_path = os.path.join(self.pairs_df.iloc[index][col_name])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        _, _, points = self.mtcnn.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), landmarks=True)
        if points is not None:
            p = points[0]
            if len(points) > 1:
                p_idx = np.argmin((points[:, 2, 0] - image.shape[1]/2)**2 + (points[:, 2, 1] - image.shape[0]/2)**2)
                p = points[p_idx]
            self.tform.estimate(p, self.arcface_src)
            M = self.tform.params[0:2, :]
            image = cv2.warpAffine(image, M, (112, 112))
        cut = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
        cut_t = torch.from_numpy(cut).permute(2, 0, 1).type(torch.float32).to(device)
        # if len(points) > 1:
        #     from torchvision import transforms
        #     transforms.ToPILImage()(cut_t).show()
        return cut_t


calc_optimal_threshold('../threshold/digital/lfw_pairs.txt', '../datasets/lfw/')


def filter_images(root_path):
    imgs_full_path = glob.glob(root_path + '/**/*.jpg', recursive=True) + \
                     glob.glob(root_path + '/**/*.png', recursive=True)
    mtcnn = MTCNN()
    bad_imgs = []
    for img_path in imgs_full_path:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        _, _, points = mtcnn.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), landmarks=True)
        if points is None or len(points) > 1:
            bad_imgs.append(img_path)
    print(len(bad_imgs))
    with open('digital/bad_images.txt', 'w') as f:
        for bad_img in bad_imgs:
            f.write(bad_img)
            f.write('\n')


# filter_images('datasets/lfw')
