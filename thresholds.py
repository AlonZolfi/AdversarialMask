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
    with open(pairs_path, 'r') as f:
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
    N = 500
    # pairs_df = pd.read_csv(pairs_path, nrows=10000000)
    pairs_df = raw_to_df(pairs_path, img_root_dir)
    embedder = load_embedder('arcface',
                             os.path.join('face_recognition', 'arcface_torch', 'weights', 'arcface_resnet100.pth'),
                             device)
    results_df = pd.DataFrame(columns=['tp_mean', 'tp_std', 'fp_mean', 'fp_std', 'threshold', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall', 'f1', 'accuracy'])
    for seed in [42]:
        np.random.seed(seed)
        # sample_df = equal_sample_from_df(pairs_df, N)
        sample_df = pairs_df
        loader = DataLoader(dataset=PairsDataset(sample_df), shuffle=False, batch_size=8)
        sims = pd.Series(dtype=np.float64)
        cos_sim = CosineSimilarity()
        for img_batch1, img_batch2 in tqdm(loader):
            emb1 = embedder(img_batch1)
            emb2 = embedder(img_batch2)
            sim = cos_sim(emb1, emb2)
            sims = sims.append(pd.Series(np.round(sim.cpu().numpy(), 4)), ignore_index=True)
        sample_df['similarity'] = sims.values
        tp_mean = round(sample_df[sample_df.same].similarity.mean(), 4)
        tp_std = round(sample_df[sample_df.same].similarity.std(), 4)
        fp_mean = round(sample_df[~sample_df.same].similarity.mean(), 4)
        fp_std = round(sample_df[~sample_df.same].similarity.std(), 4)
        sample_df[sample_df.same].similarity.plot.kde()
        sample_df[~sample_df.same].similarity.plot.kde()
        plt.show()
        num_of_sigmas = 2
        threshold = round(tp_mean - num_of_sigmas * tp_std, 4)
        sample_df["prediction"] = False  # init
        idx = sample_df[sample_df.similarity >= threshold].index
        sample_df.loc[idx, 'prediction'] = True
        cm = confusion_matrix(sample_df.same.values, sample_df.prediction.values)
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        f1 = 2 * (precision * recall) / (precision + recall)
        results = pd.Series(data=[tp_mean, tp_std, fp_mean, fp_std, threshold, tn, fp, fn, tp, precision, recall, f1, accuracy], index=results_df.columns)
        results_df = results_df.append(results, ignore_index=True)
    results_df.to_csv('threshold.csv', index=False)


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
            self.tform.estimate(points[0], self.arcface_src)
            M = self.tform.params[0:2, :]
            image = cv2.warpAffine(image, M, (112, 112))
        cut = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
        cut_t = torch.from_numpy(cut).permute(2, 0, 1).type(torch.float32).to(device)
        return cut_t


calc_optimal_threshold('pairs.txt', 'datasets/lfw/')
