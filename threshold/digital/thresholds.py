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
                             os.path.join('../../face_recognition', 'arcface_torch', 'weights', 'arcface_resnet100.pth'),
                             device)
    loader = DataLoader(dataset=PairsDataset(pairs_df), shuffle=False, batch_size=64)
    sims = pd.Series(dtype=np.float64)
    cos_sim = CosineSimilarity()
    for img_batch1, img_batch2 in tqdm(loader):
        emb1 = embedder(img_batch1)
        emb2 = embedder(img_batch2)
        sim = cos_sim(emb1, emb2)
        sims = sims.append(pd.Series(sim.cpu().numpy()), ignore_index=True)
    pairs_df['similarity'] = sims.values
    pairs_df.to_csv('pairs_with_sim.csv', index=False)
    y_true = pairs_df.same.astype(int)
    preds = pairs_df.similarity.astype(int)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, preds)
    auc_score = metrics.roc_auc_score(y_true, preds)
    plot_roc(tpr, fpr, auc_score)
    threshold_idx = np.argmin(np.square(tpr - 0.95))
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
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


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


calc_optimal_threshold('lfw_pairs.txt', '../../datasets/lfw/')


def filter_images(root_path):
    imgs_full_path = glob.glob(root_path + '/**/*.jpg', recursive=True) + \
                     glob.glob(root_path + '/**/*.png', recursive=True)
    mtcnn = MTCNN()
    bad_imgs = []
    for img_path in imgs_full_path:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        _, _, points = mtcnn.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), landmarks=True)
        if points is None:
            bad_imgs.append(img_path)
    print(len(bad_imgs))
    with open('bad_images.txt', 'w') as f:
        for bad_img in bad_imgs:
            f.write(bad_img)


# filter_images('datasets/lfw')
