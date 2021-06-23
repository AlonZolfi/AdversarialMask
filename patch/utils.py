import os
import numpy as np
import fnmatch
from PIL import Image
import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from collections import OrderedDict
import face_recognition.arcface_torch.backbones.iresnet as AFBackbone
import face_recognition.magface_torch.backbones as MFBackbone

class CustomDataset(Dataset):
    def __init__(self, img_dir, lab_dir, max_lab, img_size, shuffle=True, transform=None):
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.img_size = img_size
        self.shuffle = shuffle
        self.img_names = self.get_image_names()
        self.img_paths = self.get_image_paths()
        self.lab_paths = self.get_lab_paths()
        self.max_n_labels = max_lab
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):  # check to see if label file contains data.
            label = np.loadtxt(lab_path, ndmin=1)
        else:
            label = np.ones([1])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        # image, label = self.pad_and_scale(image, label)
        if self.transform:
            image = self.transform(image)

        label = self.pad_lab(label)
        return image, label

    def get_image_names(self):
        png_images = fnmatch.filter(os.listdir(self.img_dir), '*.png')
        jpg_images = fnmatch.filter(os.listdir(self.img_dir), '*.jpg')
        n_png_images = len(png_images)
        n_jpg_images = len(jpg_images)
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(self.lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        return png_images + jpg_images

    def get_image_paths(self):
        img_paths = []
        for img_name in self.img_names:
            img_paths.append(os.path.join(self.img_dir, img_name))
        return img_paths

    def get_lab_paths(self):
        lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            lab_paths.append(lab_path)
        return lab_paths

    def pad_and_scale(self, img, lab):
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(255, 255, 255))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(255, 255, 255))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
        resize = transforms.Resize(self.img_size)
        padded_img = resize(padded_img)  # choose here

        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if pad_size > 0:
            padded_lab = F.pad(lab, [0, 0, 0, pad_size], value=-1)
        else:
            padded_lab = lab
        return padded_lab


class CustomDataset1(Dataset):
    def __init__(self, img_dir, img_size, shuffle=True, transform=None):
        self.img_dir = img_dir
        self.img_size = img_size
        self.shuffle = shuffle
        self.img_names = self.get_image_names()
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.img_names[idx]

    def get_image_names(self):
        png_images = fnmatch.filter(os.listdir(self.img_dir), '*.png')
        jpg_images = fnmatch.filter(os.listdir(self.img_dir), '*.jpg')
        return png_images + jpg_images

    def get_image_paths(self):
        img_paths = []
        for img_name in self.img_names:
            img_paths.append(os.path.join(self.img_dir, img_name))
        return img_paths

    def pad_and_scale(self, img, lab):
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(255, 255, 255))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(255, 255, 255))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
        resize = transforms.Resize(self.img_size)
        padded_img = resize(padded_img)  # choose here

        return padded_img, lab


class SplitDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, val_split, test_split, shuffle, batch_size, *args, **kwargs):
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        val_split = int(np.floor(val_split * dataset_size))
        test_split = int(np.floor(test_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)

        train_indices = indices[val_split + test_split:]
        val_indices = indices[:val_split]
        test_indices = indices[val_split:val_split + test_split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler)
        validation_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=valid_sampler)
        test_loader = DataLoader(self.dataset, sampler=test_sampler)

        return train_loader, validation_loader, test_loader


def load_embedder(embedder_name, weights_path, device):
    embedder_name = embedder_name.lower()
    if embedder_name == 'vggface2':
        embedder = InceptionResnetV1(classify=False, pretrained='vggface2', device=device).eval()
    elif embedder_name == 'arcface':
        embedder = AFBackbone.IResNet(AFBackbone.IBasicBlock, [3, 13, 30, 3]).to(device).eval()
        embedder.load_state_dict(torch.load(weights_path, map_location=device))
    elif embedder_name == 'magface':
        embedder = MFBackbone.IResNet(MFBackbone.IBasicBlock, [3, 13, 30, 3]).to(device).eval()
        sd = torch.load(weights_path, map_location=device)['state_dict']
        sd_new = rewrite_weights_dict(sd)
        embedder.load_state_dict(sd_new)
    else:
        raise Exception('Embedder cannot be loaded')
    return embedder


def rewrite_weights_dict(sd):
    sd.pop('fc.weight')
    sd_new = OrderedDict()
    for key, value in sd.items():
        new_key = key.replace('features.module.', '')  # .replace('module.', '')
        sd_new[new_key] = value
    return sd_new


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, current_dir=''):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.current_dir = current_dir
        self.best_patch = None
        self.alpha = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L'))

    def __call__(self, val_loss, patch, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, patch, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}', flush=True)
            if self.counter >= self.patience:
                print("Training stopped - early stopping")
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, patch, epoch)
            self.counter = 0
        return False

    def save_checkpoint(self, val_loss, patch, epoch):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving patch ...')
        final_patch = torch.cat([patch.squeeze(0), self.alpha])
        transforms.ToPILImage()(final_patch).save(self.current_dir +
                                                  '/saved_patches' +
                                                  '/patch_' +
                                                  str(epoch) +
                                                  '.png', 'PNG')
        self.best_patch = patch
        self.val_loss_min = val_loss


