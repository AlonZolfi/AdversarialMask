import os
import numpy as np
import fnmatch
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from collections import OrderedDict
import face_recognition.arcface_torch.backbones.iresnet as AFBackbone
import face_recognition.magface_torch.backbones as MFBackbone
from landmark_detection.face_alignment.face_alignment import FaceAlignment, LandmarksType
from landmark_detection.pytorch_face_landmark.models import mobilefacenet


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
    def __init__(self, img_dir, celeb_lab, img_size, indices, shuffle=True, transform=None):
        self.img_dir = img_dir
        self.celeb_labs = {lab: i for i, lab in enumerate(celeb_lab)}
        self.img_size = img_size
        self.shuffle = shuffle
        self.img_names = self.get_image_names(indices)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = self.img_names[idx]
        celeb_lab = img_path.split(os.path.sep)[-2]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, self.img_names[idx], self.celeb_labs[celeb_lab]

    def get_image_names(self, indices):
        files_in_folder = get_dataset_files(self.img_dir, self.celeb_labs.keys())
        files_in_folder = [files_in_folder[i] for i in indices]
        png_images = fnmatch.filter(files_in_folder, '*.png')
        jpg_images = fnmatch.filter(files_in_folder, '*.jpg')
        return png_images + jpg_images


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


@torch.no_grad()
def apply_mask(location_extractor, fxz_projector, img_batch, patch_rgb, patch_alpha=None):
    preds = location_extractor(img_batch)
    img_batch_applied = fxz_projector(img_batch, preds, patch_rgb, uv_mask_src=patch_alpha)
    return img_batch_applied


@torch.no_grad()
def load_mask(config, mask_path, device):
    transform = transforms.Compose([transforms.Resize(config.patch_size), transforms.ToTensor()])
    img = Image.open(mask_path)
    img_t = transform(img).unsqueeze(0).to(device)
    return img_t


def get_landmark_detector(config, device):
    landmark_detector_type = config.landmark_detector_type
    if landmark_detector_type == 'face_alignment':
        return FaceAlignment(LandmarksType._2D, device=str(device))
    elif landmark_detector_type == 'mobilefacenet':
        model = mobilefacenet.MobileFaceNet([112, 112], 136).eval().to(device)
        sd = torch.load('../landmark_detection/pytorch_face_landmark/weights/mobilefacenet_model_best.pth.tar', map_location=device)['state_dict']
        model.load_state_dict(sd)
        return model


def get_dataset_files(img_dir, person_labs):
    files_in_folder = [glob.glob(os.path.join(img_dir, lab, '**/*.*g'), recursive=True) for lab in person_labs]
    files_in_folder = [item for sublist in files_in_folder for item in sublist]
    return files_in_folder


def get_split_indices(config):
    dataset_size = len(get_dataset_files(config.img_dir, config.celeb_lab))
    indices = list(range(dataset_size))
    if config.shuffle:
        np.random.shuffle(indices)

    if config.num_of_train_images > 0:
        num_of_train_images = config.num_of_train_images * len(config.celeb_lab)
        train_indices = indices[:num_of_train_images]
        val_indices = []
        test_indices = indices[num_of_train_images:]
    else:
        val_split = int(np.floor(config.val_split * dataset_size))
        test_split = int(np.floor(config.test_split * dataset_size))
        train_indices = indices[val_split + test_split:]
        val_indices = indices[:val_split]
        test_indices = indices[val_split:val_split + test_split]

    return train_indices, val_indices, test_indices


def get_split_indices_real_images(config):
    dataset_size = len(os.listdir(config.test_img_dir))
    indices = list(range(dataset_size))
    remaining_indices_ratio = 1 - config.val_split - config.test_split
    val_split = int(np.floor((config.val_split + remaining_indices_ratio*config.val_split) * dataset_size))
    if config.shuffle:
        np.random.shuffle(indices)

    val_indices = indices[:val_split]
    test_indices = indices[val_split:]

    return val_indices, test_indices


def get_loaders(config):
    train_indices, val_indices, test_indices = get_split_indices(config)
    train_dataset_no_aug = CustomDataset1(img_dir=config.img_dir,
                                          celeb_lab=config.celeb_lab,
                                          img_size=config.img_size,
                                          indices=train_indices,
                                          transform=transforms.Compose(
                                              [transforms.Resize(config.img_size),
                                               transforms.ToTensor()]))
    train_dataset = CustomDataset1(img_dir=config.img_dir,
                                   celeb_lab=config.celeb_lab,
                                   img_size=config.img_size,
                                   indices=train_indices,
                                   transform=transforms.Compose(
                                       [transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
                                        # transforms.RandomHorizontalFlip(),
                                        transforms.Resize(config.img_size),
                                        transforms.ToTensor()]))
    val_dataset = CustomDataset1(img_dir=config.img_dir,
                                 celeb_lab=config.celeb_lab,
                                 img_size=config.img_size,
                                 indices=val_indices,
                                 transform=transforms.Compose(
                                     [transforms.Resize(config.img_size), transforms.ToTensor()]))
    test_dataset = CustomDataset1(img_dir=config.img_dir,
                                  celeb_lab=config.celeb_lab,
                                  img_size=config.img_size,
                                  indices=test_indices,
                                  transform=transforms.Compose(
                                      [transforms.Resize(config.img_size), transforms.ToTensor()]))
    train_no_aug_loader = DataLoader(train_dataset_no_aug, batch_size=config.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    validation_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset)

    return train_no_aug_loader, train_loader, validation_loader, test_loader


def get_loaders_real_images(config):
    val_indices, test_indices = get_split_indices_real_images(config)
    train_indices = range(len(os.listdir(config.train_img_dir)))
    train_dataset_no_aug = CustomDataset1(img_dir=config.train_img_dir,
                                          img_size=config.img_size,
                                          indices=train_indices,
                                          transform=transforms.Compose(
                                              [transforms.Resize(config.img_size),
                                               transforms.ToTensor()]))
    train_dataset = CustomDataset1(img_dir=config.train_img_dir,
                                   img_size=config.img_size,
                                   indices=train_indices,
                                   transform=transforms.Compose(
                                       [transforms.ColorJitter(brightness=0.15, contrast=0.15, hue=0.15),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(degrees=(-15, 15)),
                                        transforms.Resize(config.img_size),
                                        transforms.ToTensor()]))
    val_dataset = CustomDataset1(img_dir=config.test_img_dir,
                                 img_size=config.img_size,
                                 indices=val_indices,
                                 transform=transforms.Compose(
                                     [transforms.Resize(config.img_size), transforms.ToTensor()]))
    test_dataset = CustomDataset1(img_dir=config.test_img_dir,
                                  img_size=config.img_size,
                                  indices=test_indices,
                                  transform=transforms.Compose(
                                      [transforms.Resize(config.img_size), transforms.ToTensor()]))
    train_no_aug_loader = DataLoader(train_dataset_no_aug, batch_size=config.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    validation_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset)

    return train_no_aug_loader, train_loader, validation_loader, test_loader


def normalize_batch(adv_mask_class, img_batch):
    preds = adv_mask_class.location_extractor(img_batch)
    return adv_mask_class.normalize_arcface(img_batch, preds)


@torch.no_grad()
def get_person_embedding(config, loader, adv_mask_class, device, include_others=False):
    person_embeddings = {i: torch.empty(0, device=device) for i in range(len(config.celeb_lab))}
    masks = [None]
    if include_others:
        masks.extend([config.blue_mask_path, config.black_mask_path, config.white_mask_path])
    for mask_path in masks:
        if mask_path is not None:
            mask_t = load_mask(config, mask_path, device)
        for img_batch, _, idx in loader:
            img_batch = img_batch.to(device)
            if mask_path is not None:
                img_batch = apply_mask(adv_mask_class.location_extractor, adv_mask_class.fxz_projector, img_batch, mask_t[:, :3], mask_t[:, 3])
            embedding = adv_mask_class.embedder(img_batch)
            person_embeddings[idx.item()] = torch.cat([person_embeddings[idx.item()], embedding], dim=0)
    final_embeddings = [person_emb.mean(dim=0).unsqueeze(0) for person_emb in person_embeddings.values()]
    final_embeddings = torch.stack(final_embeddings)
    return final_embeddings


def save_class_to_file(config, current_folder):
    import json
    with open(os.path.join(current_folder, 'config.json'), 'w') as config_file:
        d = dict(vars(config))
        d.pop('scheduler_factory')
        json.dump(d, config_file)


# def load_landmarks(self):
#     print('Starting landmark prediction')
#     landmarks_dict = {}
#     folder = self.config.landmark_folder
#     Path(self.config.landmark_folder).mkdir(parents=True, exist_ok=True)
#     lm_cur_files = os.listdir(folder)
#     for loader in [self.train_loader, self.val_loader, self.test_loader]:
#         for image, img_names in loader:
#             for img_idx in range(len(img_names)):
#                 file_name = img_names[img_idx].split('.')[0] + '.pt'
#                 if (file_name not in lm_cur_files) or self.config.recreate_landmarks:
#                     img = image[img_idx].to(device).unsqueeze(0)
#                     preds = self.location_extractor(img)
#                     torch.save(preds, os.path.join(folder, file_name))
#                 else:
#                     preds = torch.load(os.path.join(folder, file_name), map_location=device)
#                 landmarks_dict[file_name.replace('.pt', '')] = preds
#     del self.location_extractor
#     torch.cuda.empty_cache()
#     print('Finished landmark prediction')
#     return landmarks_dict
#
#
# def get_batch_landmarks(self, img_names):
#     preds = torch.empty(0, device=device)
#     for img_name in img_names:
#         dict_key_name = img_name.split('.')[0]
#         preds = torch.cat([preds, self.preds[dict_key_name]], dim=0)
#     return preds