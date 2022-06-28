import os
import numpy as np
import fnmatch
import glob
import json
from PIL import Image
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torchvision import transforms
from collections import OrderedDict
import face_recognition.insightface_torch.backbones as InsightFaceResnetBackbone
import face_recognition.magface_torch.backbones as MFBackbone
from landmark_detection.face_alignment.face_alignment import FaceAlignment, LandmarksType
from landmark_detection.pytorch_face_landmark.models import mobilefacenet
from config import embedders_dict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class CustomDataset1(Dataset):
    def __init__(self, img_dir, celeb_lab_mapper, img_size, indices, shuffle=True, transform=None):
        self.img_dir = img_dir
        self.celeb_lab_mapper = {lab: i for i, lab in celeb_lab_mapper.items()}
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

        return image, self.img_names[idx], self.celeb_lab_mapper[celeb_lab]

    def get_image_names(self, indices):
        files_in_folder = get_nested_dataset_files(self.img_dir, self.celeb_lab_mapper.keys())
        files_in_folder = [item for sublist in files_in_folder for item in sublist]
        if indices is not None:
            files_in_folder = [files_in_folder[i] for i in indices]
        png_images = fnmatch.filter(files_in_folder, '*.png')
        jpg_images = fnmatch.filter(files_in_folder, '*.jpg')
        jpeg_images = fnmatch.filter(files_in_folder, '*.jpeg')
        return png_images + jpg_images + jpeg_images


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


def load_embedder(embedder_names, device):
    embedders = {}
    for embedder_name in embedder_names:
        backbone, head = embedder_name.split('_')
        weights_path = embedders_dict[backbone]['heads'][head]['weights_path']
        sd = torch.load(weights_path, map_location=device)
        if 'magface' in embedder_name:
            embedder = MFBackbone.IResNet(MFBackbone.IBasicBlock, layers=embedders_dict[backbone]['layers']).to(device).eval()
            sd = rewrite_weights_dict(sd['state_dict'])
        else:
            embedder = InsightFaceResnetBackbone.IResNet(InsightFaceResnetBackbone.IBasicBlock,
                                                         layers=embedders_dict[backbone]['layers']).to(device).eval()
        embedder.load_state_dict(sd)
        embedders[embedder_name] = embedder
    return embedders


def rewrite_weights_dict(sd):
    sd.pop('fc.weight')
    sd_new = OrderedDict()
    for key, value in sd.items():
        new_key = key.replace('features.module.', '')
        sd_new[new_key] = value
    return sd_new


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, current_dir='', init_patch=None):
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
        self.best_patch = init_patch
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
                print("Training stopped - early stopping", flush=True)
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
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving patch ...', flush=True)
        final_patch = torch.cat([patch.squeeze(0), self.alpha])
        transforms.ToPILImage()(final_patch).save(self.current_dir +
                                                  '/saved_patches' +
                                                  '/patch_' +
                                                  str(epoch) +
                                                  '.png', 'PNG')
        self.best_patch = patch
        self.val_loss_min = val_loss


@torch.no_grad()
def apply_mask(location_extractor, fxz_projector, img_batch, patch_rgb, patch_alpha=None, is_3d=False):
    preds = location_extractor(img_batch)
    img_batch_applied = fxz_projector(img_batch, preds, patch_rgb, uv_mask_src=patch_alpha, is_3d=is_3d)
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


def get_nested_dataset_files(img_dir, person_labs):
    files_in_folder = [glob.glob(os.path.join(img_dir, lab, '**/*.*g'), recursive=True) for lab in person_labs]
    return files_in_folder


def get_split_indices(img_dir, celeb_lab, num_of_images):
    dataset_nested_files = get_nested_dataset_files(img_dir, celeb_lab)

    nested_indices = [np.array(range(len(arr))) for i, arr in enumerate(dataset_nested_files)]
    nested_indices_continuous = [nested_indices[0]]
    for i, arr in enumerate(nested_indices[1:]):
        nested_indices_continuous.append(arr + nested_indices_continuous[i][-1] + 1)
    train_indices = np.array([np.random.choice(arr_idx, size=num_of_images, replace=False) for arr_idx in
                        nested_indices_continuous]).ravel()
    test_indices = list(set(list(range(nested_indices_continuous[-1][-1]))) - set(train_indices))

    return train_indices, test_indices


def get_train_loaders(config):
    train_indices, _ = get_split_indices(config.train_img_dir, config.celeb_lab, config.num_of_train_images)
    train_dataset_no_aug = CustomDataset1(img_dir=config.train_img_dir,
                                          celeb_lab_mapper=config.celeb_lab_mapper,
                                          img_size=config.img_size,
                                          indices=train_indices,
                                          transform=transforms.Compose(
                                              [transforms.Resize(config.img_size),
                                               transforms.ToTensor()]))
    train_dataset = CustomDataset1(img_dir=config.train_img_dir,
                                   celeb_lab_mapper=config.celeb_lab_mapper,
                                   img_size=config.img_size,
                                   indices=train_indices,
                                   transform=transforms.Compose(
                                       [transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
                                        transforms.Resize(config.img_size),
                                        transforms.ToTensor()]))
    train_no_aug_loader = DataLoader(train_dataset_no_aug, batch_size=config.train_batch_size)
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size)

    return train_no_aug_loader, train_loader


def get_test_loaders(config, dataset_names):
    emb_loaders = {}
    test_loaders = {}
    for dataset_name in dataset_names:
        emb_indices, test_indices = get_split_indices(config.test_img_dir[dataset_name],
                                                      config.test_celeb_lab[dataset_name],
                                                      config.test_num_of_images_for_emb)
        emb_dataset = CustomDataset1(img_dir=config.test_img_dir[dataset_name],
                                     celeb_lab_mapper=config.test_celeb_lab_mapper[dataset_name],
                                     img_size=config.img_size,
                                     indices=emb_indices,
                                     transform=transforms.Compose(
                                         [transforms.Resize(config.img_size),
                                          transforms.ToTensor()]))
        emb_loader = DataLoader(emb_dataset, batch_size=config.test_batch_size)
        emb_loaders[dataset_name] = emb_loader
        test_dataset = CustomDataset1(img_dir=config.test_img_dir[dataset_name],
                                      celeb_lab_mapper=config.test_celeb_lab_mapper[dataset_name],
                                      img_size=config.img_size,
                                      indices=test_indices,
                                      transform=transforms.Compose(
                                          [transforms.Resize(config.img_size),
                                           transforms.ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size)
        test_loaders[dataset_name] = test_loader

    return emb_loaders, test_loaders


@torch.no_grad()
def get_person_embedding(config, loader, celeb_lab, location_extractor, fxz_projector, embedders, device, include_others=False):
    print('Calculating persons embeddings {}...'.format('with mask' if include_others else 'without mask'), flush=True)
    embeddings_by_embedder = {}
    for embedder_name, embedder in embedders.items():
        person_embeddings = {i: torch.empty(0, device=device) for i in range(len(celeb_lab))}
        masks_path = [config.blue_mask_path, config.black_mask_path, config.white_mask_path]
        for img_batch, _, person_indices in tqdm(loader):
            img_batch = img_batch.to(device)
            if include_others:
                mask_path = masks_path[random.randint(0, 2)]
                mask_t = load_mask(config, mask_path, device)
                applied_batch = apply_mask(location_extractor, fxz_projector, img_batch, mask_t[:, :3], mask_t[:, 3], is_3d=True)
                img_batch = torch.cat([img_batch, applied_batch], dim=0)
                person_indices = person_indices.repeat(2)
            embedding = embedder(img_batch)
            for idx in person_indices.unique():
                relevant_indices = torch.nonzero(person_indices == idx, as_tuple=True)
                emb = embedding[relevant_indices]
                person_embeddings[idx.item()] = torch.cat([person_embeddings[idx.item()], emb], dim=0)
        final_embeddings = [person_emb.mean(dim=0).unsqueeze(0) for person_emb in person_embeddings.values()]
        final_embeddings = torch.stack(final_embeddings)
        embeddings_by_embedder[embedder_name] = final_embeddings
    return embeddings_by_embedder


def save_class_to_file(config, current_folder):
    with open(os.path.join(current_folder, 'config.json'), 'w') as config_file:
        d = dict(vars(config))
        d.pop('scheduler_factory')
        json.dump(d, config_file)


def get_patch(config):
    if config.initial_patch == 'random':
        patch = torch.rand((1, 3, config.patch_size[0], config.patch_size[1]), dtype=torch.float32)
    elif config.initial_patch == 'white':
        patch = torch.ones((1, 3, config.patch_size[0], config.patch_size[1]), dtype=torch.float32)
    elif config.initial_patch == 'black':
        patch = torch.zeros((1, 3, config.patch_size[0], config.patch_size[1]), dtype=torch.float32) + 0.01
    uv_face = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L'))
    patch = patch * uv_face
    patch.requires_grad_(True)
    return patch


def plot_train_val_loss(config, loss, loss_type):
    xticks = [x + 1 for x in range(len(loss))]
    plt.plot(xticks, loss, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel(loss_type)
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(config.current_dir + '/final_results/train_loss_' + loss_type.lower() + '_plt.png')
    plt.close()


def plot_separate_loss(config, train_losses_epoch, dist_losses, tv_losses):
    epochs = [x + 1 for x in range(len(train_losses_epoch))]
    weights = np.array([config.dist_weight, config.tv_weight])
    number_of_subplots = weights[weights > 0].astype(bool).sum()
    fig, axes = plt.subplots(nrows=1, ncols=number_of_subplots, figsize=(6 * number_of_subplots, 2 * number_of_subplots), squeeze=False)
    idx = 0
    for weight, train_loss, label in zip(weights, [dist_losses,  tv_losses], ['Distance loss', 'Total Variation loss']):
        if weight > 0:
            axes[0, idx].plot(epochs, train_loss, c='b', label='Train')
            axes[0, idx].set_xlabel('Epoch')
            axes[0, idx].set_ylabel('Loss')
            axes[0, idx].set_title(label)
            axes[0, idx].legend(loc='upper right')
            axes[0, idx].xaxis.set_major_locator(MaxNLocator(integer=True))
            idx += 1
    fig.tight_layout()
    plt.savefig(config.current_dir + '/final_results/separate_loss_plt.png')
    plt.close()