import sys
import os
import random
import warnings
import time
import datetime
from pathlib import Path
import pickle

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.nn import CosineSimilarity, L1Loss, MSELoss
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader

from PIL import Image

from config import patch_config_types
from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation
from test import Evaluator
from utils import SplitDataset, CustomDataset1, load_embedder, EarlyStopping
from landmark_detection.face_alignment.face_alignment import FaceAlignment, LandmarksType
from landmark_detection.pytorch_face_landmark.models import mobilefacenet

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device is {}'.format(device))

if sys.base_prefix.__contains__('home/zolfi'):
    sys.path.append('/home/zolfi/AdversarialMask/patch')
    sys.path.append('/home/zolfi/AdversarialMask/arcface_torch')
    sys.path.append('/home/zolfi/AdversarialMask/face-alignment')
    sys.path.append('/home/zolfi/AdversarialMask/prnet')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def set_random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_random_seed(seed_value=42)


class AdversarialMask:
    def __init__(self, mode):
        self.config = patch_config_types[mode]()
        # custom_dataset = CustomDataset(img_dir=self.config.img_dir_train_val,
        #                                lab_dir=self.config.lab_dir_train_val,
        #                                max_lab=self.config.max_labels_per_img,
        #                                img_size=self.config.img_size,
        #                                transform=transforms.Compose(
        #                                    [transforms.Resize(self.config.img_size), transforms.ToTensor()]))
        self.train_loader, self.val_loader, self.test_loader = self.get_loaders()

        self.embedder = load_embedder(self.config.embedder_name, self.config.embedder_weights_path, device)

        face_landmark_detector = self.get_landmark_detector(device)
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, self.config.img_size).to(device)
        # self.preds = self.load_landmarks()
        self.fxz_projector = FaceXZooProjector(device, self.config.img_size, self.config.patch_size).to(device)
        self.total_variation = TotalVariation(device).to(device)
        self.dist_loss = self.get_loss().to(device)

        self.train_losses = []
        self.dist_losses = []
        self.tv_losses = []
        self.val_losses = []

        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        if 'SLURM_JOBID' not in os.environ.keys():
            self.current_dir = "experiments/" + month_name + '/' + time.strftime("%d-%m-%Y") + '_' + time.strftime(
                "%H-%M-%S")
        else:
            self.current_dir = "experiments/" + month_name + '/' + time.strftime("%d-%m-%Y") + '_' + os.environ['SLURM_JOBID']
        self.create_folders()
        self.target_embedding = self.get_person_embedding()
        self.best_patch = None

    def create_folders(self):
        Path('/'.join(self.current_dir.split('/')[:2])).mkdir(parents=True, exist_ok=True)
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/final_results').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/saved_patches').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/losses').mkdir(parents=True, exist_ok=True)

    def train(self):
        adv_patch_cpu = self.get_patch(self.config.initial_patch)
        self.initial_patch = adv_patch_cpu.clone()
        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        early_stop = EarlyStopping(current_dir=self.current_dir, patience=self.config.es_patience)
        epoch_length = len(self.train_loader)
        for epoch in range(self.config.epochs):
            train_loss = 0.0
            dist_loss = 0.0
            tv_loss = 0.0
            progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=epoch_length)
            prog_bar_desc = 'train-loss: {:.6}, dist-loss: {:.6}, tv-loss: {:.6}, lr: {:.6}'
            for i_batch, (img_batch, img_names) in progress_bar:
                (b_loss, sep_loss), vars = self.forward_step(img_batch, adv_patch_cpu, img_names)

                train_loss += b_loss.item()
                dist_loss += sep_loss[0].item()
                tv_loss += sep_loss[1].item()

                optimizer.zero_grad()
                b_loss.backward()
                optimizer.step()

                adv_patch_cpu.data.clamp_(0, 1)

                progress_bar.set_postfix_str(prog_bar_desc.format(train_loss / (i_batch + 1),
                                                                  dist_loss / (i_batch + 1),
                                                                  tv_loss / (i_batch + 1),
                                                                  optimizer.param_groups[0]["lr"]))

                if i_batch + 1 == epoch_length:
                    self.calc_validation(adv_patch_cpu)
                    self.save_losses(epoch_length, train_loss, dist_loss, tv_loss)
                    prog_bar_desc += ', val-loss: {:.6}'
                    progress_bar.set_postfix_str(prog_bar_desc.format(self.train_losses[-1],
                                                                      self.dist_losses[-1],
                                                                      self.tv_losses[-1],
                                                                      optimizer.param_groups[0]["lr"],
                                                                      self.val_losses[-1]))
                for var in vars + sep_loss:
                    del var
                del b_loss
                torch.cuda.empty_cache()

            if early_stop(self.val_losses[-1], adv_patch_cpu, epoch):
                self.best_patch = adv_patch_cpu
                break

            scheduler.step(self.val_losses[-1])

        self.best_patch = early_stop.best_patch
        self.save_final_objects()
        self.plot_train_val_loss()
        self.plot_separate_loss()

    def loss_fn(self, patch_emb, tv_loss):
        distance_loss = self.config.dist_weight * torch.mean(F.relu(self.dist_loss(patch_emb, self.target_embedding)))
        tv_loss = self.config.tv_weight * tv_loss
        total_loss = distance_loss + tv_loss
        return total_loss, [distance_loss, tv_loss]

    def get_patch(self, p_type):
        if p_type == 'stripes':
            patch = torch.zeros((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32)
            for i in range(patch.size()[2]):
                patch[0, 0, i, :] = random.randint(0, 255) / 255
                patch[0, 1, i, :] = random.randint(0, 255) / 255
                patch[0, 2, i, :] = random.randint(0, 255) / 255
        elif p_type =='l_stripes':
            patch = torch.zeros((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32)
            for i in range(int(patch.size()[2]/16)):
                patch[0, 0, int(patch.size()[2]/16)*i:int(patch.size()[2]/16)*(i+1), :] = random.randint(0, 255) / 255
                patch[0, 1, int(patch.size()[2]/16)*i:int(patch.size()[2]/16)*(i+1), :] = random.randint(0, 255) / 255
                patch[0, 2, int(patch.size()[2]/16)*i:int(patch.size()[2]/16)*(i+1), :] = random.randint(0, 255) / 255
        elif p_type == 'random':
            patch = torch.rand((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32)
        elif p_type == 'white':
            patch = torch.ones((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32)
        elif p_type == 'body':
            patch = torch.zeros((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32)
            patch[:, 0] = 1
            patch[:, 1] = 0.8
            patch[:, 2] = 0.58
        uv_face = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L'))
        patch = patch * uv_face
        patch.requires_grad_(True)
        # transforms.ToPILImage()(patch.squeeze(0) * uv_face).save('random.png')
        # transforms.ToPILImage()(patch.squeeze(0) * uv_face).show()
        return patch

    def forward_step(self, img_batch, adv_patch_cpu, img_names, train=True):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            img_batch = img_batch.to(device)
            adv_patch = adv_patch_cpu.to(device)

            preds = self.location_extractor(img_batch)
            # preds = self.get_batch_landmarks(img_names)
            img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch)
            # img_batch_applied = torch.nn.functional.interpolate(img_batch_applied, 112)
            patch_emb = self.embedder(img_batch_applied)

            tv_loss = self.total_variation(adv_patch, train)
            loss = self.loss_fn(patch_emb, tv_loss)

            return loss, [img_batch, adv_patch, img_batch_applied, patch_emb, tv_loss]

    def calc_validation(self, adv_patch_cpu):
        val_loss = 0.0
        with torch.no_grad():
            for img_batch, img_names in self.val_loader:
                img_batch = img_batch.to(device)
                (loss, _), _ = self.forward_step(img_batch, adv_patch_cpu, img_names, train=False)
                val_loss += loss.item()

                del img_batch, loss
                torch.cuda.empty_cache()
        val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(val_loss)

    def save_losses(self, epoch_length, train_loss, dist_loss, tv_loss):
        train_loss /= epoch_length
        dist_loss /= epoch_length
        tv_loss /= epoch_length
        self.train_losses.append(train_loss)
        self.dist_losses.append(dist_loss)
        self.tv_losses.append(tv_loss)

    def plot_train_val_loss(self):
        epochs = [x + 1 for x in range(len(self.train_losses))]
        plt.plot(epochs, self.train_losses, 'b', label='Training loss')
        plt.plot(epochs, self.val_losses, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(self.current_dir + '/final_results/train_val_loss_plt.png')
        plt.close()

    def plot_separate_loss(self):
        epochs = [x + 1 for x in range(len(self.train_losses))]
        weights = np.array([self.config.dist_weight, self.config.tv_weight])
        number_of_subplots = weights[weights > 0].astype(bool).sum()
        fig, axes = plt.subplots(nrows=1, ncols=number_of_subplots, figsize=(6 * number_of_subplots, 2 * number_of_subplots), squeeze=False)
        idx = 0
        for weight, train_loss, label in zip(weights, [self.dist_losses,  self.tv_losses], ['Distance loss', 'Total Variation loss']):
            if weight > 0:
                axes[0, idx].plot(epochs, train_loss, c='b', label='Train')
                axes[0, idx].set_xlabel('Epoch')
                axes[0, idx].set_ylabel('Loss')
                axes[0, idx].set_title(label)
                axes[0, idx].legend(loc='upper right')
                axes[0, idx].xaxis.set_major_locator(MaxNLocator(integer=True))
                idx += 1
        fig.tight_layout()
        plt.savefig(self.current_dir + '/final_results/separate_loss_plt.png')
        plt.close()

    def save_final_objects(self):
        alpha = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L'))
        final_patch = torch.cat([self.best_patch.squeeze(0), alpha])
        transforms.ToPILImage()(final_patch.squeeze(0)).save(
            self.current_dir + '/final_results/final_patch.png', 'PNG')
        torch.save(self.best_patch, self.current_dir + '/final_results/final_patch_raw.pt')

        with open(self.current_dir + '/losses/train_losses', 'wb') as fp:
            pickle.dump(self.train_losses, fp)
        with open(self.current_dir + '/losses/val_losses', 'wb') as fp:
            pickle.dump(self.val_losses, fp)
        with open(self.current_dir + '/losses/dist_losses', 'wb') as fp:
            pickle.dump(self.dist_losses, fp)
        with open(self.current_dir + '/losses/tv_losses', 'wb') as fp:
            pickle.dump(self.tv_losses, fp)

    def get_person_embedding(self):
        with torch.no_grad():
            person_embeddings = torch.empty(0, device=device)
            for img_batch, _ in self.train_loader:
                img_batch = img_batch.to(device)
                embedding = self.embedder(img_batch)
                person_embeddings = torch.cat([person_embeddings, embedding], dim=0)

                del img_batch
                torch.cuda.empty_cache()
            return person_embeddings.mean(dim=0).unsqueeze(0)

    def load_landmarks(self):
        print('Starting landmark prediction')
        landmarks_dict = {}
        folder = self.config.landmark_folder
        Path(self.config.landmark_folder).mkdir(parents=True, exist_ok=True)
        lm_cur_files = os.listdir(folder)
        for loader in [self.train_loader, self.val_loader, self.test_loader]:
            for image, img_names in loader:
                for img_idx in range(len(img_names)):
                    file_name = img_names[img_idx].split('.')[0] + '.pt'
                    if (file_name not in lm_cur_files) or self.config.recreate_landmarks:
                        img = image[img_idx].to(device).unsqueeze(0)
                        preds = self.location_extractor(img)
                        torch.save(preds, os.path.join(folder, file_name))
                    else:
                        preds = torch.load(os.path.join(folder, file_name), map_location=device)
                    landmarks_dict[file_name.replace('.pt', '')] = preds
        del self.location_extractor
        torch.cuda.empty_cache()
        print('Finished landmark prediction')
        return landmarks_dict

    def get_batch_landmarks(self, img_names):
        preds = torch.empty(0, device=device)
        for img_name in img_names:
            dict_key_name = img_name.split('.')[0]
            preds = torch.cat([preds, self.preds[dict_key_name]], dim=0)
        return preds

    def get_loss(self):
        if self.config.dist_loss_type == 'cossim':
            return CosineSimilarity()
        elif self.config.dist_loss_type == 'L2':
            return MSELoss()
        elif self.config.dist_loss_type == 'L1':
            return L1Loss()
        raise ValueError()

    def get_landmark_detector(self, device):
        landmark_detector_type = self.config.landmark_detector_type
        if landmark_detector_type == 'face_alignment':
            return FaceAlignment(LandmarksType._2D, device=str(device))
        elif landmark_detector_type == 'mobilefacenet':
            model = mobilefacenet.MobileFaceNet([112, 112], 136).eval().to(device)
            sd = torch.load('../landmark_detection/pytorch_face_landmark/weights/mobilefacenet_model_best.pth.tar', map_location=device)['state_dict']
            model.load_state_dict(sd)
            return model

    def get_split_indices(self):
        dataset_size = len(os.listdir(self.config.img_dir))
        indices = list(range(dataset_size))
        val_split = int(np.floor(self.config.val_split * dataset_size))
        test_split = int(np.floor(self.config.test_split * dataset_size))
        if self.config.shuffle:
            np.random.shuffle(indices)

        train_indices = indices[val_split + test_split:]
        val_indices = indices[:val_split]
        test_indices = indices[val_split:val_split + test_split]

        return train_indices, val_indices, test_indices

    def get_loaders(self):
        train_indices, val_indices, test_indices = self.get_split_indices()
        train_dataset = CustomDataset1(img_dir=self.config.img_dir,
                                       img_size=self.config.img_size,
                                       indices=train_indices,
                                       transform=transforms.Compose(
                                           [transforms.Resize(self.config.img_size),
                                            transforms.RandomPerspective(distortion_scale=0.2),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(degrees=(-20, 20)),
                                            transforms.ToTensor()]))
        val_dataset = CustomDataset1(img_dir=self.config.img_dir,
                                     img_size=self.config.img_size,
                                     indices=val_indices,
                                     transform=transforms.Compose(
                                         [transforms.Resize(self.config.img_size), transforms.ToTensor()]))
        test_dataset = CustomDataset1(img_dir=self.config.img_dir,
                                      img_size=self.config.img_size,
                                      indices=test_indices,
                                      transform=transforms.Compose(
                                          [transforms.Resize(self.config.img_size), transforms.ToTensor()]))

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size)
        validation_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        test_loader = DataLoader(test_dataset)

        return train_loader, validation_loader, test_loader
        # self.train_loader, self.val_loader, self.test_loader = SplitDataset(custom_dataset)(
        #     val_split=self.config.val_split,
        #     test_split=self.config.test_split,
        #     shuffle=self.config.shuffle,
        #     batch_size=self.config.batch_size)


def main():
    mode = 'private'
    # mode = 'cluster'
    adv_mask = AdversarialMask(mode)
    adv_mask.train()
    evaluator = Evaluator(adv_mask)
    evaluator.test()


if __name__ == '__main__':
    main()
