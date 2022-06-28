import sys
import os
if sys.base_prefix.__contains__('home/zolfi'):
    sys.path.append('/home/zolfi/AdversarialMask')
    sys.path.append('/home/zolfi/AdversarialMask/patch')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import warnings
import utils
import torch
from nn_modules import LandmarkExtractor, FaceXZooProjector

from config import patch_config_types
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import label_binarize
import matplotlib
from pathlib import Path
import pickle
import seaborn as sns
import pandas as pd
matplotlib.use('Agg')

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, config, best_patch) -> None:
        super().__init__()
        self.config = config
        self.best_patch = best_patch
        face_landmark_detector = utils.get_landmark_detector(self.config, device)
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, self.config.img_size).to(device)
        self.fxz_projector = FaceXZooProjector(device, self.config.img_size, self.config.patch_size).to(device)
        self.transform = transforms.Compose([transforms.Resize(self.config.patch_size), transforms.ToTensor()])
        self.embedders = utils.load_embedder(self.config.test_embedder_names, device=device)
        emb_loaders, self.test_loaders = utils.get_test_loaders(self.config, self.config.test_celeb_lab.keys())

        self.target_embedding_w_mask, self.target_embedding_wo_mask = {}, {}
        for dataset_name, loader in emb_loaders.items():
            self.target_embedding_w_mask[dataset_name] = utils.get_person_embedding(self.config, loader, self.config.test_celeb_lab_mapper[dataset_name], self.location_extractor,
                                                                                    self.fxz_projector, self.embedders, device, include_others=True)
            self.target_embedding_wo_mask[dataset_name] = utils.get_person_embedding(self.config, loader, self.config.test_celeb_lab_mapper[dataset_name], self.location_extractor,
                                                                                     self.fxz_projector, self.embedders, device, include_others=False)

        self.random_mask_t = utils.load_mask(self.config, self.config.random_mask_path, device)
        self.blue_mask_t = utils.load_mask(self.config, self.config.blue_mask_path, device)
        self.face1_mask_t = utils.load_mask(self.config, self.config.face1_mask_path, device)
        self.face3_mask_t = utils.load_mask(self.config, self.config.face3_mask_path, device)
        self.mask_names = ['Clean', 'Adv', 'Random', 'Blue', 'Face1', 'Face3']

        Path(self.config.current_dir).mkdir(parents=True, exist_ok=True)
        utils.save_class_to_file(self.config, self.config.current_dir)

    def test(self):
        self.calc_overall_similarity()
        for dataset_name in self.test_loaders.keys():
            similarities_target_with_mask_by_person = self.get_final_similarity_from_disk('with_mask', dataset_name=dataset_name, by_person=True)
            similarities_target_without_mask_by_person = self.get_final_similarity_from_disk('without_mask', dataset_name=dataset_name, by_person=True)
            self.calc_similarity_statistics(similarities_target_with_mask_by_person, target_type='with', dataset_name=dataset_name, by_person=True)
            self.calc_similarity_statistics(similarities_target_without_mask_by_person, target_type='without', dataset_name=dataset_name, by_person=True)
            self.plot_sim_box(similarities_target_with_mask_by_person, target_type='with', dataset_name=dataset_name, by_person=True)
            self.plot_sim_box(similarities_target_without_mask_by_person, target_type='without', dataset_name=dataset_name, by_person=True)

    @torch.no_grad()
    def calc_overall_similarity(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            adv_patch = self.best_patch.to(device)
            for dataset_name, loader in self.test_loaders.items():
                df_with_mask = pd.DataFrame(columns=['y_true', 'y_pred'])
                df_without_mask = pd.DataFrame(columns=['y_true', 'y_pred'])
                for img_batch, img_names, cls_id in tqdm(loader):
                    img_batch = img_batch.to(device)
                    cls_id = cls_id.to(device).type(torch.int32)

                    # Apply different types of masks
                    img_batch_applied = self.apply_all_masks(img_batch, adv_patch)

                    # Get embedding
                    all_embeddings = self.get_all_embeddings(img_batch, img_batch_applied)

                    self.calc_all_similarity(all_embeddings, img_names, cls_id, 'with_mask', dataset_name)
                    self.calc_all_similarity(all_embeddings, img_names, cls_id, 'without_mask', dataset_name)

                    df_with_mask = df_with_mask.append(self.calc_preds(cls_id, all_embeddings, target_type='with_mask', dataset_name=dataset_name))
                    df_without_mask = df_without_mask.append(self.calc_preds(cls_id, all_embeddings, target_type='without_mask', dataset_name=dataset_name))

                Path(os.path.join(self.config.current_dir, 'saved_preds', dataset_name)).mkdir(parents=True, exist_ok=True)
                df_with_mask.to_csv(os.path.join(self.config.current_dir, 'saved_preds', dataset_name, 'preds_with_mask.csv'), index=False)
                df_without_mask.to_csv(os.path.join(self.config.current_dir, 'saved_preds', dataset_name, 'preds_without_mask.csv'), index=False)

    def plot_sim_box(self, similarities, target_type, dataset_name, by_person=False):
        Path(os.path.join(self.config.current_dir, 'final_results', 'sim-boxes', dataset_name, target_type)).mkdir(parents=True, exist_ok=True)
        for emb_name in self.config.test_embedder_names:
            sim_df = pd.DataFrame()
            for i in range(len(similarities[emb_name])):
                sim_df[self.mask_names[i]] = similarities[emb_name][i]
            sorted_index = sim_df.mean().sort_values(ascending=False).index
            sim_df_sorted = sim_df[sorted_index]
            sns.boxplot(data=sim_df_sorted).set_title('Similarities for Different Masks')
            plt.xlabel('Mask Type')
            plt.ylabel('Similarity')
            avg_type = 'person' if by_person else 'image'
            plt.savefig(os.path.join(self.config.current_dir, 'final_results', 'sim-boxes', dataset_name, target_type, avg_type + '_' + emb_name + '.png'))
            plt.close()

    def write_similarities_to_disk(self, sims, img_names, cls_ids, sim_type, emb_name, dataset_name):
        Path(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name)).mkdir(parents=True, exist_ok=True)
        for i, lab in self.config.test_celeb_lab_mapper[dataset_name].items():
            Path(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, lab)).mkdir(parents=True, exist_ok=True)
            for similarity, mask_name in zip(sims, self.mask_names):
                sim = similarity[cls_ids.cpu().numpy() == i].tolist()
                sim = {img_name: s for img_name, s in zip(img_names, sim)}
                with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, lab, sim_type + '_' + mask_name + '.pickle'), 'ab') as f:
                    pickle.dump(sim, f)
        for similarity, mask_name in zip(sims, self.mask_names):
            sim = {img_name: s for img_name, s in zip(img_names, similarity.tolist())}
            with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, sim_type + '_' + mask_name + '.pickle'), 'ab') as f:
                pickle.dump(sim, f)

    def apply_all_masks(self, img_batch, adv_patch):
        img_batch_applied_adv = utils.apply_mask(self.location_extractor,
                                                 self.fxz_projector, img_batch, adv_patch)
        img_batch_applied_random = utils.apply_mask(self.location_extractor,
                                                    self.fxz_projector, img_batch,
                                                    self.random_mask_t)
        img_batch_applied_blue = utils.apply_mask(self.location_extractor,
                                                  self.fxz_projector, img_batch,
                                                  self.blue_mask_t[:, :3],
                                                  self.blue_mask_t[:, 3], is_3d=True)
        img_batch_applied_face1 = utils.apply_mask(self.location_extractor,
                                                   self.fxz_projector, img_batch,
                                                   self.face1_mask_t[:, :3],
                                                   self.face1_mask_t[:, 3], is_3d=True)
        img_batch_applied_face3 = utils.apply_mask(self.location_extractor,
                                                   self.fxz_projector, img_batch,
                                                   self.face3_mask_t[:, :3],
                                                   self.face3_mask_t[:, 3], is_3d=True)

        return img_batch_applied_adv, img_batch_applied_random, img_batch_applied_blue, img_batch_applied_face1, img_batch_applied_face3

    def get_all_embeddings(self, img_batch, img_batch_applied_masks):
        batch_embs = {}
        for emb_name, emb_model in self.embedders.items():
            batch_embs[emb_name] = [emb_model(img_batch.to(device)).cpu().numpy()]
            for img_batch_applied_mask in img_batch_applied_masks:
                batch_embs[emb_name].append(emb_model(img_batch_applied_mask.to(device)).cpu().numpy())
        return batch_embs

    def calc_all_similarity(self, all_embeddings, img_names, cls_id, target_type, dataset_name):
        for emb_name in self.config.test_embedder_names:
            target = self.target_embedding_w_mask[dataset_name][emb_name] if target_type == 'with_mask' else self.target_embedding_wo_mask[dataset_name][emb_name]
            target_embedding = torch.index_select(target, index=cls_id, dim=0).cpu().numpy().squeeze(-2)
            sims = []
            for emb in all_embeddings[emb_name]:
                sims.append(np.diag(cosine_similarity(emb, target_embedding)))
            self.write_similarities_to_disk(sims, img_names, cls_id, sim_type=target_type, emb_name=emb_name, dataset_name=dataset_name)

    def get_final_similarity_from_disk(self, sim_type, dataset_name, by_person=False):
        sims = {}
        for emb_name in self.config.test_embedder_names:
            sims[emb_name] = []
            for i, mask_name in enumerate(self.mask_names):
                if not by_person:
                    with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, sim_type + '_' + mask_name + '.pickle'), 'rb') as f:
                        sims[emb_name].append([])
                        while True:
                            try:
                                data = pickle.load(f).values()
                                sims[emb_name][i].extend(list(data))
                            except EOFError:
                                break
                else:
                    sims[emb_name].append([])
                    for lab in self.config.test_celeb_lab[dataset_name]:
                        with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, lab, sim_type + '_' + mask_name + '.pickle'), 'rb') as f:
                            person_sims = []
                            while True:
                                try:
                                    data = pickle.load(f).values()
                                    person_sims.extend(list(data))
                                except EOFError:
                                    break
                            person_avg_sim = sum(person_sims) / len(person_sims)
                            sims[emb_name][i].append(person_avg_sim)
        return sims


    def calc_preds(self, cls_id, all_embeddings, target_type, dataset_name):
        df = pd.DataFrame(columns=['emb_name', 'mask_name', 'y_true', 'y_pred'])
        class_labels = list(range(0, len(self.config.test_celeb_lab_mapper[dataset_name])))
        y_true = label_binarize(cls_id.cpu().numpy(), classes=class_labels)
        y_true = [lab.tolist() for lab in y_true]
        for emb_name in self.config.test_embedder_names:
            target_embedding = self.target_embedding_w_mask[dataset_name][emb_name] \
                if target_type == 'with_mask' else self.target_embedding_wo_mask[dataset_name][emb_name]
            target_embedding = target_embedding.cpu().numpy().squeeze(-2)
            for i, mask_name in enumerate(self.mask_names):
                emb = all_embeddings[emb_name][i]
                cos_sim = cosine_similarity(emb, target_embedding)
                y_pred = [lab.tolist() for lab in cos_sim]
                new_rows = pd.DataFrame({
                    'emb_name': [emb_name] * len(y_true),
                    'mask_name': [mask_name] * len(y_true),
                    'y_true': y_true,
                    'y_pred': y_pred
                })
                df = df.append(new_rows, ignore_index=True)
        return df

    def calc_similarity_statistics(self, sim_dict, target_type, dataset_name, by_person=False):
        df_mean = pd.DataFrame(columns=['emb_name'] + self.mask_names)
        df_std = pd.DataFrame(columns=['emb_name'] + self.mask_names)
        for emb_name, sim_values in sim_dict.items():
            sim_values = np.array([np.array(lst) for lst in sim_values])
            sim_mean = np.round(sim_values.mean(axis=1), decimals=3)
            sim_std = np.round(sim_values.std(axis=1), decimals=3)
            df_mean = df_mean.append(pd.Series([emb_name] + sim_mean.tolist(), index=df_mean.columns), ignore_index=True)
            df_std = df_std.append(pd.Series([emb_name] + sim_std.tolist(), index=df_std.columns), ignore_index=True)

        avg_type = 'person' if by_person else 'image'
        Path(os.path.join(self.config.current_dir, 'final_results', 'stats', 'similarity', dataset_name, target_type)).mkdir(parents=True, exist_ok=True)
        df_mean.to_csv(os.path.join(self.config.current_dir, 'final_results', 'stats', 'similarity', dataset_name, target_type, 'mean_df' + '_' + avg_type + '.csv'), index=False)
        df_std.to_csv(os.path.join(self.config.current_dir, 'final_results', 'stats', 'similarity', dataset_name, target_type, 'std_df' + '_' + avg_type + '.csv'), index=False)


def main():
    mode = 'universal'
    config = patch_config_types[mode]()
    adv_mask = Image.open('../data/masks/final_patch-arccosmag.png').convert('RGB')
    adv_mask_t = transforms.ToTensor()(adv_mask).unsqueeze(0)
    print('Starting test...', flush=True)
    evaluator = Evaluator(config, adv_mask_t)
    evaluator.test()
    print('Finished test...', flush=True)


if __name__ == '__main__':
    main()
