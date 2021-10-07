import warnings
import utils
import torch
import os
from torch.nn import CosineSimilarity
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve as pr, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import label_binarize
import matplotlib
import scipy
from pathlib import Path
import pickle
import seaborn as sns
import pandas as pd
matplotlib.use('Agg')

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, adv_mask_class) -> None:
        super().__init__()
        self.config = adv_mask_class.config
        self.adv_mask_class = adv_mask_class
        self.transform = transforms.Compose([transforms.Resize(self.config.patch_size), transforms.ToTensor()])
        self.embedders = utils.load_embedder(self.config.test_embedder_names, device=device)
        self.loaders = utils.get_test_loaders(self.config)
        self.target_embedding_w_mask, self.target_embedding_wo_mask = {}, {}
        for dataset_name, loader in self.loaders.items():
            self.target_embedding_w_mask[dataset_name] = utils.get_person_embedding(self.config, loader, self.config.test_celeb_lab_mapper[dataset_name], self.adv_mask_class.location_extractor,
                                                                                    self.adv_mask_class.fxz_projector, self.embedders, device, include_others=True)
            self.target_embedding_wo_mask[dataset_name] = utils.get_person_embedding(self.config, loader, self.config.test_celeb_lab_mapper[dataset_name], self.adv_mask_class.location_extractor,
                                                                                     self.adv_mask_class.fxz_projector, self.embedders, device, include_others=False)
        self.random_mask_t = utils.load_mask(self.config, self.config.random_mask_path, device)
        self.blue_mask_t = utils.load_mask(self.config, self.config.blue_mask_path, device)
        self.black_mask_t = utils.load_mask(self.config, self.config.black_mask_path, device)
        self.white_mask_t = utils.load_mask(self.config, self.config.white_mask_path, device)
        self.face1_mask_t = utils.load_mask(self.config, self.config.face1_mask_path, device)
        self.face2_mask_t = utils.load_mask(self.config, self.config.face2_mask_path, device)
        self.face3_mask_t = utils.load_mask(self.config, self.config.face3_mask_path, device)
        self.mask_names = ['Clean', 'Adv', 'Random', 'Blue', 'Black', 'White', 'Face1', "Face2", 'Face3']

    def test(self):
        if self.config.is_real_person:
            return
        self.calc_overall_similarity()
        for dataset_name in self.config.test_dataset_names:
            similarities_target_with_mask = self.get_final_similarity_from_disk('with_mask', dataset_name=dataset_name)
            similarities_target_without_mask = self.get_final_similarity_from_disk('without_mask', dataset_name=dataset_name)
            self.calc_similarity_statistics(similarities_target_with_mask, target_type='with', dataset_name=dataset_name)
            self.calc_similarity_statistics(similarities_target_without_mask, target_type='without', dataset_name=dataset_name)
            self.plot_sim_box(similarities_target_with_mask, target_type='with', dataset_name=dataset_name)
            self.plot_sim_box(similarities_target_without_mask, target_type='without', dataset_name=dataset_name)

            similarities_target_with_mask_by_person = self.get_final_similarity_from_disk('with_mask', dataset_name=dataset_name, by_person=True)
            similarities_target_without_mask_by_person = self.get_final_similarity_from_disk('without_mask', dataset_name=dataset_name, by_person=True)
            self.calc_similarity_statistics(similarities_target_with_mask_by_person, target_type='with', dataset_name=dataset_name, by_person=True)
            self.calc_similarity_statistics(similarities_target_without_mask_by_person, target_type='without', dataset_name=dataset_name, by_person=True)
            self.plot_sim_box(similarities_target_with_mask_by_person, target_type='with', dataset_name=dataset_name, by_person=True)
            self.plot_sim_box(similarities_target_without_mask_by_person, target_type='without', dataset_name=dataset_name, by_person=True)

        if len(self.config.celeb_lab) > 1:
            converters = {"y_true": lambda x: list(map(int, x.strip("[]").split(", "))),
                          "y_pred": lambda x: list(map(float, x.strip("[]").split(", ")))}
            for dataset_name in self.config.test_dataset_names:
                preds_with_mask_df = pd.read_csv(os.path.join(self.config.current_dir, 'saved_preds', dataset_name, 'preds_with_mask.csv'), converters=converters)
                precisions_with_mask, recalls_with_mask, aps_with_mask = self.get_pr(preds_with_mask_df, dataset_name=dataset_name)
                self.plot_pr_curve(precisions_with_mask, recalls_with_mask, aps_with_mask, target_type='with_mask', dataset_name=dataset_name)
                self.calc_ap_statistics(aps_with_mask, target_type='with_mask', dataset_name=dataset_name)
                preds_without_mask_df = pd.read_csv(os.path.join(self.config.current_dir, 'saved_preds', dataset_name, 'preds_without_mask.csv'), converters=converters)
                precisions_without_mask, recalls_without_mask, aps_without_mask = self.get_pr(preds_without_mask_df, dataset_name=dataset_name)
                self.plot_pr_curve(precisions_without_mask, recalls_without_mask, aps_without_mask, target_type='without_mask', dataset_name=dataset_name)
                self.calc_ap_statistics(aps_without_mask, target_type='without_mask', dataset_name=dataset_name)

    @torch.no_grad()
    def calc_overall_similarity(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            adv_patch = self.adv_mask_class.best_patch.to(device)
            for dataset_name, loader in self.loaders.items():
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
                # sim_df[self.mask_names[i]] = (similarities[i].squeeze() + 1) / 2
                sim_df[self.mask_names[i]] = similarities[emb_name][i]
            sorted_index = sim_df.mean().sort_values(ascending=False).index
            sim_df_sorted = sim_df[sorted_index]
            sns.boxplot(data=sim_df_sorted).set_title('Similarities for Different Masks')
            # plt.axhline(y=self.config.same_person_threshold, color='red', linestyle='--', label='System Threshold (FAR=.05)')
            plt.xlabel('Mask Type')
            plt.ylabel('Similarity')
            # plt.gca().set_ylim([, 1.05])
            # plt.legend()
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
        img_batch_applied_adv = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                 self.adv_mask_class.fxz_projector, img_batch, adv_patch)
        img_batch_applied_random = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                    self.adv_mask_class.fxz_projector, img_batch,
                                                    self.random_mask_t)
        img_batch_applied_blue = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                  self.adv_mask_class.fxz_projector, img_batch,
                                                  self.blue_mask_t[:, :3],
                                                  self.blue_mask_t[:, 3], is_3d=True)
        img_batch_applied_black = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                   self.adv_mask_class.fxz_projector, img_batch,
                                                   self.black_mask_t[:, :3],
                                                   self.black_mask_t[:, 3], is_3d=True)
        img_batch_applied_white = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                   self.adv_mask_class.fxz_projector, img_batch,
                                                   self.white_mask_t[:, :3],
                                                   self.white_mask_t[:, 3], is_3d=True)
        img_batch_applied_face1 = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                   self.adv_mask_class.fxz_projector, img_batch,
                                                   self.face1_mask_t[:, :3],
                                                   self.face1_mask_t[:, 3], is_3d=True)
        img_batch_applied_face2 = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                   self.adv_mask_class.fxz_projector, img_batch,
                                                   self.face2_mask_t[:, :3],
                                                   self.face2_mask_t[:, 3], is_3d=True)
        img_batch_applied_face3 = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                   self.adv_mask_class.fxz_projector, img_batch,
                                                   self.face3_mask_t[:, :3],
                                                   self.face3_mask_t[:, 3], is_3d=True)

        return img_batch_applied_adv, img_batch_applied_random, img_batch_applied_blue, img_batch_applied_black, \
               img_batch_applied_white, img_batch_applied_face1, img_batch_applied_face2, img_batch_applied_face3

    def get_all_embeddings(self, img_batch, img_batch_applied_masks):
        batch_embs = {}
        for emb_name, emb_model in self.embedders.items():
            batch_embs[emb_name] = [emb_model(img_batch.to(device)).cpu().numpy()]
            for img_batch_applied_mask in img_batch_applied_masks:
                batch_embs[emb_name].append(emb_model(img_batch_applied_mask.to(device)).cpu().numpy())
        return batch_embs

    def save_all_embeddings(self, all_test_image_clean, all_test_image_adv, all_test_image_random, all_test_image_blue,
                            all_test_image_black, all_test_image_white,
                            batch_emb_clean, batch_emb_adv, batch_emb_random, batch_emb_blue, batch_emb_black,
                            batch_emb_white):
        all_test_image_clean = np.concatenate([all_test_image_clean, batch_emb_clean])
        all_test_image_adv = np.concatenate([all_test_image_adv, batch_emb_adv])
        all_test_image_random = np.concatenate([all_test_image_random, batch_emb_random])
        all_test_image_blue = np.concatenate([all_test_image_blue, batch_emb_blue])
        all_test_image_black = np.concatenate([all_test_image_black, batch_emb_black])
        all_test_image_white = np.concatenate([all_test_image_white, batch_emb_white])
        return all_test_image_clean, all_test_image_adv, all_test_image_random, all_test_image_blue, all_test_image_black, all_test_image_white

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

    def get_pr(self, df, dataset_name):
        precisions, recalls, aps = {}, {}, {}
        for emb_name in self.config.test_embedder_names:
            precisions[emb_name], recalls[emb_name], aps[emb_name] = [], [], []
            for mask_name in self.mask_names:
                precision, recall, ap = dict(), dict(), dict()
                tmp_df = df[(df.emb_name == emb_name) & (df.mask_name == mask_name)]
                y_true = np.array([np.array(lst) for lst in tmp_df.y_true.values])
                y_pred = np.array([np.array(lst) for lst in tmp_df.y_pred.values])
                for i in range(len(self.config.test_celeb_lab[dataset_name])):
                    precision[i], recall[i], _ = pr(y_true[:, i], y_pred[:, i])
                    ap[i] = average_precision_score(y_true[:, i], y_pred[:, i])

                precision["micro"], recall["micro"], _ = pr(y_true.ravel(), y_pred.ravel())
                ap["micro"] = average_precision_score(y_true, y_pred, average="micro")

                precisions[emb_name].append(precision)
                recalls[emb_name].append(recall)
                aps[emb_name].append(ap)

        return precisions, recalls, aps

    def plot_pr_curve(self, precisions, recalls, aps, target_type, dataset_name):
        Path(os.path.join(self.config.current_dir, 'final_results', 'pr-curves', dataset_name, target_type)).mkdir(parents=True, exist_ok=True)
        for emb_name in self.config.test_embedder_names:
            plt.plot([0, 1.05], [0, 1.05], '--', color='gray')
            title = 'Precision-Recall Curve'
            plt.title(title)
            for i in range(len(self.mask_names)):
                plt.plot(recalls[emb_name][i]['micro'],
                         precisions[emb_name][i]['micro'],
                         label='{}: AP: {}%'.format(self.mask_names[i], round(aps[emb_name][i]['micro'] * 100, 2)))

            plt.gca().set_ylabel('Precision')
            plt.gca().set_xlabel('Recall')
            plt.gca().set_xlim([0, 1.05])
            plt.gca().set_ylim([0, 1.05])

            handles, labels = plt.gca().get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles),
                                          key=lambda t: float(t[0].split('AP: ')[1].replace('%', '')),
                                          reverse=True))
            plt.gca().legend(handles, labels, loc=4)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(os.path.join(self.config.current_dir, 'final_results', 'pr-curves', dataset_name, target_type, emb_name + '.png'))
            plt.close()

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
                # cos_sim = (cosine_similarity(emb, target_embedding) + 1) / 2
                # max_idx = np.argmax(cos_sim, axis=1)
                # max_mask = label_binarize(max_idx, classes=class_labels)
                # y_pred = np.round(cos_sim * max_mask, decimals=3)
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

    def calc_ap_statistics(self, ap_dict, target_type, dataset_name):
        df_mean = pd.DataFrame(columns=['emb_name'] + self.mask_names)
        df_std = pd.DataFrame(columns=['emb_name'] + self.mask_names)
        for emb_name, sim_values in ap_dict.items():
            micro_values = []
            for value in sim_values:
                micro_values.append(round(value['micro'], 3))
            df_mean = df_mean.append(pd.Series([emb_name] + micro_values, index=df_mean.columns), ignore_index=True)
            df_std = df_std.append(pd.Series([emb_name] + micro_values, index=df_std.columns), ignore_index=True)
        Path(os.path.join(self.config.current_dir, 'final_results', 'stats', 'average_precision', dataset_name)).mkdir(parents=True, exist_ok=True)
        df_mean.to_csv(os.path.join(self.config.current_dir, 'final_results', 'stats', 'average_precision', dataset_name, 'mean_df' + '_' + target_type + '.csv'), index=False)
        df_std.to_csv(os.path.join(self.config.current_dir, 'final_results', 'stats', 'average_precision', dataset_name, 'std_df' + '_' + target_type + '.csv'), index=False)

