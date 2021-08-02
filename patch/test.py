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
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
from pathlib import Path
import seaborn as sns
import pandas as pd
matplotlib.use('Agg')
import pickle

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, adv_mask_class) -> None:
        super().__init__()
        self.config = adv_mask_class.config
        self.adv_mask_class = adv_mask_class
        self.transform = transforms.Compose([transforms.Resize(self.config.patch_size), transforms.ToTensor()])
        self.random_mask_t = utils.load_mask(self.config, self.config.random_mask_path, device)
        self.blue_mask_t = utils.load_mask(self.config, self.config.blue_mask_path, device)
        self.black_mask_t = utils.load_mask(self.config, self.config.black_mask_path, device)
        self.white_mask_t = utils.load_mask(self.config, self.config.white_mask_path, device)
        self.mask_names = ['Clean', 'Adv', 'Random', 'Blue', 'Black', 'White']

    @torch.no_grad()
    def calc_overall_similarity(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            adv_patch = self.adv_mask_class.best_patch.to(device)
            for img_batch, img_names, cls_id in tqdm(self.adv_mask_class.test_loader):
                img_batch = img_batch.to(device)
                cls_id = cls_id.to(device).type(torch.int32)

                # Apply different types of masks
                img_batch_applied = self.apply_all_masks(img_batch, adv_patch)

                # Get embedding
                all_embeddings = self.get_all_embeddings(img_batch, img_batch_applied)

                self.calc_all_similarity(all_embeddings, cls_id, 'with_mask')
                self.calc_all_similarity(all_embeddings, cls_id, 'without_mask')

    def test(self):
        if self.config.is_real_person:
            return
        self.calc_overall_similarity()
        similarities_target_with_mask = self.get_final_similarity_from_disk('with_mask')
        similarities_target_without_mask = self.get_final_similarity_from_disk('without_mask')
        self.plot_sim_box(similarities_target_with_mask, target_type='with')
        self.plot_sim_box(similarities_target_without_mask, target_type='without')
        # precisions, recalls, thresholds, aps = self.get_pr(similarities)
        # self.plot_pr_curve(precisions, recalls, thresholds, aps)

    def plot_sim_box(self, similarities, target_type):
        for emb_name in self.config.embedder_name:
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
            plt.savefig(self.config.current_dir + '/final_results/sim-boxes_' + target_type + '_' + emb_name + '.png')
            plt.close()

    def write_similarities_to_disk(self, sims, cls_ids, sim_type, emb_name):
        Path(os.path.join(self.config.current_dir, 'saved_similarities', emb_name)).mkdir(parents=True, exist_ok=True)
        for i, lab in self.config.celeb_lab_mapper.items():
            Path(os.path.join(self.config.current_dir, 'saved_similarities', emb_name, lab)).mkdir(parents=True, exist_ok=True)
            for similarity, mask_name in zip(sims, self.mask_names):
                sim = similarity[cls_ids.cpu().numpy() == i].tolist()
                with open(os.path.join(self.config.current_dir, 'saved_similarities', emb_name, lab, mask_name + '.pickle'), 'ab') as f:
                    pickle.dump(sim, f)
        for similarity, mask_name in zip(sims, self.mask_names):
            with open(os.path.join(self.config.current_dir, 'saved_similarities', emb_name, sim_type + '_' + mask_name + '.pickle'), 'ab') as f:
                pickle.dump(similarity.tolist(), f)

    def apply_all_masks(self, img_batch, adv_patch):
        img_batch_applied_adv = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                 self.adv_mask_class.fxz_projector, img_batch, adv_patch)
        img_batch_applied_random = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                    self.adv_mask_class.fxz_projector, img_batch,
                                                    self.random_mask_t)
        img_batch_applied_blue = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                  self.adv_mask_class.fxz_projector, img_batch,
                                                  self.blue_mask_t[:, :3],
                                                  self.blue_mask_t[:, 3])
        img_batch_applied_black = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                   self.adv_mask_class.fxz_projector, img_batch,
                                                   self.black_mask_t[:, :3],
                                                   self.black_mask_t[:, 3])
        img_batch_applied_white = utils.apply_mask(self.adv_mask_class.location_extractor,
                                                   self.adv_mask_class.fxz_projector, img_batch,
                                                   self.white_mask_t[:, :3],
                                                   self.white_mask_t[:, 3])

        return img_batch_applied_adv, img_batch_applied_random, img_batch_applied_blue, img_batch_applied_black, img_batch_applied_white

    def get_all_embeddings(self, img_batch, img_batch_applied_masks):
        batch_embs = {}
        for emb_name, emb_model in self.adv_mask_class.embedders.items():
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

    def calc_all_similarity(self, all_embeddings, cls_id, target_type):
        for emb_name in self.config.embedder_name:
            target = self.adv_mask_class.test_target_embedding[emb_name] if target_type == 'with_mask' else self.adv_mask_class.target_embedding[emb_name]
            target_embedding = torch.index_select(target, index=cls_id, dim=0).cpu().numpy().squeeze(-2)
            sims = []
            for emb in all_embeddings[emb_name]:
                sims.append(np.diag(cosine_similarity(emb, target_embedding)))
            self.write_similarities_to_disk(sims, cls_id, sim_type=target_type, emb_name=emb_name)

    def get_final_similarity_from_disk(self, sim_type):
        sims = {}
        for emb_name in self.config.embedder_name:
            sims[emb_name] = []
            for i, mask_name in enumerate(self.mask_names):
                with open(os.path.join(self.config.current_dir, 'saved_similarities', sim_type + '_' + mask_name + '.pickle'), 'rb') as f:
                    sims[emb_name].append([])
                    while True:
                        try:
                            sims[emb_name][i].extend(pickle.load(f))
                        except EOFError:
                            break
        return sims



'''def get_pr(self, similarities):
        y_true = np.ones(similarities[0].shape[0])

        precisions, recalls, thresholds, aps = [], [], [], []
        for similarity in similarities:
            precision, recall, _ = pr(pd.DataFrame(y_true), pd.DataFrame(similarity.cpu().numpy()))
            ap = average_precision_score(y_true, similarity.cpu().numpy())
            precisions.append(precision)
            recalls.append(recall)
            aps.append(ap)

        return precisions, recalls, thresholds, aps'''

'''def plot_pr_curve(self, precisions, recalls, thresholds, aps):
    plt.plot([0, 1.05], [0, 1.05], '--', color='gray')
    title = 'Precision-Recall Curve'
    plt.title(title)
    for i in range(len(precisions)):
        plt.plot(recalls[i], precisions[i], label='{}: AP: {}%'.format(self.mask_names[i], round(aps[i] * 100, 2)))

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
    plt.imshow()
    plt.savefig(self.adv_mask_class.current_dir + '/final_results/pr-curve.png')'''