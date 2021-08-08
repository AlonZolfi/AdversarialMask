import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

mask_names = ['Clean', 'Adv', 'Random', 'Blue', 'Black', 'White']


def load_all_mask_similarities(raw_data_path, mask_name):
    sims = np.empty(0)
    for file_name in glob.glob(os.path.join(raw_data_path, '**', mask_name + '.npy'), recursive=True):
        with open(file_name, 'rb') as f:
            sim = np.load(f).squeeze()
            sims = np.concatenate([sims, sim])
    return sims


def plot_sim_box(similarities, target_type, emb_name, output_folder):
    sim_df = pd.DataFrame()
    for i in range(len(similarities)):
        sim_df[mask_names[i]] = similarities[i]
    sorted_index = sim_df.mean().sort_values(ascending=False).index
    sim_df_sorted = sim_df[sorted_index]
    sns.boxplot(data=sim_df_sorted).set_title('Similarities Difference')
    plt.xlabel('Mask Type')
    plt.ylabel('Similarity')
    plt.savefig(os.path.join(output_folder, 'combined_sim_boxes', target_type + '_' + emb_name + '.png'))
    plt.close()


def get_final_similarity_from_disk(file_name):
    sims = []
    with open(file_name, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                sims.extend(list(data.values()))
            except EOFError:
                break
    return sims


def gather_sim_and_plot(target_type, embedder_names, job_id):
    for emb_name in embedder_names:
        output_folders = glob.glob('../patch/experiments/**/*{}'.format(job_id), recursive=True)
        sims = []
        for i, mask_name in enumerate(mask_names):
            file_names = [glob.glob(os.path.join(folder, 'saved_similarities', emb_name, target_type + '_' + mask_name + '.pickle')) for folder in output_folders[1:]]
            sims.append([])
            for file_name in file_names:
                sim = get_final_similarity_from_disk(file_name[0])
                sims[i].extend(sim)
        plot_sim_box(sims, target_type, emb_name, output_folders[0])


# gather_sim_and_plot(target_type='with_mask', embedder_names=['resnet100_arcface', 'resnet50_arcface', 'resnet34_arcface', 'resnet18_arcface',
#                                     'resnet100_cosface', 'resnet50_cosface', 'resnet34_cosface', 'resnet18_cosface',
#                                     'resnet100_magface'], job_id='148758')
# gather_sim_and_plot(target_type='without_mask', embedder_names=['resnet100_arcface', 'resnet50_arcface', 'resnet34_arcface', 'resnet18_arcface',
#                                     'resnet100_cosface', 'resnet50_cosface', 'resnet34_cosface', 'resnet18_cosface',
#                                     'resnet100_magface'], job_id='148758')
