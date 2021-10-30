import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

mask_names = ['Clean', 'Adv', 'Random', 'Blue', 'Black', 'White', 'Face1', "Face2", 'Face3']


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


def gather_sim_and_plot(dataset_name, target_type, embedder_names, job_id):
    df_mean = pd.DataFrame(columns=['emb_name'] + mask_names)
    df_std = pd.DataFrame(columns=['emb_name'] + mask_names)
    for emb_name in embedder_names:
        output_folders = glob.glob('../patch/experiments/**/*{}'.format(job_id), recursive=True)
        sims = []
        for i, mask_name in enumerate(mask_names):
            file_names = [glob.glob(os.path.join(folder, 'saved_similarities', dataset_name, emb_name, target_type + '_' + mask_name + '.pickle')) for folder in output_folders[1:]]
            sims.append([])
            for file_name in file_names:
                sim = get_final_similarity_from_disk(file_name[0])
                avg_sim = sum(sim) / len(sim)
                sims[i].append(avg_sim)
        sim_values = np.array([np.array(lst) for lst in sims])
        sim_mean = np.round(sim_values.mean(axis=1), decimals=3)
        sim_std = np.round(sim_values.std(axis=1), decimals=3)
        df_mean = df_mean.append(pd.Series([emb_name] + sim_mean.tolist(), index=df_mean.columns), ignore_index=True)
        df_std = df_std.append(pd.Series([emb_name] + sim_std.tolist(), index=df_std.columns), ignore_index=True)
        plot_sim_box(sims, target_type, emb_name, output_folders[0])

    Path(os.path.join(output_folders[0], 'stats')).mkdir(parents=True, exist_ok=True)
    df_mean.to_csv(os.path.join(output_folders[0], 'stats', target_type + '_' + 'mean_df.csv'), index=False)
    df_std.to_csv(os.path.join(output_folders[0], 'stats', target_type + '_' + 'std_df.csv'), index=False)


# gather_sim_and_plot(target_type='with_mask', embedder_names=['resnet100_arcface', 'resnet50_arcface', 'resnet34_arcface', 'resnet18_arcface',
#                                     'resnet100_cosface', 'resnet50_cosface', 'resnet34_cosface', 'resnet18_cosface',
#                                     'resnet100_magface'], job_id='148758')
# gather_sim_and_plot(target_type='without_mask', embedder_names=['resnet100_arcface', 'resnet50_arcface', 'resnet34_arcface', 'resnet18_arcface',
#                                     'resnet100_cosface', 'resnet50_cosface', 'resnet34_cosface', 'resnet18_cosface',
#                                     'resnet100_magface'], job_id='148758')
