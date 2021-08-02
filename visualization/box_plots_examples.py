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


def plot_sim_box(similarities, target_type):
    sim_df = pd.DataFrame()
    for i in range(len(similarities)):
        sim_df[mask_names[i]] = similarities[i]
    sorted_index = sim_df.mean().sort_values(ascending=False).index
    sim_df_sorted = sim_df[sorted_index]
    sns.boxplot(data=sim_df_sorted).set_title('Similarities Difference')
    plt.xlabel('Mask Type')
    plt.ylabel('Similarity')
    plt.savefig('combined_box_plot_' + target_type + '.png')
    plt.close()


def get_final_similarity_from_disk(file_name):
    sims = []
    with open(file_name, 'rb') as f:
        while True:
            try:
                sims.extend(pickle.load(f))
            except EOFError:
                break
    return sims


def gather_sim_and_plot(target_type, job_id):
    output_folders = glob.glob('../patch/experiments/**/*{}'.format(job_id))
    sims = []
    for i, mask_name in enumerate(mask_names):
        file_names = [glob.glob(os.path.join(folder, 'saved_similarities', target_type + '_' + mask_name + '.pickle')) for folder in output_folders]
        sims.append([])
        for file_name in file_names:
            sims[i].extend(get_final_similarity_from_disk(file_name[0]))
    plot_sim_box(sims, target_type)


gather_sim_and_plot(target_type='with_mask', job_id='140570')
gather_sim_and_plot(target_type='without_mask', job_id='140570')
