# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:13:29 2024

This script generates preprocessed data from wireless communication scenarios, 
including token generation, patch creation, and data sampling for machine learning models.

@author: salikha4
"""

import numpy as np
import os
from tqdm import tqdm
import time
import pickle
import DeepMIMOv3
import torch
from lwm.utils import plot_coverage, generate_gaussian_noise

def add_noise(
    H_all,
    rx_power_dbm,
    snr_bins=((0, 10), (10, 20), (20, 30), (30, 60)),
    pct_bins=(10, 25, 45, 100),
    seed=42,
):
    
    H_all = np.asarray(H_all)
    rx_power_dbm = np.asarray(rx_power_dbm).reshape(-1)

    rng = np.random.default_rng(seed)
    num_users = H_all.shape[0]

    # Rank users by Rx power (low → high)
    order = np.argsort(rx_power_dbm)
    rank = np.empty_like(order)
    rank[order] = np.arange(num_users)
    percentile = rank / max(num_users - 1, 1)

    # Assign target SNRs per percentile
    snr_target_db = np.zeros(num_users, dtype=float)
    pct_edges = np.asarray(pct_bins, dtype=float) / 100.0

    prev_p = 0.0
    for (snr_lo, snr_hi), p_hi in zip(snr_bins, pct_edges):
        mask = (percentile >= prev_p) & (percentile < p_hi)
        denom = max(p_hi - prev_p, 1e-12)
        t = (percentile[mask] - prev_p) / denom
        snr_target_db[mask] = snr_lo + t * (snr_hi - snr_lo)
        prev_p = p_hi

    # Add per-user complex AWGN
    H_noisy = np.empty_like(H_all, dtype=np.complex128)
    for u in range(num_users):
        sig_pow = np.mean(np.abs(H_all[u]) ** 2)
        snr_lin = 10 ** (snr_target_db[u] / 10)
        noise_pow = sig_pow / max(snr_lin, 1e-12)
        noise_std = np.sqrt(noise_pow / 2)

        noise = noise_std * (
            rng.standard_normal(H_all[u].shape) + 1j * rng.standard_normal(H_all[u].shape)
        )
        H_noisy[u] = H_all[u] + noise

    return H_noisy.astype(H_all.dtype, copy=False), snr_target_db

#%% Scenarios List
def scenarios_list():
    """Returns an array of available scenarios."""
    return np.array([
        'city_18_denver', 'city_15_indianapolis', 'city_19_oklahoma',
        'city_12_fortworth', 'city_11_santaclara', 'city_7_sandiego',
        'O1_3p5_v1', 'O1_3p5_v2', 'Boston5G_3p5', 'asu_campus1',
        'city_0_newyork', 'city_1_losangeles', 'city_2_chicago',
        'city_3_houston', 'city_4_phoenix', 'city_5_philadelphia',
        'city_6_miami', 'city_8_dallas', 'city_9_sanfrancisco',
        'city_10_austin', 'city_13_columbus', 'city_17_seattle'
    ])

#%% Token Generation
def tokenizer(
    selected_scenario_names=None,
    manual_data=None,
    gen_raw=True,
    snr_db=None,
    dataset_folder=None,
):
    """
    Generates tokens by preparing and preprocessing the dataset.

    Args:
        scenario_idxs (list): Indices of the scenarios.
        patch_gen (bool): Whether to generate patches. Defaults to True.
        patch_size (int): Size of each patch. Defaults to 16.
        gen_deepMIMO_data (bool): Whether to generate DeepMIMO data. Defaults to False.
        gen_raw (bool): Whether to generate raw data. Defaults to False.
        save_data (bool): Whether to save the preprocessed data. Defaults to False.
    
    Returns:
        preprocessed_data, sequence_length, element_length: Preprocessed data and related dimensions.
    """

    if manual_data is not None:
        patches = patch_maker(np.expand_dims(np.array(manual_data), axis=1), snr_db=snr_db)
    else:
        # Patch generation or loading
        if isinstance(selected_scenario_names, str):
            selected_scenario_names = [selected_scenario_names]
        deepmimo_data = [
            DeepMIMO_data_gen(scenario_name, dataset_folder=dataset_folder)
            for scenario_name in selected_scenario_names
        ]
        n_scenarios = len(selected_scenario_names)
        
        cleaned_deepmimo_data = [deepmimo_data_cleaning(deepmimo_data[scenario_idx]) for scenario_idx in range(n_scenarios)]
        
        patches = [patch_maker(cleaned_deepmimo_data[scenario_idx], snr_db=snr_db) for scenario_idx in range(n_scenarios)]
        patches = np.vstack(patches)

    # Define dimensions
    patch_size = patches.shape[2]
    n_patches = patches.shape[1]
    n_masks_half = int(0.15 * n_patches / 2)
    
    word2id = {'[CLS]': 0.2 * np.ones((patch_size)), '[MASK]': 0.1 * np.ones((patch_size))}

    # Generate preprocessed channels
    preprocessed_data = []
    for user_idx in tqdm(range(len(patches)), desc="Processing items"):
        sample = make_sample(user_idx, patches, word2id, n_patches, n_masks_half, patch_size, gen_raw=gen_raw)
        preprocessed_data.append(sample)
            
    return preprocessed_data

#%%
def deepmimo_data_cleaning(deepmimo_data):
    idxs = np.where(deepmimo_data['user']['LoS'] != -1)[0]
    cleaned_deepmimo_data = deepmimo_data['user']['channel'][idxs]
    return np.array(cleaned_deepmimo_data) * 1e6

#%% Patch Creation
def patch_maker(original_ch, patch_size=16, norm_factor=1e6, snr_db=None):
    """
    Creates patches from the dataset based on the scenario.

    Args:-
        patch_size (int): Size of each patch.
        scenario (str): Selected scenario for data generation.
        gen_deepMIMO_data (bool): Whether to generate DeepMIMO data.
        norm_factor (int): Normalization factor for channels.

    Returns:
        patch (numpy array): Generated patches.
    """
    flat_channels = original_ch.reshape((original_ch.shape[0], -1)).astype(np.csingle)
    if snr_db is not None:
        flat_channels += generate_gaussian_noise(flat_channels, snr_db)
        
    flat_channels_complex = np.hstack((flat_channels.real, flat_channels.imag))
        
    # Create patches
    n_patches = flat_channels_complex.shape[1] // patch_size
    patch = np.zeros((len(flat_channels_complex), n_patches, patch_size))
    for idx in range(n_patches):
        patch[:, idx, :] = flat_channels_complex[:, idx * patch_size:(idx + 1) * patch_size]
    
    return patch

#%% Data Generation for Scenario Areas
def DeepMIMO_data_gen(scenario, dataset_folder=None, add_awgn=True, seed=42, snr_bins=((0,10),(10,20),(20,30),(30,60)), pct_bins=(10,25,45,100)):
    """
    Generates or loads data for a given scenario.

    Args:
        scenario (str): Scenario name.
        dataset_folder (str, optional): Path to DeepMIMO scenarios root.
        gen_deepMIMO_data (bool): Whether to generate DeepMIMO data.
        save_data (bool): Whether to save generated data.
    
    Returns:
        data (dict): Loaded or generated data.
    """
    import DeepMIMOv3
    
    parameters, row_column_users, n_ant_bs, n_ant_ue, n_subcarriers = get_parameters(
        scenario, dataset_folder=dataset_folder
    )

    deepMIMO_dataset = DeepMIMOv3.generate_data(parameters)
    if scenario not in row_column_users:
        # Fallback for unseen scenarios: use all users as provided by DeepMIMO.
        return deepMIMO_dataset[0]

    users_per_row = row_column_users[scenario]['n_per_row']
    max_cols = row_column_users[scenario].get('n_cols_use', users_per_row)
    uniform_idxs = uniform_sampling(
        deepMIMO_dataset,
        [1, 1],
        len(parameters['user_rows']),
        users_per_row=users_per_row,
        max_cols=max_cols,
    )
    data = select_by_idx(deepMIMO_dataset, uniform_idxs)[0]
    
    [data_noisy, snr_target_db] = add_noise(data,deepMIMO_dataset.power[uniform_idxs,0])

    return data, data_noisy

#%%%
def get_parameters(scenario, dataset_folder=None):
    
    n_ant_bs = 32 
    n_ant_ue = 1
    n_subcarriers = 32 
    scs = 30e3
        
    # O1_3p5 is split into v1/v2 to match grid sizes; v2 limits to first 181 users per row.
    row_column_users = {
        'asu_campus1': {
            'n_rows': 321,
            'n_per_row': 411
        },
        'Boston5G_3p5': {
            'n_rows': [812, 1622],
            'n_per_row': 595
        },
        'Boston5G_3p5_RIS': {
            'n_rows': 1622,
            'n_per_row': 595
        },
        'city_0_newyork': {
            'n_rows': 44,
            'n_per_row': 117
        },
        'city_1_losangeles': {
            'n_rows': 57,
            'n_per_row': 81
        },
        'city_2_chicago': {
            'n_rows': 56,
            'n_per_row': 80
        },
        'city_3_houston': {
            'n_rows': 62,
            'n_per_row': 81
        },
        'city_4_phoenix': {
            'n_rows': 79,
            'n_per_row': 86
        },
        'city_5_philadelphia': {
            'n_rows': 96,
            'n_per_row': 66
        },
        'city_6_miami': {
            'n_rows': 80,
            'n_per_row': 87
        },
        'city_8_dallas': {
            'n_rows': 83,
            'n_per_row': 76
        },
        'city_9_sanfrancisco': {
            'n_rows': 79,
            'n_per_row': 83
        },
        'city_10_austin': {
            'n_rows': 102,
            'n_per_row': 55
        },
        'city_13_columbus': {
            'n_rows': 71,
            'n_per_row': 96
        },
        'city_17_seattle': {
            'n_rows': 74,
            'n_per_row': 82
        },
        'city_18_denver': {
            'n_rows': 85,
            'n_per_row': 82
        },
        'city_15_indianapolis': {
            'n_rows': 80,
            'n_per_row': 79
        },
        'city_19_oklahoma': {
            'n_rows': 82,
            'n_per_row': 75
        },
        'city_12_fortworth': {
            'n_rows': 86,
            'n_per_row': 72
        },
        'city_11_santaclara': {
            'n_rows': 47,
            'n_per_row': 114
        },
        'city_7_sandiego': {
            'n_rows': 71,
            'n_per_row': 83
        },
        'O1_3p5_v1': {
            'n_rows': 3852,
            'n_per_row': 181
        },
        'O1_3p5_v2': {
            'n_rows': [3853, 5203],
            'n_per_row': 361,
            'n_cols_use': 181
        },
        'O1_3p5B': {
            'n_rows': 2751,
            'n_per_row': 181
        }
    }
    
    parameters = DeepMIMOv3.default_params()
    if dataset_folder is None:
        dataset_folder = os.environ.get('LWM_SCENARIOS_DIR', './lwm/LWM/scenarios')
    parameters['dataset_folder'] = os.path.abspath(os.path.expanduser(dataset_folder))
    scenario_base = scenario.split('_v')[0]
    parameters['scenario'] = scenario_base

    if scenario_base == 'O1_3p5':
        parameters['active_BS'] = np.array([4])
    elif scenario in ['city_18_denver', 'city_15_indianapolis']:
        parameters['active_BS'] = np.array([3])
    else:
        parameters['active_BS'] = np.array([1])
        
    if scenario in row_column_users:
        n_rows = row_column_users[scenario]['n_rows']
        if isinstance(n_rows, (list, tuple, np.ndarray)) and len(n_rows) == 2:
            parameters['user_rows'] = np.arange(n_rows[0], n_rows[1])
        else:
            parameters['user_rows'] = np.arange(n_rows)
    parameters['bs_antenna']['shape'] = np.array([n_ant_bs, 1]) # Horizontal, Vertical 
    parameters['bs_antenna']['rotation'] = np.array([0,0,-135]) # (x,y,z)
    parameters['ue_antenna']['shape'] = np.array([n_ant_ue, 1])
    parameters['enable_BS2BS'] = False
    parameters['OFDM']['subcarriers'] = n_subcarriers
    parameters['OFDM']['selected_subcarriers'] = np.arange(n_subcarriers)
    
    parameters['OFDM']['bandwidth'] = scs * n_subcarriers / 1e9
    parameters['num_paths'] = 20
    
    return parameters, row_column_users, n_ant_bs, n_ant_ue, n_subcarriers

#%% Sample Generation
def make_sample(user_idx, patch, word2id, n_patches, n_masks, patch_size, gen_raw=False):
    """
    Generates a sample for each user, including masking and tokenizing.

    Args:
        user_idx (int): Index of the user.
        patch (numpy array): Patches data.
        word2id (dict): Dictionary for special tokens.
        n_patches (int): Number of patches.
        n_masks (int): Number of masks.
        patch_size (int): Size of each patch.
        gen_raw (bool): Whether to generate raw tokens.

    Returns:
        sample (list): Generated sample for the user.
    """
    
    tokens = patch[user_idx]
    input_ids = np.vstack((word2id['[CLS]'], tokens))
    
    real_tokens_size = int(n_patches / 2)
    masks_pos_real = np.random.choice(range(0, real_tokens_size), size=n_masks, replace=False)
    masks_pos_imag = masks_pos_real + real_tokens_size
    masked_pos = np.hstack((masks_pos_real, masks_pos_imag)) + 1
    
    masked_tokens = []
    for pos in masked_pos:
        original_masked_tokens = input_ids[pos].copy()
        masked_tokens.append(original_masked_tokens)
        if not gen_raw:
            rnd_num = np.random.rand()
            if rnd_num < 0.1:
                input_ids[pos] = np.random.rand(patch_size)
            elif rnd_num < 0.9:
                input_ids[pos] = word2id['[MASK]']
                
    return [input_ids, masked_tokens, masked_pos]


#%% Sampling and Data Selection
def uniform_sampling(dataset, sampling_div, n_rows, users_per_row, max_cols=None):
    """
    Performs uniform sampling on the dataset.

    Args:
        dataset (dict): DeepMIMO dataset.
        sampling_div (list): Step sizes along [x, y] dimensions.
        n_rows (int): Number of rows for user selection.
        users_per_row (int): Number of users per row.
        max_cols (int, optional): Limit columns per row (useful for partial grids).

    Returns:
        uniform_idxs (numpy array): Indices of the selected samples.
    """
    if max_cols is None:
        max_cols = users_per_row
    cols = np.arange(max_cols, step=sampling_div[0])
    rows = np.arange(n_rows, step=sampling_div[1])
    uniform_idxs = np.array([j + i * users_per_row for i in rows for j in cols])
    
    return uniform_idxs

def select_by_idx(dataset, idxs):
    """
    Selects a subset of the dataset based on the provided indices.

    Args:
        dataset (dict): Dataset to trim.
        idxs (numpy array): Indices of users to select.

    Returns:
        dataset_t (list): Trimmed dataset based on selected indices.
    """
    dataset_t = []  # Trimmed dataset
    for bs_idx in range(len(dataset)):
        dataset_t.append({})
        for key in dataset[bs_idx].keys():
            dataset_t[bs_idx]['location'] = dataset[bs_idx]['location']
            dataset_t[bs_idx]['user'] = {k: dataset[bs_idx]['user'][k][idxs] for k in dataset[bs_idx]['user']}
    
    return dataset_t

#%% Save and Load Utilities
def save_var(var, path):
    """
    Saves a variable to a pickle file.

    Args:
        var (object): Variable to be saved.
        path (str): Path to save the file.

    Returns:
        None
    """
    path_full = path if path.endswith('.p') else (path + '.pickle')    
    with open(path_full, 'wb') as handle:
        pickle.dump(var, handle)

def load_var(path):
    """
    Loads a variable from a pickle file.

    Args:
        path (str): Path of the file to load.

    Returns:
        var (object): Loaded variable.
    """
    path_full = path if path.endswith('.p') else (path + '.pickle')
    with open(path_full, 'rb') as handle:
        var = pickle.load(handle)
    
    return var

#%% Label Generation
def label_gen(task, data, scenario, n_beams=64, visualize=False):
    
    idxs = np.where(data['user']['LoS'] != -1)[0]
            
    if task == 'LoS/NLoS Classification':
        label = data['user']['LoS'][idxs]
        
        losChs = np.where(data['user']['LoS'] == -1, np.nan, data['user']['LoS'])
        if visualize:
            plot_coverage(data['user']['location'], losChs)
        
    elif task == 'Beam Prediction':
        parameters, row_column_users = get_parameters(scenario)[:2]
        n_users = len(data['user']['channel'])
        n_subbands = 1
        fov = 180

        # Setup Beamformers
        beam_angles = np.around(np.arange(-fov/2, fov/2+.1, fov/(n_beams-1)), 2)

        F1 = np.array([steering_vec(parameters['bs_antenna']['shape'], 
                                    phi=azi*np.pi/180, 
                                    kd=2*np.pi*parameters['bs_antenna']['spacing']).squeeze()
                       for azi in beam_angles])

        full_dbm = np.zeros((n_beams, n_subbands, n_users), dtype=float)
        for ue_idx in tqdm(range(n_users), desc='Computing the channel for each user'):
            if data['user']['LoS'][ue_idx] == -1:
                full_dbm[:,:,ue_idx] = np.nan
            else:
                chs = F1 @ data['user']['channel'][ue_idx]
                full_linear = np.abs(np.mean(chs.squeeze().reshape((n_beams, n_subbands, -1)), axis=-1))
                full_dbm[:,:,ue_idx] = np.around(20*np.log10(full_linear) + 30, 1)

        best_beams = np.argmax(np.mean(full_dbm,axis=1), axis=0)
        best_beams = best_beams.astype(float)
        best_beams[np.isnan(full_dbm[0,0,:])] = np.nan
        
        if visualize:
            plot_coverage(data['user']['location'], best_beams)
        
        label = best_beams[idxs]
        
    return label.astype(int)

def steering_vec(array, phi=0, theta=0, kd=np.pi):
    idxs = DeepMIMOv3.ant_indices(array)
    resp = DeepMIMOv3.array_response(idxs, phi, theta+np.pi/2, kd)
    return resp / np.linalg.norm(resp)

def label_prepend(deepmimo_data, preprocessed_chs, task, scenario_idxs, n_beams=64, visualize=False):
    labels = []
    for scenario_idx in scenario_idxs:
        scenario_name = scenarios_list()[scenario_idx]
        data = deepmimo_data[scenario_idx]
        labels.extend(label_gen(task, data, scenario_name, n_beams=n_beams, visualize=visualize))
    
    preprocessed_chs = [preprocessed_chs[i] + [labels[i]] for i in range(len(preprocessed_chs))]
    
    return preprocessed_chs

def create_labels(task, scenario_names, n_beams=64, dataset_folder=None, visualize=False):
    labels = []
    if isinstance(scenario_names, str):
        scenario_names = [scenario_names]
    for scenario_name in scenario_names:
        data = DeepMIMO_data_gen(scenario_name, dataset_folder=dataset_folder)
        labels.extend(label_gen(task, data, scenario_name, n_beams=n_beams, visualize=visualize))
    return torch.tensor(labels).long()
#%%