import json
import os
from collections import defaultdict
import numpy as np
import pandas as pd

def diff_o3_im(data_base_directory):


    data_group_A = defaultdict(list)
    data_group_A_algo_count = defaultdict(int)
    data_group_C = defaultdict(list)
    data_group_C_algo_count = defaultdict(int)

    mean_kld_group_A = defaultdict(float)
    mean_kld_group_C = defaultdict(float)
    std_kld_group_A = defaultdict(float)
    std_kld_group_C = defaultdict(float)
    max_kld_group_A = defaultdict(float)
    max_kld_group_C = defaultdict(float)
    min_kld_group_A = defaultdict(float)
    min_kld_group_C = defaultdict(float)
    spread_kld_group_A = defaultdict(float)
    spread_kld_group_C = defaultdict(float)


    for date_folder in sorted(os.listdir(data_base_directory)):
        if not date_folder.startswith('2024'):
            continue
        date_path = os.path.join(data_base_directory, date_folder)

        input_file_path = os.path.join(date_path, 'analyzed_results_kl.json')
        tvd_data = {}
        with open(input_file_path, 'r') as input_file:
            tvd_data = json.load(input_file)
        
        for group in tvd_data:
            for computer_name in tvd_data[group]:
                for algo_index in tvd_data[group][computer_name]:
                    for run_index in tvd_data[group][computer_name][algo_index]:
                        if group == 'C':
                            data_group_C[algo_index].append(tvd_data[group][computer_name][algo_index][run_index])
                            data_group_C_algo_count[algo_index] += 1


    for date_folder in sorted(os.listdir(data_base_directory)):
        if not date_folder.startswith('2024'):
            continue
        date_path = os.path.join(data_base_directory, date_folder)

        input_file_path = os.path.join(date_path, 'analyzed_results_kl.json')
        tvd_data = {}
        with open(input_file_path, 'r') as input_file:
            tvd_data = json.load(input_file)
        
        for group in tvd_data:
            for computer_name in tvd_data[group]:
                for algo_index in tvd_data[group][computer_name]:
                    for run_index in tvd_data[group][computer_name][algo_index]:
                        if group == 'A':
                            data_group_A[algo_index].append(tvd_data[group][computer_name][algo_index][run_index])
                            data_group_A_algo_count[algo_index] += 1

    for algo_index in data_group_C:
        mean_kld_group_C[algo_index] = sum(data_group_C[algo_index]) / len(data_group_C[algo_index])
    
    for algo_index in data_group_A:
        mean_kld_group_A[algo_index] = sum(data_group_A[algo_index]) / len(data_group_A[algo_index])

    for algo_index in data_group_C:
        std_kld_group_C[algo_index] = np.std(data_group_C[algo_index])
    
    for algo_index in data_group_A:
        std_kld_group_A[algo_index] = np.std(data_group_A[algo_index])
    
    for algo_index in data_group_C:
        max_kld_group_C[algo_index] = np.max(data_group_C[algo_index])
    
    for algo_index in data_group_A:
        max_kld_group_A[algo_index] = np.max(data_group_A[algo_index])
    
    for algo_index in data_group_C:
        min_kld_group_C[algo_index] = np.min(data_group_C[algo_index])
    
    for algo_index in data_group_A:
        min_kld_group_A[algo_index] = np.min(data_group_A[algo_index])
    
    for algo_index in data_group_C:
        spread_kld_group_C[algo_index] = max_kld_group_C[algo_index] - min_kld_group_C[algo_index]
    
    for algo_index in data_group_A:
        spread_kld_group_A[algo_index] = max_kld_group_A[algo_index] - min_kld_group_A[algo_index]
    
    # Define the algorithm indices we want to calculate the difference for
    target_indices = list(range(15)) + [17]

    # Calculate the difference between the average tvd for the specified indices
    kld_mean_differences = {}

    for algo_index in target_indices:
        if algo_index == 6:
            kld_mean_differences[algo_index] = 0
        else:
            kld_mean_differences[algo_index] = - (mean_kld_group_C[str(algo_index)] - mean_kld_group_A[str(algo_index)])

    with open('../data/kld_differences.json', 'w') as json_file:
        json.dump(kld_mean_differences, json_file, indent=4)

data_base_directory = '../data'
diff_o3_im(data_base_directory)