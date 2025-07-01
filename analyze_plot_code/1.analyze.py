import json
import os
from collections import defaultdict
import numpy as np
import re
from scipy.spatial.distance import jensenshannon
import math
from scipy.stats import entropy

NUM_ALGOS = 20

def calculate_tvd(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))

def calculate_jsd(p, q):
    # Ensure that both probability distributions have the same keys
    all_keys = set(p.keys()).union(set(q.keys()))
    
    # Fill missing keys in both distributions with 0 probability
    p_aligned = np.array([p.get(k, 0) for k in all_keys])
    q_aligned = np.array([q.get(k, 0) for k in all_keys])
    
    # Calculate the Jensen-Shannon Divergence
    return jensenshannon(p_aligned, q_aligned, base=2)


import math

# Function to calculate Hellinger Distance
def calculate_hellinger_distance(p, q):
    # Ensure that both probability distributions have the same keys
    all_keys = set(p.keys()).union(set(q.keys()))
    
    # Fill missing keys in both distributions with 0 probability
    p_aligned = [p.get(k, 0) for k in all_keys]
    q_aligned = [q.get(k, 0) for k in all_keys]
    
    # Calculate the Hellinger distance, ensuring the value inside sqrt is clamped between 0 and 1
    sum_value = sum(math.sqrt(p_val) * math.sqrt(q_val) for p_val, q_val in zip(p_aligned, q_aligned))
    
    # Clamp the sum_value between 0 and 1 to avoid numerical issues
    sum_value = min(max(sum_value, 0), 1)
    
    return math.sqrt(1 - sum_value)



# Function to calculate Kullback-Leibler Divergence
def calculate_kl_divergence(P, Q):
    """Calculate Kullback-Leibler Divergence with smoothing."""
    epsilon = 1e-10
    p_sum, q_sum = sum(P.values()), sum(Q.values())
    kld = 0.0
    
    for key in set(P.keys()) | set(Q.keys()):
        p = P.get(key, 0) / p_sum
        q = Q.get(key, 0) / q_sum
        
        # Add smoothing
        p = p + epsilon
        q = q + epsilon
        
        if p > 0:
            kld += p * np.log(p / q)
            
    return kld


# Load JSON data from files
def load_results(data_directory):
    results = defaultdict(dict)
    for file in os.listdir(data_directory):
        if (file.startswith('group') and file.endswith('.json')):
            computer_name_match = re.search(r'_([a-zA-Z]+)_\d{8}_', file)
            computer_name = computer_name_match.group(1) if computer_name_match else 'unknown'
            
            with open(os.path.join(data_directory, file), 'r') as f:
                data = json.load(f)
                for item in data:
                    group = item.get('group')
                    algo_index = item.get('algorithm_index')
                    run_index = item.get('run_index')
                    if group is not None and algo_index is not None and run_index is not None:
                        if group not in results:
                            results[group] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
                        results[group][computer_name][algo_index][run_index].append(item)
    return results

# Load ideal distributions
def load_ideal_distributions(directory):
    ideal_distributions = {}
    for filename in os.listdir(directory):
        if filename.startswith('ideal_distribution_') and filename.endswith('.json'):
            index = int(filename.split('_')[2].split('.')[0])  # assuming 'ideal_distribution_INDEX.json'
            with open(os.path.join(directory, filename), 'r') as f:
                ideal_distributions[index] = json.load(f)
    return ideal_distributions

# Process and analyze results
def analyze_results(results, ideal_distributions):
    analysis = {}
    for group, computers in results.items():
        analysis[group] = {}
        for computer_name, algos in computers.items():
            analysis[group][computer_name] = {}
            for algo_index, runs in algos.items():
                analysis[group][computer_name][algo_index] = {}
                for run_index, data in runs.items():
                    run_tvds = []
                    for run_data in data:
                        counts = run_data.get('counts', {})
                        total_shots = sum(counts.values())
                        prob_dist = {k: v / total_shots for k, v in counts.items()}
                        ideal_dist = ideal_distributions.get(algo_index, {})
                        # tvd = calculate_jsd(prob_dist, ideal_dist)
                        tvd = calculate_kl_divergence(prob_dist, ideal_dist)
                        run_tvds.append(tvd)
                    average_tvd = np.mean(run_tvds) if run_tvds else None
                    analysis[group][computer_name][algo_index][run_index] = average_tvd
    return analysis

# Save analysis to JSON file
def save_results_to_json(analysis, output_filename):
    with open(output_filename, 'w') as f:
        json.dump(analysis, f, indent=4)

########## Main ##########
ideal_distributions_directory = '../input_algorithms/ideal_distributions'
ideal_distributions = load_ideal_distributions(ideal_distributions_directory)

data_base_directory = '../data' 
for date_folder in sorted(os.listdir(data_base_directory)):
    date_path = os.path.join(data_base_directory, date_folder)
    if os.path.isdir(date_path):
        results = load_results(date_path)
        analyzed_results = analyze_results(results, ideal_distributions)
        result_file = os.path.join(date_path, 'analyzed_results_kl.json')
        save_results_to_json(analyzed_results, result_file)
        print(f"Analysis complete and saved to {result_file} for {date_folder}.")