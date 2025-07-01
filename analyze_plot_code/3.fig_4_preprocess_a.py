import json
import os
from collections import defaultdict
import numpy as np
import re

NUM_ALGOS = 20

def calculate_tvd(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))

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
                        tvd = calculate_tvd(prob_dist, ideal_dist)
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
        result_file = os.path.join(date_path, 'analyzed_results.json')
        save_results_to_json(analyzed_results, result_file)
        print(f"Analysis complete and saved to {result_file} for {date_folder}.")