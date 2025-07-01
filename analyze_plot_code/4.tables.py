import json
import os
from collections import defaultdict
from scipy import stats
COMPUTERS = ['Brisbane', 'Cusco', 'Kyiv', 'Kyoto', 'Nazca']
NUM_QUBITS = 127

def calculate_qubit_usage_to_qubit_properties(data_base_directory):

    qubit_properties = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'T1': 0, 'T2': 0, 'frequency': 0,
        'readout_error': 0, 'readout_length': 0,
        'sx_error': 0, 'sx_length': 0,
        'ecr_error': 0, 'ecr_length': 0,
        'days_count': 0
    })))

    days_count = 0
    for date_folder in sorted(os.listdir(data_base_directory)):
        if not date_folder.startswith('2024'):
            continue
        date_path = os.path.join(data_base_directory, date_folder)
        date = date_path[2:]

        for file_name in os.listdir(date_path):
            if file_name.startswith("quantum_computer_properties"):
                qubit_properties_file_path = os.path.join(date_path, file_name)
                with open(qubit_properties_file_path, 'r') as f:
                    raw_properties = json.load(f)
                    computers = raw_properties.keys()
                    for computer_key in computers:
                        computer = computer_key[4:]
                        for qubit_data in raw_properties[computer_key]['qubit_properties']:
                            qubit_id = qubit_data['qubit']
                            qubit_properties[date][computer][qubit_id]['T1'] += qubit_data['T1']
                            qubit_properties[date][computer][qubit_id]['T2'] += qubit_data['T2']
                            qubit_properties[date][computer][qubit_id]['frequency'] += qubit_data['frequency']
                            qubit_properties[date][computer][qubit_id]['readout_error'] += qubit_data['readout_error']
                            qubit_properties[date][computer][qubit_id]['readout_length'] += qubit_data['readout_length']
                            qubit_properties[date][computer][qubit_id]['sx_error'] += qubit_data['sx_error']
                            qubit_properties[date][computer][qubit_id]['sx_length'] += qubit_data['sx_length']
                            qubit_properties[date][computer][qubit_id]['days_count'] += 1
                        for two_qubit_data in raw_properties[computer_key]['two_qubit_properties']:
                            qubit_properties[date][computer][two_qubit_data['control']]['ecr_error'] += two_qubit_data['ecr_error']
                            qubit_properties[date][computer][two_qubit_data['control']]['ecr_length'] += two_qubit_data['ecr_length']
                            qubit_properties[date][computer][two_qubit_data['target']]['ecr_error'] += two_qubit_data['ecr_error']
                            qubit_properties[date][computer][two_qubit_data['target']]['ecr_length'] += two_qubit_data['ecr_length']
                break

    mean_qubit_properties = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'T1': 0, 'T2': 0, 'frequency': 0,
        'readout_error': 0, 'readout_length': 0,
        'sx_error': 0, 'sx_length': 0,
        'ecr_error': 0, 'ecr_length': 0,
        'days_count': 0
    })))
    for date in qubit_properties:
        for computer in qubit_properties[date]:
            for qubit_id in qubit_properties[date][computer]:
                mean_qubit_properties[date][computer][qubit_id] = {
                    'T1': qubit_properties[date][computer][qubit_id]['T1'] / qubit_properties[date][computer][qubit_id]['days_count'],
                    'T2': qubit_properties[date][computer][qubit_id]['T2'] / qubit_properties[date][computer][qubit_id]['days_count'],
                    'frequency': qubit_properties[date][computer][qubit_id]['frequency'] / qubit_properties[date][computer][qubit_id]['days_count'],
                    'readout_error': qubit_properties[date][computer][qubit_id]['readout_error'] / qubit_properties[date][computer][qubit_id]['days_count'],
                    'readout_length': qubit_properties[date][computer][qubit_id]['readout_length'] / qubit_properties[date][computer][qubit_id]['days_count'],
                    'sx_error': qubit_properties[date][computer][qubit_id]['sx_error'] / qubit_properties[date][computer][qubit_id]['days_count'],
                    'sx_length': qubit_properties[date][computer][qubit_id]['sx_length'] / qubit_properties[date][computer][qubit_id]['days_count'],
                    'ecr_error': qubit_properties[date][computer][qubit_id]['ecr_error'] / qubit_properties[date][computer][qubit_id]['days_count'],
                    'ecr_length': qubit_properties[date][computer][qubit_id]['ecr_length'] / qubit_properties[date][computer][qubit_id]['days_count']
                }

    # Load qubit usage data
    with open('../data/computer_date_qubit_usage.json', 'r') as f:
        qubit_usage = json.load(f)

    all_usages = []
    all_T1 = []
    all_T2 = []
    all_frequency = []
    all_readout_error = []
    all_readout_length = []
    all_sx_error = []
    all_sx_length = []
    all_ecr_error = []
    all_ecr_length = []

    for date in mean_qubit_properties:
        for computer in mean_qubit_properties[date]:
            for qubit_id in mean_qubit_properties[date][computer]:
                properties = mean_qubit_properties[date][computer][qubit_id]
                print(computer, date, qubit_id)
                usage = 0
                if computer in qubit_usage[date]:
                    if str(qubit_id) in qubit_usage[date][computer.lower()]:
                        usage = qubit_usage[date][computer.lower()][str(qubit_id)]
                    else:
                        usage = 0
                else:
                    continue
                print(usage)

                all_usages.append(usage)
                all_T1.append(properties['T1'])
                all_T2.append(properties['T2'])
                all_frequency.append(properties['frequency'])
                all_readout_error.append(properties['readout_error'])
                all_readout_length.append(properties['readout_length'])
                all_sx_error.append(properties['sx_error'])
                all_sx_length.append(properties['sx_length'])
                all_ecr_error.append(properties['ecr_error'])
                all_ecr_length.append(properties['ecr_length'])


    correlations = {
        'T1': {
            'correlation': stats.spearmanr(all_usages, all_T1).correlation,
            'p_value': stats.spearmanr(all_usages, all_T1).pvalue
        },
        'T2': {
            'correlation': stats.spearmanr(all_usages, all_T2).correlation,
            'p_value': stats.spearmanr(all_usages, all_T2).pvalue
        },
        'frequency': {
            'correlation': stats.spearmanr(all_usages, all_frequency).correlation,
            'p_value': stats.spearmanr(all_usages, all_frequency).pvalue
        },
        'readout_error': {
            'correlation': stats.spearmanr(all_usages, all_readout_error).correlation,
            'p_value': stats.spearmanr(all_usages, all_readout_error).pvalue
        },
        'readout_length': {
            'correlation': stats.spearmanr(all_usages, all_readout_length).correlation,
            'p_value': stats.spearmanr(all_usages, all_readout_length).pvalue
        },
        'sx_error': {
            'correlation': stats.spearmanr(all_usages, all_sx_error).correlation,
            'p_value': stats.spearmanr(all_usages, all_sx_error).pvalue
        },
        'sx_length': {
            'correlation': stats.spearmanr(all_usages, all_sx_length).correlation,
            'p_value': stats.spearmanr(all_usages, all_sx_length).pvalue
        },
        'ecr_error': {
            'correlation': stats.spearmanr(all_usages, all_ecr_error).correlation,
            'p_value': stats.spearmanr(all_usages, all_ecr_error).pvalue
        },
        'ecr_length': {
            'correlation': stats.spearmanr(all_usages, all_ecr_length).correlation,
            'p_value': stats.spearmanr(all_usages, all_ecr_length).pvalue
        }
    }
    print(correlations)
    
    # Save the correlations to a JSON file
    with open('../data/qubit_usage_correlations.json', 'w') as f:
        json.dump(correlations, f, indent=4)


data_base_directory = '../data'
calculate_qubit_usage_to_qubit_properties(data_base_directory)

import json
import glob
import os
from datetime import datetime
import numpy as np
from collections import defaultdict
from scipy import stats

def calculate_kld(P, Q):
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

def get_time_bin(hour):
    """Convert hour to 3-hour bin (0-7)."""
    return int(hour // 3)

def min_max_normalize(data):
    """Normalize data to 0-1 range."""
    if not data:
        return data
    min_val = min(data)
    max_val = max(data)
    if min_val == max_val:
        return [0.5 for _ in data]  # If all values are the same, return 0.5
    return [(x - min_val) / (max_val - min_val) for x in data]

def process_time_fidelity_data(start_date, end_date, base_path="."):
    """Process files and collect KLD data with completion times."""
    EXCLUDED_INDICES = {15, 16, 18, 19}
    VALID_INDICES = sorted([i for i in range(20) if i not in EXCLUDED_INDICES])
    
    # Load ideal distributions
    ideal_distributions = {}
    for idx in VALID_INDICES:
        try:
            with open(f'../input_algorithms/ideal_distributions/ideal_distribution_{idx}.json') as f:
                ideal_distributions[idx] = json.load(f)
        except Exception as e:
            continue
    
    # Initialize data structures
    computer_algo_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Get files for date range
    files = []
    for date in range(int(start_date), int(end_date) + 1):
        files.extend(glob.glob(os.path.join(f'../data/{base_path}', str(date), 'group_C_*.json')))
    
    # Collect KLD values and timestamps for each computer-algorithm pair
    for file in files:
        try:
            with open(file) as f:
                data = json.load(f)
            
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                data = [data]
            
            for entry in data:
                if isinstance(entry, str):
                    continue
                
                completion_time = entry.get('completion_time')
                idx = entry.get('algorithm_index')
                counts = entry.get('counts')
                computer = entry.get('computer')
                
                if completion_time and idx in VALID_INDICES and counts and computer:
                    ideal_dist = ideal_distributions.get(idx)
                    
                    if ideal_dist:
                        try:
                            dt = datetime.fromisoformat(completion_time)
                            hour = dt.hour + dt.minute/60.0 + dt.second/3600.0
                            kld = calculate_kld(ideal_dist, counts)
                            
                            # Store raw hour and KLD for correlation analysis
                            computer_algo_data[computer][idx]['hours'].append(hour)
                            computer_algo_data[computer][idx]['klds'].append(kld)
                            
                        except Exception:
                            continue
                            
        except Exception:
            continue
    
    return computer_algo_data

def calculate_correlations(computer_algo_data):
    """Calculate Spearman correlation between time and KLD for each algorithm across all computers."""
    algo_correlations = {}
    
    # Collect data for each algorithm across all computers
    for computer in computer_algo_data:
        for algo_idx, data in computer_algo_data[computer].items():
            if algo_idx not in algo_correlations:
                algo_correlations[algo_idx] = {'hours': [], 'klds': []}
            
            # Get hours and KLDs
            hours = data['hours']
            klds = data['klds']
            
            # Store data, keeping track of which computer it came from
            algo_correlations[algo_idx]['hours'].extend(hours)
            algo_correlations[algo_idx]['klds'].extend(klds)
    
    # Calculate correlations
    results = {}
    for algo_idx, data in algo_correlations.items():
        if data['hours'] and data['klds']:
            # Normalize KLDs for fair comparison across computers
            normalized_klds = min_max_normalize(data['klds'])
            
            # Calculate correlation between time of day and normalized KLD
            correlation, p_value = stats.spearmanr(data['hours'], normalized_klds)
            results[algo_idx] = {
                'correlation': correlation,
                'p_value': p_value
            }
            
            # Print additional info for verification
            print(f"Algorithm {algo_idx}:")
            print(f"Total data points: {len(data['hours'])}")
            print(f"Number of unique hours: {len(set(data['hours']))}")
            print(f"Correlation: {correlation:.2f}")
            print(f"P-value: {p_value:.2e}")
            print("---")
    
    return results

def print_latex_table(correlations):
    """Print correlations in LaTeX table format."""
    print(r"\begin{table}[t]")
    print(r"    \centering")
    print(r"    \caption{Spearman correlation between time of day and KLD values for each algorithm across all computers.}")
    print(r"    \vspace{-1mm}")
    print(r"    \scalebox{0.9}{")
    print(r"    \begin{tabular}{llllll}")
    print(r"    \hline")
    print(r"    \textbf{Algorithm} & \textbf{Corr.} & \textbf{$p$-value} & \textbf{Algorithm} & \textbf{Corr.} & \textbf{$p$-value} \\")
    print(r"    \hline")
    
    # Sort algorithms by absolute correlation strength
    sorted_algos = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
    
    # Split into two columns
    half = len(sorted_algos) // 2 + len(sorted_algos) % 2
    for i in range(half):
        left_algo = sorted_algos[i]
        right_algo = sorted_algos[i + half] if i + half < len(sorted_algos) else None
        
        left_str = f"    {left_algo[0]} & {left_algo[1]['correlation']:.2f} & {left_algo[1]['p_value']:.2e}"
        if right_algo:
            right_str = f"& {right_algo[0]} & {right_algo[1]['correlation']:.2f} & {right_algo[1]['p_value']:.2e}"
        else:
            right_str = "& & &"
        
        print(f"{left_str} {right_str} \\\\")
    
    print(r"    \hline")
    print(r"    \end{tabular}}")
    print(r"    \vspace{-5mm}")
    print(r"    \label{tab:time_correls}")
    print(r"\end{table}")

def main():
    # Process data
    computer_algo_data = process_time_fidelity_data("20240810", "20240822")
    
    # Calculate correlations
    correlations = calculate_correlations(computer_algo_data)
    
    # Print LaTeX table
    print_latex_table(correlations)

if __name__ == "__main__":
    main()