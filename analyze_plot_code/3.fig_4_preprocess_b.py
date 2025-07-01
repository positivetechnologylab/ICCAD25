import json
from collections import defaultdict
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats


def load_and_count_2_qubit_gate(data_directory):

    two_qubit_gate_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))

    two_qubit_gate_pattern = re.compile(r'q\[(\d+)\],q\[(\d+)\]')

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
                
                    qasm = item['transpiled_qasm']
                    matches = two_qubit_gate_pattern.findall(qasm)
                    
                    for match in matches:
                        qubit_pair = str(tuple(sorted((int(match[0]), int(match[1])))))
                        two_qubit_gate_counts[group][computer_name][algo_index][run_index][qubit_pair] += 1
                    

    # Output the results to a JSON file in the same directory
    output_file_path = os.path.join(data_directory, 'two_qubit_gate_counts.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(two_qubit_gate_counts, output_file, indent=4)

    print(f"Two-qubit gate counts saved to {output_file_path}")

def load_and_count_1_qubit_gate(data_directory):

    one_qubit_gate_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))
    one_qubit_gate_pattern = re.compile(r'(?<!\,)q\[(\d+)\](?!,)')

    for file in os.listdir(data_directory):
        if file.startswith('group') and file.endswith('.json'):
            computer_name_match = re.search(r'_([a-zA-Z]+)_\d{8}_', file)
            computer_name = computer_name_match.group(1) if computer_name_match else 'unknown'
            with open(os.path.join(data_directory, file), 'r') as f:
                data = json.load(f)
                for item in data:
                    group = item.get('group')
                    algo_index = item.get('algorithm_index')
                    run_index = item.get('run_index')                
                    if group == "B":
                        run_index = item.get('optimization_level')


                    qasm = item['transpiled_qasm']
                    matches = one_qubit_gate_pattern.findall(qasm)

                    for match in matches:
                        qubit = int(match)
                        one_qubit_gate_counts[group][computer_name][algo_index][run_index][qubit] += 1

    # Output the results to a JSON file in the same directory
    output_file_path = os.path.join(data_directory, 'one_qubit_gate_counts.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(one_qubit_gate_counts, output_file, indent=4)

    print(f"One-qubit gate counts saved to {output_file_path}")


def calculate_single_qubit_average_tvd_for_a_day(data_base_directory):
    print(data_base_directory)

    qubit_to_sum_tvd = defaultdict(float)
    qubit_to_count = defaultdict(int)
    qubit_to_average_tvd = defaultdict(float)

    input_file_path = os.path.join(data_base_directory, 'analyzed_results.json')
    with open(input_file_path, 'r') as input_file:
        tvd_data = json.load(input_file)


    one_qubit_data = os.path.join(data_base_directory, "one_qubit_gate_counts.json")
    with open(os.path.join(one_qubit_data), 'r') as f:
        data = json.load(f)
        for group in data:
            if group == "C":
                for computer_name in data[group]:
                        for algo_index in data[group][computer_name]:
                            if int(algo_index) <=13 :
                                for run_index in data[group][computer_name][algo_index]:
                                    run_tvd = tvd_data[group][computer_name][algo_index][run_index]
                                    for qubit, count in data[group][computer_name][algo_index][run_index].items():
                                        qubit_to_sum_tvd[qubit] += run_tvd * count # add *count
                                        qubit_to_count[qubit] += count  # change 1 to count 
    for qubit, sum_tvd in qubit_to_sum_tvd.items():
        qubit_to_average_tvd[qubit] = sum_tvd / qubit_to_count[qubit]
    

    values = np.array(list(qubit_to_average_tvd.values()), dtype=float)
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1)
    standardized = (values - mean) / std_dev
    standardized_data = dict(zip(qubit_to_average_tvd.keys(), standardized))


    output_file_path = os.path.join(data_base_directory, 'one_qubit_average_tvd.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(standardized_data, output_file, indent=4)
    print(f"One-qubit average TVD saved to {output_file_path}")
    

def calculate_two_qubit_average_tvd_for_a_day(data_base_directory):
    print(data_base_directory)

    qubit_to_sum_tvd = defaultdict(float)
    qubit_to_count = defaultdict(int)
    qubit_to_average_tvd = defaultdict(float)

    input_file_path = os.path.join(data_base_directory, 'analyzed_results.json')
    with open(input_file_path, 'r') as input_file:
        tvd_data = json.load(input_file)


    two_qubit_data = os.path.join(data_base_directory, "two_qubit_gate_counts.json")
    with open(os.path.join(two_qubit_data), 'r') as f:
        data = json.load(f)
        for group in data:
            if group != "B":
                for computer_name in data[group]:
                    if computer_name == "nazca":
                        for algo_index in data[group][computer_name]:
                            if int(algo_index) <=13 :
                                for run_index in data[group][computer_name][algo_index]:
                                    run_tvd = tvd_data[group][computer_name][algo_index][run_index]
                                    for qubit, count in data[group][computer_name][algo_index][run_index].items():
                                        qubit_to_sum_tvd[qubit] += run_tvd *count # add *count
                                        qubit_to_count[qubit] += count  # change 1 to count
    for qubit, sum_tvd in qubit_to_sum_tvd.items():
        qubit_to_average_tvd[qubit] = sum_tvd / qubit_to_count[qubit]
    

    values = np.array(list(qubit_to_average_tvd.values()), dtype=float)
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1)
    standardized = (values - mean) / std_dev
    standardized_data = dict(zip(qubit_to_average_tvd.keys(), standardized))

    output_file_path = os.path.join(data_base_directory, 'two_qubit_average_tvd.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(standardized_data, output_file, indent=4)
    print(f"One-qubit average TVD saved to {output_file_path}")

def plot_single_qubit_average_tvd(data_base_directory):
    standardized_data = {}

    # Load the standardized data from the JSON file
    input_file_path = os.path.join(data_base_directory, 'one_qubit_average_tvd.json')
    with open(input_file_path, 'r') as input_file:
        standardized_data = json.load(input_file)
    
    # Convert data to a sorted list of tuples for plotting
    qubits, standardized_tvd = zip(*sorted(standardized_data.items(), key=lambda x: int(x[0])))
    standardized_tvd = np.array(standardized_tvd, dtype=float)

    # Create a plot
    plt.figure(figsize=(16, 10))
    sns.barplot(x=list(qubits), y=standardized_tvd, palette="viridis")

    plt.title('Standardized Single Qubit Average TVD')
    plt.xlabel('Qubit')
    plt.ylabel('Standardized TVD')
    plt.xticks(rotation=90)
    
    # Save the plot
    plot_output_path = os.path.join('../plots', 'single_qubit_average_tvd_standardized_plot.png')
    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close()
    
    print(f"Plot saved to {plot_output_path}")


def plot_two_qubit_average_tvd_heatmap(data_base_directory):
    # Load the standardized data from the JSON file
    input_file_path = os.path.join(data_base_directory, 'two_qubit_average_tvd.json')
    with open(input_file_path, 'r') as input_file:
        standardized_data = json.load(input_file)
    
    # Convert the standardized data into a 2D matrix format
    qubit_pairs = standardized_data.keys()
    qubit_indices = set()
    for pair in qubit_pairs:
        q1, q2 = eval(pair)
        qubit_indices.add(q1)
        qubit_indices.add(q2)

    # Sort the qubits to create a consistent order
    sorted_qubits = sorted(qubit_indices)
    qubit_index_map = {qubit: i for i, qubit in enumerate(sorted_qubits)}

    # Initialize a matrix of NaNs (or zeros) with size based on the number of qubits
    matrix_size = len(sorted_qubits)
    heatmap_matrix = np.full((matrix_size, matrix_size), np.nan)

    # Fill the matrix with standardized TVD values
    for pair, value in standardized_data.items():
        q1, q2 = eval(pair)
        i, j = qubit_index_map[q1], qubit_index_map[q2]
        heatmap_matrix[i, j] = value

    # Create a heatmap plot
    plt.figure(figsize=(16, 14))
    sns.heatmap(heatmap_matrix, annot=False, fmt=".2f", cmap="Reds", 
                xticklabels=sorted_qubits, yticklabels=sorted_qubits)

    plt.title('Standardized Two-Qubit Average TVD Heatmap')
    plt.xlabel('Qubit')
    plt.ylabel('Qubit')
    
    # Save the plot
    plot_output_path = os.path.join('../plots', 'two_qubit_average_tvd_standardized_heatmap.png')
    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close()
    
    print(f"Two-qubit TVD heatmap saved to {plot_output_path}")


def calculate_coefficient(data_base_directory):

    input_file_path = os.path.join(data_base_directory, 'one_qubit_average_tvd.json')
    with open(input_file_path, 'r') as input_file:
        standardized_data = json.load(input_file)
    
    property_data = {}
    for file in os.listdir(data_base_directory):
        if file.startswith('quantum_computer_properties') and file.endswith('.json'):
            with open(os.path.join(data_base_directory, file), 'r') as f:
                property_data = json.load(f)
                break
                
    for computer_name, properties in property_data.items():
        if computer_name == "ibm_nazca":
            qubit_properties = properties.get('qubit_properties')
            break

    valid_qubits = [prop for prop in qubit_properties if str(prop["qubit"]) in standardized_data]
    qubits = [prop["qubit"] for prop in valid_qubits]

    # Extract properties and TVD values for valid qubits
    T1_values = [prop["T1"] for prop in valid_qubits]
    T2_values = [prop["T2"] for prop in valid_qubits]
    frequency_values = [prop["frequency"] for prop in valid_qubits]
    readout_error_values = [prop["readout_error"] for prop in valid_qubits]
    tvd = [standardized_data[str(prop["qubit"])] for prop in valid_qubits]

    print(T1_values)
    print(tvd)
    r_value, p_value = stats.pearsonr(T1_values, tvd)
    print(f"Pearson correlation between T1 and TVD: {r_value:.3f} (p-value: {p_value:.3f})")
    r_value, p_value = stats.pearsonr(T2_values, tvd)
    print(f"Pearson correlation between T2 and TVD: {r_value:.3f} (p-value: {p_value:.3f})")
    r_value, p_value = stats.pearsonr(frequency_values, tvd)
    print(f"Pearson correlation between frequency and TVD: {r_value:.3f} (p-value: {p_value:.3f})")
    r_value, p_value = stats.pearsonr(readout_error_values, tvd)
    print(f"Pearson correlation between readout error and TVD: {r_value:.3f} (p-value: {p_value:.3f})")

    spearman_corr_T1, spearman_p_T1 = stats.spearmanr(T1_values, tvd)
    spearman_corr_T2, spearman_p_T2 = stats.spearmanr(T2_values, tvd)
    spearman_corr_freq, spearman_p_freq = stats.spearmanr(frequency_values, tvd)
    spearman_corr_readout, spearman_p_readout = stats.spearmanr(readout_error_values, tvd)
    print(f"Spearman correlation between T1 and TVD: {spearman_corr_T1:.3f} (p-value: {spearman_p_T1:.3f})")
    print(f"Spearman correlation between T2 and TVD: {spearman_corr_T2:.3f} (p-value: {spearman_p_T2:.3f})")
    print(f"Spearman correlation between frequency and TVD: {spearman_corr_freq:.3f} (p-value: {spearman_p_freq:.3f})")
    print(f"Spearman correlation between readout error and TVD: {spearman_corr_readout:.3f} (p-value: {spearman_p_readout:.3f})")



data_base_directory = '../data'
for date_folder in sorted(os.listdir(data_base_directory)):
    date_path = os.path.join(data_base_directory, date_folder)
    print(date_path)
    if os.path.isdir(date_path):
        load_and_count_1_qubit_gate(date_path)
        load_and_count_2_qubit_gate(date_path)
        calculate_single_qubit_average_tvd_for_a_day(date_path)
