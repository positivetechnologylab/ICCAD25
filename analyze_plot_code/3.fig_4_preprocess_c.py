import json
from collections import defaultdict
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


def calculate_single_qubit_average_tvd_for_a_day(data_base_directory):

    qubit_to_sum_tvd = defaultdict(float)
    qubit_to_count = defaultdict(int)
    qubit_to_gate_count = defaultdict(int)
    qubit_to_average_tvd = defaultdict(float)
    qubit_to_min_tvd = defaultdict(lambda: 4)
    qubit_to_max_tvd = defaultdict(float)
    qubit_to_tvd_list = defaultdict(list)
    qubit_to_std_tvd = defaultdict(float)
    qubit_to_spread_tvd = defaultdict(float)
    qubit_to_median_tvd = defaultdict(float)
    qubit_to_25th_percentile_tvd = defaultdict(float)
    qubit_to_75th_percentile_tvd = defaultdict(float)
    qubit_to_iqr_tvd = defaultdict(float)

    
    for date_folder in sorted(os.listdir(data_base_directory)):
        if not date_folder.startswith('2024'):
            continue
        date_path = os.path.join(data_base_directory, date_folder)

        print("Analyzing date: " + date_path)


        input_file_path = os.path.join(date_path, 'analyzed_results_kl.json')
        with open(input_file_path, 'r') as input_file:
            tvd_data = json.load(input_file)


        # one_qubit_data = os.path.join(date_path, "one_qubit_gate_counts.json")
        # two_qubit_data = os.path.join(date_path, "two_qubit_gate_counts.json")
        # with open(os.path.join(one_qubit_data), 'r') as f:
        #     data = json.load(f)
        #     for group in data:
        #         if group == "C":
        #             for computer_name in data[group]:
        #                     for algo_index in data[group][computer_name]:
        #                         if int(algo_index) <= 14 or int(algo_index) == 17:
        #                             for run_index in data[group][computer_name][algo_index]:
        #                                 run_tvd = tvd_data[group][computer_name][algo_index][run_index]
        #                                 for qubit, count in data[group][computer_name][algo_index][run_index].items():
        #                                     qubit_to_sum_tvd[qubit] += run_tvd # add *count
        #                                     qubit_to_count[qubit] += 1  # change 1 to count
        #                                     qubit_to_min_tvd[qubit] = min(qubit_to_min_tvd[qubit], run_tvd)
        #                                     qubit_to_max_tvd[qubit] = max(qubit_to_max_tvd[qubit], run_tvd)
        #                                     qubit_to_tvd_list[qubit].append(run_tvd)
        one_qubit_data = os.path.join(date_path, "one_qubit_gate_counts.json")
        two_qubit_data = os.path.join(date_path, "two_qubit_gate_counts.json")

        # Load one-qubit gate counts
        with open(one_qubit_data, 'r') as f:
            one_qubit_data = json.load(f)

        # Load two-qubit gate counts
        with open(two_qubit_data, 'r') as f:
            two_qubit_data = json.load(f)

        for group in one_qubit_data:
            if group == "A":
                for computer_name in one_qubit_data[group]:
                    for algo_index in one_qubit_data[group][computer_name]:
                        if int(algo_index) <= 14 or int(algo_index) == 17:
                            for run_index in one_qubit_data[group][computer_name][algo_index]:
                                run_tvd = tvd_data[group][computer_name][algo_index][run_index]
                                counted_qubits = set()  # Track qubits counted in this run

                                # Process one-qubit gates
                                for qubit, count in one_qubit_data[group][computer_name][algo_index][run_index].items():
                                    # if qubit not in counted_qubits:  # Check if qubit is already counted
                                    qubit_to_sum_tvd[(qubit,computer_name)] += run_tvd
                                    qubit_to_count[(qubit,computer_name)] += 1
                                    qubit_to_gate_count[(qubit,computer_name)] += count
                                    qubit_to_min_tvd[(qubit,computer_name)] = min(qubit_to_min_tvd[(qubit,computer_name)], run_tvd)
                                    qubit_to_max_tvd[(qubit,computer_name)] = max(qubit_to_max_tvd[(qubit,computer_name)], run_tvd)
                                    qubit_to_tvd_list[(qubit,computer_name)].append(run_tvd)
                                    counted_qubits.add((qubit,computer_name))  # Mark qubit as counted

                                # Process two-qubit gates
                                for qubit_pair, count in two_qubit_data[group][computer_name][algo_index][run_index].items():
                                    qubit1, qubit2 = map(int, qubit_pair.strip("()").split(", "))
                                    qubit1 = str(qubit1)
                                    qubit2 = str(qubit2)
                                    # qubit1, qubit2 = qubit_pair
                                    qubit_to_gate_count[(qubit1,computer_name)] += count
                                    qubit_to_gate_count[(qubit2,computer_name)] += count

                                    # Update metrics for qubit1 if not already counted
                                    if (qubit1,computer_name) not in counted_qubits:
                                        qubit_to_sum_tvd[(qubit1,computer_name)] += run_tvd
                                        qubit_to_count[(qubit1,computer_name)] += 1
                                        qubit_to_min_tvd[(qubit1,computer_name)] = min(qubit_to_min_tvd[(qubit1,computer_name)], run_tvd)
                                        qubit_to_max_tvd[(qubit1,computer_name)] = max(qubit_to_max_tvd[(qubit1,computer_name)], run_tvd)
                                        qubit_to_tvd_list[(qubit1,computer_name)].append(run_tvd)
                                        counted_qubits.add((qubit1,computer_name))

                                    # # Update metrics for qubit2 if not already counted
                                    if (qubit2, computer_name) not in counted_qubits:
                                        qubit_to_sum_tvd[(qubit2,computer_name)] += run_tvd
                                        qubit_to_count[(qubit2,computer_name)] += 1
                                        qubit_to_min_tvd[(qubit2,computer_name)] = min(qubit_to_min_tvd[(qubit2,computer_name)], run_tvd)
                                        qubit_to_max_tvd[(qubit2,computer_name)] = max(qubit_to_max_tvd[(qubit2,computer_name)], run_tvd)
                                        qubit_to_tvd_list[(qubit2,computer_name)].append(run_tvd)
                                        counted_qubits.add((qubit2,computer_name))
                                                
       
    
    for qubit_computer, tvd_list in qubit_to_tvd_list.items():
        qubit_to_average_tvd[qubit_computer] = np.mean(tvd_list)
        qubit_to_std_tvd[qubit_computer] = np.std(tvd_list, ddof=1)  # Standard deviation
        qubit_to_spread_tvd[qubit_computer] = qubit_to_max_tvd[qubit_computer] - qubit_to_min_tvd[qubit_computer] 

        qubit_to_median_tvd[qubit_computer] = np.median(tvd_list)
        qubit_to_25th_percentile_tvd[qubit_computer] = np.percentile(tvd_list, 25)
        qubit_to_75th_percentile_tvd[qubit_computer] = np.percentile(tvd_list, 75)
        qubit_to_iqr_tvd[qubit_computer] = qubit_to_75th_percentile_tvd[qubit_computer] - qubit_to_25th_percentile_tvd[qubit_computer]

    for qubit_computer, sum_tvd in qubit_to_sum_tvd.items():
        qubit_to_average_tvd[qubit_computer] = sum_tvd / qubit_to_count[qubit_computer]

    # Save raw numbers to a single CSV file
    data = []

    qubit_pairs = list(qubit_to_count.keys())
    for qubit_pair in qubit_pairs:
        qubit, computer = qubit_pair
        qubit_id = int(qubit)
        data.append([
            qubit_id,
            computer,
            qubit_to_count[qubit_pair],
            qubit_to_gate_count[qubit_pair],
            qubit_to_average_tvd[qubit_pair],
            qubit_to_min_tvd[qubit_pair],
            qubit_to_max_tvd[qubit_pair],
            qubit_to_std_tvd[qubit_pair],
            qubit_to_spread_tvd[qubit_pair],
            qubit_to_median_tvd[qubit_pair],
            qubit_to_25th_percentile_tvd[qubit_pair],
            qubit_to_75th_percentile_tvd[qubit_pair],
            qubit_to_iqr_tvd[qubit_pair]
        ])

    df = pd.DataFrame(data, columns=[
        'Qubit ID',
        'Computer',
        'Number of Algorithms Run',
        'Total Gates Run',
        'Mean KVD',
        'Min KVD',
        'Max KVD',
        'Std of KVD',
        'Spread of KVD',
        'Median KVD',
        '25th Percentile KVD',
        '75th Percentile KVD',
        'IQR of KVD'
    ])
            df.to_csv('../data/raw_numbers_with_double_qubit.csv', index=False)


    # values = np.array(list(qubit_to_average_tvd.values()), dtype=float)
    # mean = np.mean(values)
    # std_dev = np.std(values, ddof=1)
    # standardized = (values - mean) / std_dev
    # standardized_data = dict(zip(qubit_to_average_tvd.keys(), standardized))

    # min_vals = np.array(list(qubit_to_min_tvd.values()), dtype=float)
    # max_vals = np.array(list(qubit_to_max_tvd.values()), dtype=float)
    # std_vals = np.array(list(qubit_to_std_tvd.values()), dtype=float)
    # spread_vals = np.array(list(qubit_to_spread_tvd.values()), dtype=float)
    # median_vals = np.array(list(qubit_to_median_tvd.values()), dtype=float)
    # percentile_25th_vals = np.array(list(qubit_to_25th_percentile_tvd.values()), dtype=float)
    # percentile_75th_vals = np.array(list(qubit_to_75th_percentile_tvd.values()), dtype=float)
    # iqr_vals = np.array(list(qubit_to_iqr_tvd.values()), dtype=float)

    # # print the Pearson correlation between the qubit count and the average TVD
    # qubit_count = np.array(list(qubit_to_count.values()), dtype=float)
        # Get all keys in a fixed order
    all_qubits = list(qubit_to_count.keys())

    # Create arrays using the fixed key order
    values = np.array([qubit_to_average_tvd[q] for q in all_qubits], dtype=float)
    min_vals = np.array([qubit_to_min_tvd[q] for q in all_qubits], dtype=float)
    max_vals = np.array([qubit_to_max_tvd[q] for q in all_qubits], dtype=float)
    std_vals = np.array([qubit_to_std_tvd[q] for q in all_qubits], dtype=float)
    spread_vals = np.array([qubit_to_spread_tvd[q] for q in all_qubits], dtype=float)
    median_vals = np.array([qubit_to_median_tvd[q] for q in all_qubits], dtype=float)
    percentile_25th_vals = np.array([qubit_to_25th_percentile_tvd[q] for q in all_qubits], dtype=float)
    percentile_75th_vals = np.array([qubit_to_75th_percentile_tvd[q] for q in all_qubits], dtype=float)
    iqr_vals = np.array([qubit_to_iqr_tvd[q] for q in all_qubits], dtype=float)

    qubit_count = np.array([qubit_to_count[q] for q in all_qubits], dtype=float)
    gate_count = np.array([qubit_to_gate_count[q] for q in all_qubits], dtype=float)

    # Save the results to an Excel file
    results = []

    # Pearson and Spearman correlations
    # results.append(["Pearson correlation (Qubit Count vs Average TVD)", *stats.pearsonr(qubit_count, values)])
    # results.append(["Spearman correlation (Qubit Count vs Average TVD)", *stats.spearmanr(qubit_count, values)])
    
    # results.append(["Pearson correlation (Qubit Count vs Max TVD)", *stats.pearsonr(qubit_count, max_vals)])
    # results.append(["Spearman correlation (Qubit Count vs Max TVD)", *stats.spearmanr(qubit_count, max_vals)])
    
    # results.append(["Pearson correlation (Qubit Count vs Min TVD)", *stats.pearsonr(qubit_count, min_vals)])
    # results.append(["Spearman correlation (Qubit Count vs Min TVD)", *stats.spearmanr(qubit_count, min_vals)])
    
    # results.append(["Pearson correlation (Qubit Count vs Std of TVD)", *stats.pearsonr(qubit_count, std_vals)])
    # results.append(["Spearman correlation (Qubit Count vs Std of TVD)", *stats.spearmanr(qubit_count, std_vals)])
    
    # results.append(["Pearson correlation (Qubit Count vs Spread of TVD)", *stats.pearsonr(qubit_count, spread_vals)])
    # results.append(["Spearman correlation (Qubit Count vs Spread of TVD)", *stats.spearmanr(qubit_count, spread_vals)])

    # results.append(["Pearson correlation (Qubit Count vs Median TVD)", *stats.pearsonr(qubit_count, median_vals)])
    # results.append(["Spearman correlation (Qubit Count vs Median TVD)", *stats.spearmanr(qubit_count, median_vals)])

    # results.append(["Pearson correlation (Qubit Count vs 25th Percentile TVD)", *stats.pearsonr(qubit_count, percentile_25th_vals)])
    # results.append(["Spearman correlation (Qubit Count vs 25th Percentile TVD)", *stats.spearmanr(qubit_count, percentile_25th_vals)])

    # results.append(["Pearson correlation (Qubit Count vs 75th Percentile TVD)", *stats.pearsonr(qubit_count, percentile_75th_vals)])
    # results.append(["Spearman correlation (Qubit Count vs 75th Percentile TVD)", *stats.spearmanr(qubit_count, percentile_75th_vals)])

    # results.append(["Pearson correlation (Qubit Count vs 75th-25th TVD)", *stats.pearsonr(qubit_count, iqr_vals)])
    # results.append(["Spearman correlation (Qubit Count vs 75th-25th TVD)", *stats.spearmanr(qubit_count, iqr_vals)])


    results.append(["Pearson correlation (Gate Count vs Average TVD)", *stats.pearsonr(gate_count, values)])
    results.append(["Spearman correlation (Gate Count vs Average TVD)", *stats.spearmanr(gate_count, values)])
    
    results.append(["Pearson correlation (Gate Count vs Max TVD)", *stats.pearsonr(gate_count, max_vals)])
    results.append(["Spearman correlation (Gate Count vs Max TVD)", *stats.spearmanr(gate_count, max_vals)])
    
    results.append(["Pearson correlation (Gate Count vs Min TVD)", *stats.pearsonr(gate_count, min_vals)])
    results.append(["Spearman correlation (Gate Count vs Min TVD)", *stats.spearmanr(gate_count, min_vals)])
    
    results.append(["Pearson correlation (Gate Count vs Std of TVD)", *stats.pearsonr(gate_count, std_vals)])
    results.append(["Spearman correlation (Gate Count vs Std of TVD)", *stats.spearmanr(gate_count, std_vals)])
    
    results.append(["Pearson correlation (Gate Count vs Spread of TVD)", *stats.pearsonr(gate_count, spread_vals)])
    results.append(["Spearman correlation (Gate Count vs Spread of TVD)", *stats.spearmanr(gate_count, spread_vals)])

    results.append(["Pearson correlation (Qubit Count vs Median TVD)", *stats.pearsonr(gate_count, median_vals)])
    results.append(["Spearman correlation (Qubit Count vs Median TVD)", *stats.spearmanr(gate_count, median_vals)])

    results.append(["Pearson correlation (Qubit Count vs 25th Percentile TVD)", *stats.pearsonr(gate_count, percentile_25th_vals)])
    results.append(["Spearman correlation (Qubit Count vs 25th Percentile TVD)", *stats.spearmanr(gate_count, percentile_25th_vals)])

    results.append(["Pearson correlation (Qubit Count vs 75th Percentile TVD)", *stats.pearsonr(gate_count, percentile_75th_vals)])
    results.append(["Spearman correlation (Qubit Count vs 75th Percentile TVD)", *stats.spearmanr(gate_count, percentile_75th_vals)])

    results.append(["Pearson correlation (Qubit Count vs 75th-25th TVD)", *stats.pearsonr(gate_count, iqr_vals)])
    results.append(["Spearman correlation (Qubit Count vs 75th-25th TVD)", *stats.spearmanr(gate_count, iqr_vals)])

    # Create a DataFrame for saving the results
    df_results = pd.DataFrame(results, columns=["Description", "Correlation Coefficient", "P-Value"])

    # Save the results to an Excel file
    output_file = 'correlation_results_kl.xlsx'
    df_results.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Create dictionaries mapping qubit usage to their metrics
    # Map qubit counts to their corresponding qubit IDs
    qubit_usage_dict = {}
    for qubit_id, count in zip(qubit_to_count.keys(), qubit_count):
        qubit_usage_dict[str(count)] = qubit_id

    # Create lists of exponential values
    exp_values = [np.exp(val) for val in values]
    exp_std_vals = [np.exp(val) for val in std_vals]
    exp_min_vals = [np.exp(val) for val in min_vals]
    exp_max_vals = [np.exp(val) for val in max_vals] 
    exp_spread_vals = [np.exp(val) for val in spread_vals]
    # exp_std_vals = [val for val in std_vals]
    # exp_min_vals = [val for val in min_vals]
    # exp_max_vals = [val for val in max_vals] 
    # exp_spread_vals = [val for val in spread_vals]

    # count_vals = zip(qubit_count, values)
    # std_vals = zip(qubit_count, exp_std_vals)
    # min_vals = zip(qubit_count, exp_min_vals)
    # max_vals = zip(qubit_count, exp_max_vals)
    # spread_vals = zip(qubit_count, exp_spread_vals)

    count_vals = zip(gate_count, exp_values)
    std_vals = zip(gate_count, exp_std_vals)
    min_vals = zip(gate_count, exp_min_vals)
    max_vals = zip(gate_count, exp_max_vals)
    spread_vals = zip(gate_count, exp_spread_vals)

    # count_vals = [x for x in count_vals if x[0] < 1000]
    # std_vals = [x for x in std_vals if x[0] < 1000]
    # min_vals = [x for x in min_vals if x[0] < 1000 and x[1] < 10]
    # max_vals = [x for x in max_vals if x[0] < 1000]
    # spread_vals = [x for x in spread_vals if x[0] < 1000]


    count_vals = [x for x in count_vals]
    std_vals = [x for x in std_vals if x[1] < 6000]
    min_vals = [x for x in min_vals if x[1] < 1.001]
    max_vals = [x for x in max_vals]
    spread_vals = [x for x in spread_vals]

    # Get first values (counts) from each list separately since they may differ
    counts_mean = [x[0] for x in count_vals]
    counts_std = [x[0] for x in std_vals]
    counts_min = [x[0] for x in min_vals] 
    counts_max = [x[0] for x in max_vals]
    counts_spread = [x[0] for x in spread_vals]

    # Get second values (metrics) from each list
    means = [x[1] for x in count_vals]
    stds = [x[1] for x in std_vals]
    mins = [x[1] for x in min_vals]
    maxs = [x[1] for x in max_vals]
    spreads = [x[1] for x in spread_vals]

    # Normalize counts for each metric separately
    mean_count_max = max(counts_mean)
    std_count_max = max(counts_std)
    min_count_max = max(counts_min)
    max_count_max = max(counts_max)
    spread_count_max = max(counts_spread)

    normalized_counts_mean = [(c - min(counts_mean)) / (mean_count_max - min(counts_mean)) for c in counts_mean]
    normalized_counts_std = [(c - min(counts_std)) / (std_count_max - min(counts_std)) for c in counts_std]
    normalized_counts_min = [(c - min(counts_min)) / (min_count_max - min(counts_min)) for c in counts_min]
    normalized_counts_max = [(c - min(counts_max)) / (max_count_max - min(counts_max)) for c in counts_max]
    normalized_counts_spread = [(c - min(counts_spread)) / (spread_count_max - min(counts_spread)) for c in counts_spread]

    # Normalize metric values
    mean_max = max(means)
    std_max = max(stds)
    min_max = max(mins)
    max_max = max(maxs)
    spread_max = max(spreads)

    # Min-max normalization for metric values
    normalized_means = [(m - min(means)) / (mean_max - min(means)) for m in means]
    normalized_stds = [(s - min(stds)) / (std_max - min(stds)) for s in stds]
    normalized_mins = [(m - min(mins)) / (min_max - min(mins)) for m in mins]
    normalized_maxs = [(m - min(maxs)) / (max_max - min(maxs)) for m in maxs]
    normalized_spreads = [(s - min(spreads)) / (spread_max - min(spreads)) for s in spreads]

    # Recombine into tuples
    average_vals = list(zip(normalized_counts_mean, normalized_means))
    std_vals = list(zip(normalized_counts_std, normalized_stds))
    min_vals = list(zip(normalized_counts_min, normalized_mins))
    max_vals = list(zip(normalized_counts_max, normalized_maxs))
    spread_vals = list(zip(normalized_counts_spread, normalized_spreads))

    np.save('../data/average_vals.npy', np.array(average_vals))
    np.save('../data/std_vals.npy', np.array(std_vals))
    np.save('../data/min_vals.npy', np.array(min_vals))
    np.save('../data/max_vals.npy', np.array(max_vals))
    np.save('../data/spread_vals.npy', np.array(spread_vals))


    # Assuming `qubit_count` and `max_vals` are already populated from your code
    # qubit_count: array of qubit counts
    # max_vals: array of maximum TVD values corresponding to each qubit

    # Create the plot
    # plt.figure(figsize=(10, 6))
    # # plt.scatter(qubit_count, [np.exp(val) for val in spread_vals], color='b', label='Spread TVD')
    # # plt.scatter(normalized_counts, norm_spread_vals, color='b', label='Spread TVD')
    # # plt.scatter(normalized_counts, values, color='b', label='Mean TVD')
    # plt.scatter([x[0] for x in spread_vals], [x[1] for x in spread_vals], color='r', label='Count')

    # # Add labels and title
    # plt.xlabel('Qubit Count')
    # plt.ylabel('Spread TVD')
    # plt.title('Spread TVD vs Qubit Count')
    # plt.grid(True)
    # plt.legend()

    # # Show the plot
    # plt.show()


data_base_directory = '.'
calculate_single_qubit_average_tvd_for_a_day(data_base_directory)