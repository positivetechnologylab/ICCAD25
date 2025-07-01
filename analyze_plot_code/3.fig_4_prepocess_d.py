import json
from collections import defaultdict
import os

def get_qubit_usage(data_base_directory):

    computer_qubit_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # computer_qubit_count = defaultdict(lambda: defaultdict(int))

    for date_folder in sorted(os.listdir(data_base_directory)):
        if not date_folder.startswith('2024'):
            continue
        date_path = os.path.join(data_base_directory, date_folder)
        date = date_path[2:]


        one_qubit_data = os.path.join(date_path, "one_qubit_gate_counts.json")
        two_qubit_data = os.path.join(date_path, "two_qubit_gate_counts.json")

        with open(one_qubit_data, 'r') as f:
            one_qubit_data = json.load(f)

        with open(two_qubit_data, 'r') as f:
            two_qubit_data = json.load(f)

        for group in one_qubit_data:
            if group == "C":
                for computer_name in one_qubit_data[group]:
                    for algo_index in one_qubit_data[group][computer_name]:
                        if int(algo_index) <= 14 or int(algo_index) == 17:
                            for run_index in one_qubit_data[group][computer_name][algo_index]:
                                for qubit, count in one_qubit_data[group][computer_name][algo_index][run_index].items():
                                    # computer_qubit_count[computer_name][int(qubit)] += count
                                    computer_qubit_count[date][computer_name][int(qubit)] += count
                                
                                for qubit_pair, count in two_qubit_data[group][computer_name][algo_index][run_index].items():
                                    qubit1, qubit2 = map(int, qubit_pair.strip("()").split(", "))
                                    # computer_qubit_count[computer_name][int(qubit1)] += count
                                    # computer_qubit_count[computer_name][int(qubit2)] += count
                                    computer_qubit_count[date][computer_name][int(qubit1)] += count
                                    computer_qubit_count[date][computer_name][int(qubit2)] += count
                                
    # Normalize counts for each computer using max-min normalization
    # normalized_counts = defaultdict(dict)
    # for computer in computer_qubit_count:
    #     max_count = max(computer_qubit_count[computer].values())
    #     min_count = min(computer_qubit_count[computer].values())
    #     range_count = max_count - min_count
    #     if range_count > 0:  # Avoid division by zero
    #         for qubit, count in computer_qubit_count[computer].items():
    #             # Capitalize first letter of computer name
    #             computer_name = computer[0].upper() + computer[1:]
    #             normalized_counts[computer_name][qubit] = (count - min_count) / range_count
    # return computer_qubit_count
    return computer_qubit_count

data_base_directory = '.'
qubit_usage = get_qubit_usage(data_base_directory)

# Convert defaultdict to regular dict for JSON serialization
qubit_usage = {computer: dict(qubits) for computer, qubits in qubit_usage.items()}

# Save to JSON file
with open('../data/computer_date_qubit_usage.json', 'w') as f:
    json.dump(qubit_usage, f, indent=4)

print("Results saved to qubit_usage.json")