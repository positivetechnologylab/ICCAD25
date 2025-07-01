import json
from collections import defaultdict

def generate_normalized_qubit_usage():
    # Load the raw computer-date-qubit usage data
    with open('../data/computer_date_qubit_usage.json', 'r') as f:
        raw_data = json.load(f)
    
    # Aggregate qubit usage counts by computer across all dates
    computer_qubit_counts = defaultdict(lambda: defaultdict(int))
    
    for date, computers in raw_data.items():
        for computer_name, qubit_counts in computers.items():
            for qubit_id, count in qubit_counts.items():
                computer_qubit_counts[computer_name][int(qubit_id)] += count
    
    # Apply min-max normalization for each computer
    normalized_usage = {}
    
    for computer_name, qubit_counts in computer_qubit_counts.items():
        # Skip if no data
        if not qubit_counts:
            continue
            
        # Get min and max counts for this computer
        min_count = min(qubit_counts.values())
        max_count = max(qubit_counts.values())
        
        # Calculate range (avoid division by zero)
        count_range = max_count - min_count
        
        # Capitalize first letter of computer name for consistency
        capitalized_name = computer_name.capitalize()
        
        # Apply min-max normalization
        normalized_usage[capitalized_name] = {}
        
        if count_range > 0:
            for qubit_id, count in qubit_counts.items():
                normalized_value = (count - min_count) / count_range
                normalized_usage[capitalized_name][str(qubit_id)] = normalized_value
        else:
            # If all counts are the same, set all normalized values to 0
            for qubit_id, count in qubit_counts.items():
                normalized_usage[capitalized_name][str(qubit_id)] = 0.0
    
    # Save the normalized data
    with open('../data/qubit_usage.json', 'w') as f:
        json.dump(normalized_usage, f, indent=4)
    
    print("Successfully generated qubit_usage.json with normalized qubit usage data")
    print(f"Processed {len(normalized_usage)} computers:")
    
    for computer_name in sorted(normalized_usage.keys()):
        num_qubits = len(normalized_usage[computer_name])
        max_usage = max(normalized_usage[computer_name].values()) if normalized_usage[computer_name] else 0
        print(f"  - {computer_name}: {num_qubits} qubits, max normalized usage: {max_usage:.4f}")

if __name__ == "__main__":
    generate_normalized_qubit_usage() 