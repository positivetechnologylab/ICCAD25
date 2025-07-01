import os
import json
import re
import csv
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

NUM_SHOTS = 10000  # Set the number of shots to 10000
MAX_QUBITS = 20  # Maximum number of qubits to process

def load_algorithm_info(csv_file):
    algorithm_info = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            algorithm_id = int(row['Algorithm ID'])
            num_qubits = int(row['Number of Qubits'])
            if num_qubits <= MAX_QUBITS:
                algorithm_info[algorithm_id] = {
                    'name': row['Algorithm Name'],
                    'num_qubits': num_qubits
                }
    return algorithm_info

def generate_ideal_distribution(qasm_file, num_qubits):
    try:
        # Read the QASM file
        circuit = QuantumCircuit.from_qasm_file(qasm_file)
        
        # Use the QASM simulator
        backend = AerSimulator()
        
        # Transpile the circuit for the backend
        transpiled_circuit = transpile(circuit, backend)
        
        # Run the simulation with 10000 shots
        job = backend.run(transpiled_circuit, shots=NUM_SHOTS)
        result = job.result()
        
        # Get the counts and convert to probabilities
        counts = result.get_counts()
        total_shots = sum(counts.values())
        probabilities = {state: count / total_shots for state, count in counts.items()}
        
        return probabilities
    except Exception as e:
        print(f"Error processing {qasm_file}: {str(e)}")
        return {}

def save_distribution_to_json(distribution, filename):
    with open(filename, 'w') as f:
        json.dump(distribution, f, indent=2)

if __name__ == "__main__":
    algorithm_info = load_algorithm_info('id_lookup.csv')
    
    # Create a directory to store individual JSON files
    os.makedirs('ideal_distributions', exist_ok=True)
    
    # Iterate through all QASM files in the current directory
    for filename in os.listdir('.'):
        if filename.endswith('.qasm'):
            match = re.match(r'algorithm_(\d+)\.qasm', filename)
            if match:
                index = int(match.group(1))
                if index in algorithm_info:
                    print(f"Processing {filename}...")
                    qasm_file = os.path.join('.', filename)
                    num_qubits = algorithm_info[index]['num_qubits']
                    if num_qubits < 21:
                        ideal_dist = generate_ideal_distribution(qasm_file, num_qubits)
                        
                        # Save individual JSON file
                        json_filename = os.path.join('ideal_distributions', f'ideal_distribution_{index}.json')
                        save_distribution_to_json(ideal_dist, json_filename)
                        
                        print(f"Saved distribution for algorithm {index} to {json_filename}")
                    else:
                        print(f"Skipping {filename} (not found in filtered id_lookup.csv or has >= 20 qubits)")
    
    print(f"Ideal distributions have been generated for algorithms with < 20 qubits using {NUM_SHOTS} shots and saved to individual JSON files in the 'ideal_distributions' directory")