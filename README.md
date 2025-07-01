# Revisiting Noise-adaptive Transpilation in Quantum Computing: How Much Impact Does it Have?

This repository contains the artifacts and experimental data for the paper "Revisiting Noise-adaptive Transpilation in Quantum Computing: How Much Impact Does it Have?". The paper will appear at the Proceedings of the International Conference on Computer-Aided Design (ICCAD), 2025.

## Overview

This work systematically evaluates the impact of noise-adaptive transpilation techniques in quantum computing by conducting extensive experiments on real IBM quantum hardware. We analyze the effectiveness of different transpilation strategies and their influence on quantum circuit execution fidelity under realistic noise conditions.

## Repository Structure

```
transpilation_final/
├── analyze_plot_code/          # Analysis scripts for generating figures and tables
│   ├── 1.analyze.py            
│   ├── 2.fig_3_preprocess_a.py 
│   ├── 2.fig_3_preprocess_b.py 
│   ├── 3.fig_4_preprocess_a.py 
│   ├── 3.fig_4_preprocess_b.py
│   ├── 3.fig_4_preprocess_c.py
│   ├── 3.fig_4_prepocess_d.py 
│   ├── 4.tables.py            
│   ├── 5.fig_3_4_5.py         
│   └── 6.fig_6_7_8.py          
├── data/                       # Experimental data from IBM quantum computers
│   ├── 20240810/ - 20240822/ 
│   ├── 20240913/ - 20240916/
│   └── *.json, *.csv, *.npy   # Processed analysis results
├── input_algorithms/           # Quantum algorithms used in experiments
│   ├── algorithm_*.qasm     
│   ├── ideal_distributions/   # Ideal output distributions
│   └── cal_distribution.py
├── plots/                      # Generated figures and plots
└── README.md   
```

## Usage

### Prerequisites
Ensure you have Python 3.11.9+ installed with the required packages:

```bash
pip install -r requirements.txt
```

Then run the scrips in analyze_plot_code to generate plots and tables.

# Code Licensing

© 2025 Rice University subject to Creative Commons Attribution 4.0 International license (Creative Commons — Attribution 4.0 International — CC BY 4.0)

Contact ptl@rice.edu for permissions.
