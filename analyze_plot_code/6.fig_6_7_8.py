import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
import json
import glob
import os
from datetime import datetime
import numpy as np
from collections import defaultdict
import argparse
from dateutil import parser

params = {
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'xelatex',
    'pgf.preamble': r'\usepackage{fontspec,physics}',
}

plt.rcParams.update(params)
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')

EXCLUDED_INDICES = {15, 16, 18, 19}
VALID_INDICES = sorted([i for i in range(20) if i not in EXCLUDED_INDICES])
X_TICK_LABELS = ['VQE', 'HHL', 'QPE1', 'ADD1', 'SAT', 'SECA', 'GCM', 'MULT', 
                 'QPE2', 'DNN', 'QEC', 'ADD2', 'SQRT', 'QRAM', 'KNN1', 'SWAP']

TIMING_AVERAGES = {
    "Algorithm 0":  {"opt1": 2597.68, "opt2": 2546.30, "opt3": 2987.63},
    "Algorithm 1":  {"opt1": 2125.75, "opt2": 1904.13, "opt3": 1703.46},
    "Algorithm 2":  {"opt1": 1887.17, "opt2": 2200.93, "opt3": 1747.78},
    "Algorithm 3":  {"opt1": 1547.01, "opt2": 1725.55, "opt3": 1827.50},
    "Algorithm 4":  {"opt1": 1845.36, "opt2": 1843.39, "opt3": 3100.98},
    "Algorithm 5":  {"opt1": 1579.78, "opt2": 1723.95, "opt3": 1803.39},
    "Algorithm 6":  {"opt1": 2308.96, "opt2": 4808.03, "opt3": 5816.24},
    "Algorithm 7":  {"opt1": 1643.21, "opt2": 1580.84, "opt3": 1613.78},
    "Algorithm 8":  {"opt1": 1682.37, "opt2": 1674.60, "opt3": 1687.69},
    "Algorithm 9":  {"opt1": 1774.30, "opt2": 1996.57, "opt3": 1812.20},
    "Algorithm 10": {"opt1": 1548.71, "opt2": 2120.83, "opt3": 1610.63},
    "Algorithm 11": {"opt1": 1830.34, "opt2": 1657.21, "opt3": 1926.09},
    "Algorithm 12": {"opt1": 3985.92, "opt2": 4637.44, "opt3": 6832.62},
    "Algorithm 13": {"opt1": 1693.81, "opt2": 1969.93, "opt3": 1975.27},
    "Algorithm 14": {"opt1": 1547.96, "opt2": 1662.73, "opt3": 1800.24},
    "Algorithm 17": {"opt1": 1734.57, "opt2": 1792.83, "opt3": 1656.81}
}

def process_data(start_date, end_date, base_path="."):
    """Process files and collect KLD data."""
    ideal_distributions = {}
    for idx in VALID_INDICES:
        try:
            with open(f'../input_algorithms/ideal_distributions/ideal_distribution_{idx}.json') as f:
                ideal_distributions[idx] = json.load(f)
        except Exception as e:
            print(f"Error loading ideal distribution {idx}: {e}")
            continue
    
    files = []
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    for date in range(int(start_date), int(end_date) + 1):
        files.extend(glob.glob(os.path.join(f'../data/{base_path}', str(date), 'group_B_*.json')))
    
    kld_results = {i: {level: [] for level in range(1, 4)} for i in VALID_INDICES}
    
    for file in files:
        try:
            with open(file) as f:
                data = json.load(f)
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, list):
                data = [data]
                
            for entry in data:
                if isinstance(entry, str):
                    continue
                    
                opt_level = entry.get('optimization_level')
                idx = entry.get('algorithm_index')
                
                if opt_level in [1, 2, 3] and idx in VALID_INDICES:
                    actual_dist = entry.get('counts', {})
                    ideal_dist = ideal_distributions.get(idx)
                    if actual_dist and ideal_dist:
                        kld = calculate_kld(ideal_dist, actual_dist)
                        kld_results[idx][opt_level].append(kld)
                        
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    return kld_results

def create_comparison_plots(kld_results):
    """Create vertical figures comparing differences between optimization levels."""
    plot_dir = "../plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create figure with adjusted height and tighter vertical spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4), dpi=500)
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.1)  # Reduced from default
    
    # Colors and labels
    colors = ['#FFA07A', '#90EE90']
    labels = ['Optimization Level 1', 'Optimization Level 2']
    hatches = ['///', '\\\\\\']
    
    # Bar settings
    bar_width = 0.35
    group_spacing = 0.85
    x = np.arange(len(VALID_INDICES)) * group_spacing
    
    # KLD Plot
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)
    
    # Calculate KLD means
    kld_means = {opt_level: [] for opt_level in [1, 2, 3]}
    for idx in VALID_INDICES:
        for opt_level in [1, 2, 3]:
            mean = np.mean(kld_results[idx][opt_level] or [0])
            kld_means[opt_level].append(mean)
    
    # Plot differences
    kld_comparisons = [(1, 3), (2, 3)]  # For KLD (unchanged)
    time_comparisons = [(3, 1), (3, 2)]  # For time (changed order)
    
    # KLD differences (subplot 1)
    for i, (opt_level1, opt_level2) in enumerate(kld_comparisons):
        differences = [mean1 - mean2 for mean1, mean2 in 
                      zip(kld_means[opt_level1], kld_means[opt_level2])]
        bar_position = x + i*bar_width
        
        ax1.bar(bar_position, differences, bar_width,
               color=colors[i], linewidth=1.0, edgecolor='black',
               hatch=hatches[i], label=labels[i])
    
    # Time differences (subplot 2)
    for i, (opt_level1, opt_level2) in enumerate(time_comparisons):
        differences = [
            (TIMING_AVERAGES[f"Algorithm {idx}"][f"opt{opt_level1}"] - 
             TIMING_AVERAGES[f"Algorithm {idx}"][f"opt{opt_level2}"]) / 1000
            for idx in VALID_INDICES
        ]
        bar_position = x + i*bar_width
        ax2.bar(bar_position, differences, bar_width,
               color=colors[i], linewidth=1.0, edgecolor='black',
               hatch=hatches[i])
    
    # KLD Plot settings
    ax1.set_ylim(-10, 10)
    ax1.set_yticks(np.arange(-10, 11, 2))
    ax1.set_xlim(-0.6, x[-1] + 2*bar_width + 0.3)
    ax1.set_xticks(x + bar_width)
    ax1.set_xticklabels([])
    plt.setp(ax1.get_yticklabels(), fontsize=12)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Modified legend with 2 columns and adjusted position
    legend = ax1.legend(fontsize=11, ncol=2, loc='upper left', bbox_to_anchor=(0.01, 1.0),
                       frameon=True, edgecolor='black')
    legend.get_frame().set_linewidth(0.8)  # Set the legend border width
    
    # Time Plot settings
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)
    ax2.set_xlim(-0.6, x[-1] + 2*bar_width + 0.3)
    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels(X_TICK_LABELS, fontsize=10, rotation=90)
    plt.setp(ax2.get_yticklabels(), fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Adjusted y-axis label positions for tighter layout
    fig.text(0, 0.7, 'KLD Degradation', va='center', rotation='vertical', fontsize=13)
    fig.text(0.03, 0.7, 'over Optimization', va='center', rotation='vertical', fontsize=13)
    fig.text(0.06, 0.695, 'Level 3', va='center', rotation='vertical', fontsize=13)
    fig.text(0, 0.29, 'Time Improvement', va='center', rotation='vertical', fontsize=13)
    fig.text(0.03, 0.29, 'over Optimization', va='center', rotation='vertical', fontsize=13)
    fig.text(0.06, 0.295, 'Level 3 (s)', va='center', rotation='vertical', fontsize=13)
    
    # Save plots with tight layout
    plt.savefig(os.path.join(plot_dir, "RQ4.pdf"), 
                format='pdf', bbox_inches='tight', pad_inches=0.01, dpi=500)
    plt.savefig(os.path.join(plot_dir, "RQ4.png"), 
                format='png', bbox_inches='tight', pad_inches=0.01, dpi=500)
    
    plt.close()

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
    
    # Initialize data structures for per-(algorithm,computer) collection
    algo_computer_data = defaultdict(lambda: defaultdict(list))
    
    # Get files for date range
    files = []
    for date in range(int(start_date), int(end_date) + 1):
        files.extend(glob.glob(os.path.join(f'../data/{base_path}', str(date), 'group_C_*.json')))
    
    # First pass: collect KLD values per (algorithm,computer) pair
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
                            time_bin = get_time_bin(hour)
                            
                            # Store KLD value with algorithm, computer and time bin
                            key = (idx, computer)
                            algo_computer_data[key]['klds'].append(kld)
                            algo_computer_data[key]['bins'].append(time_bin)
                            
                        except Exception:
                            continue
                            
        except Exception:
            continue
    
    # Second pass: normalize per (algorithm,computer) pair and organize by time bins
    normalized_bins = defaultdict(list)
    
    # Normalize each (algorithm,computer) pair separately
    for (algo, computer), data in algo_computer_data.items():
        klds = data['klds']
        bins = data['bins']
        
        # Normalize KLDs for this (algorithm,computer) pair
        normalized_klds = min_max_normalize(klds)
        
        # Add normalized values to their respective time bins
        for norm_kld, bin_num in zip(normalized_klds, bins):
            normalized_bins[bin_num].append(norm_kld)
    
    return normalized_bins

def create_boxplot(bin_data):
    """Create box plot with fixed y-axis scale."""
    plt.figure(figsize=(7, 2.0), dpi=500)
    
    # Prepare data for box plot
    plot_data = [bin_data.get(i, []) for i in range(8)]
    
    # Create the base box plot
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)
    
    # Create box plot
    bp = plt.boxplot(plot_data,
                    patch_artist=True,
                    widths=0.4,
                    whis=1.5,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=1.0),
                    boxprops=dict(linewidth=1.0, edgecolor='black'))
    
    for patch in bp['boxes']:
        patch.set_facecolor('#2a9d8f')
        patch.set_alpha(0.7)
    
    time_labels = [f"{i*3:02d}-{(i+1)*3:02d}" for i in range(8)]
    plt.xticks(range(1, 9), time_labels, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Set y-axis limits from 0 to 1
    plt.ylim(-0.02, 1.02)
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.xlabel('Time of Day (Hours)', fontsize=14)
    plt.ylabel('Normalized KLD', fontsize=14)
    
    plot_dir = "../plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.savefig(os.path.join(plot_dir, "RQ5.pdf"), 
                format='pdf', 
                bbox_inches='tight', 
                pad_inches=0.01, 
                dpi=500)
    
    plt.savefig(os.path.join(plot_dir, "RQ5.png"), 
                format='png', 
                bbox_inches='tight', 
                pad_inches=0.01, 
                dpi=500)
    
    plt.close()

def filter_hourly_data_points(data_points):
    sorted_points = sorted(data_points)
    last_hour = defaultdict(float)
    filtered_points = []
    
    for hours, kld, computer in sorted_points:
        current_hour = int(hours)
        if current_hour > last_hour[computer]:
            filtered_points.append((hours, kld, computer))
            last_hour[computer] = current_hour
    
    return filtered_points

def process_continuous_time_data(start_date, end_date, base_path="."):
    EXCLUDED_INDICES = {15, 16, 18, 19}
    VALID_INDICES = sorted([i for i in range(20) if i not in EXCLUDED_INDICES])
    
    ideal_distributions = {}
    for idx in VALID_INDICES:
        try:
            with open(f'../input_algorithms/ideal_distributions/ideal_distribution_{idx}.json') as f:
                ideal_distributions[idx] = json.load(f)
        except Exception:
            continue
    
    all_data_points = []
    files = []
    for date in range(int(start_date), int(end_date) + 1):
        date_files = glob.glob(os.path.join(f'../data/{base_path}', str(date), 'group_D_*.json'))
        files.extend(date_files)
    
    start_timestamp = datetime.strptime(start_date, "%Y%m%d")
    
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
                            dt = parser.parse(completion_time)
                            if dt.tzinfo:
                                dt = dt.astimezone().replace(tzinfo=None)
                            
                            hours_since_start = (dt - start_timestamp).total_seconds() / 3600.0
                            kld = calculate_kld(ideal_dist, counts)
                            
                            all_data_points.append((hours_since_start, kld, computer))
                            
                        except Exception:
                            continue
                        
        except Exception:
            continue
    
    return filter_hourly_data_points(all_data_points)

def create_combined_plot(data_points):
    """Create horizontal subplots for each computer."""
    if not data_points:
        return
        
    plot_dir = "../plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Style configurations with your color mapping
    style_map = {
        'ibm_brisbane': ('#FFC639', 'x'),  # Brisbane color
        'ibm_nazca': ('#000032', 's'),     # Nazca color
        'ibm_sherbrooke': ('#00B300', '^') # Green color for Sherbrooke
    }
    
    # Create figure with three horizontal subplots - adjusted figure width
    fig, axes = plt.subplots(1, 3, figsize=(7, 2), dpi=500)
    
    # Calculate global y-axis limits
    all_klds = [k for _, k, _ in data_points]
    y_min = min(all_klds)
    y_max = max(all_klds)
    y_margin = (y_max - y_min) * 0.05  # 5% margin
    y_limits = [y_min - y_margin, y_max + y_margin]
    
    # Sort computers to ensure consistent order
    unique_computers = sorted(set(computer for _, _, computer in data_points))
    
    for idx, computer in enumerate(unique_computers):
        ax = axes[idx]
        
        computer_points = [(h, k) for h, k, comp in data_points if comp == computer]
        computer_points.sort(key=lambda x: x[0])
        
        if computer_points:
            hours, klds = zip(*computer_points)
            
            color, marker = style_map[computer]
            
            # Plot line
            line = ax.plot(hours, 
                   klds,
                   color=color,
                   alpha=0.5,
                   linewidth=1,
                   zorder=1)[0]
            
            if marker == 'x':
                ax.scatter(hours,
                          klds,
                          marker=marker,
                          s=10,
                          color='black',
                          alpha=0.8,
                          zorder=2)
            else:
                ax.scatter(hours,
                          klds,
                          marker=marker,
                          s=10,
                          color=color,
                          edgecolor='black',
                          linewidth=0.3,
                          alpha=0.8,
                          zorder=2)
            
            computer_name = computer.replace('ibm_', '').capitalize()
            ax.legend([line], 
                     [computer_name],
                     loc='upper left',
                     frameon=True,
                     framealpha=0.8,
                     fontsize=11,
                     handletextpad=0.5,
                     borderaxespad=0.4,
                     borderpad=0.3,
                     edgecolor='black')
        
        if idx == 0:
            ax.set_ylabel('KLD', fontsize=16)
        
        if idx > 0:
            ax.set_yticklabels([])
        
        ax.tick_params(axis='x', labelsize=9.5)
        ax.tick_params(axis='y', labelsize=11)
        
        ax.grid(which='major',
                axis='both',
                linestyle='--',
                linewidth=0.5,
                color='gray',
                alpha=0.7)
        
        ax.set_xlim(-1, 72)
        ax.set_ylim(-1, 1)
        ax.set_xticks(range(0, 73, 12))
    
    fig.text(0.5, -0.03, 'Time (3 Day Period)', ha='center', fontsize=16)
    
    plt.subplots_adjust(wspace=0.075, bottom=0.2)
    
    # Save plot
    plt.savefig(os.path.join(plot_dir, "RQ6.pdf"),
                format='pdf',
                bbox_inches='tight',
                pad_inches=0.01)
    
    plt.savefig(os.path.join(plot_dir, "RQ6.png"),
                format='png',
                bbox_inches='tight',
                pad_inches=0.01)
    
    plt.close()

def main():
    data_points = process_continuous_time_data("20240914", "20240916")
    create_combined_plot(data_points)
    bin_data = process_time_fidelity_data("20240810", "20240822")
    create_boxplot(bin_data)
    kld_results = process_data("20240810", "20240822")
    create_comparison_plots(kld_results)

if __name__ == "__main__":
    main()
    
