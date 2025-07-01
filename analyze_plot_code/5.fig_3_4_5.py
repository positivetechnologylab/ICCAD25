import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as mp

params = {
	'font.family': 'serif',
	'text.usetex': True,
	'pgf.rcfonts': False,
	'pgf.texsystem': 'xelatex',
	'pgf.preamble': r'\usepackage{fontspec,physics}',
}

mpl.rcParams.update(params)

COMPUTERS = ['Brisbane', 'Cusco', 'Kyiv', 'Kyoto', 'Nazca']

NUM_QUBITS = 127

X_TICKS = [0, 21, 42, 63, 84, 105, 126]

COLORS = {}
COLORS[COMPUTERS[0]] = '#FFC639'
COLORS[COMPUTERS[1]] = '#FF659A'
COLORS[COMPUTERS[2]] = '#8F05FA'
COLORS[COMPUTERS[3]] = '#0000F4'
COLORS[COMPUTERS[4]] = '#000032'

LS = {}
LS[COMPUTERS[0]] = '-'
LS[COMPUTERS[1]] = '-.'
LS[COMPUTERS[2]] = '--'
LS[COMPUTERS[3]] = '-'
LS[COMPUTERS[4]] = '-.'

import json

data = {}
with open('../data/qubit_usage.json', 'r') as f:
    qubit_usage = json.load(f)

for computer_name in COMPUTERS:
    data[computer_name] = []
    for qubit_id in range(NUM_QUBITS):
        if computer_name in qubit_usage and str(qubit_id) in qubit_usage[computer_name]:
            data[computer_name].append(qubit_usage[computer_name][str(qubit_id)])
        else:
            data[computer_name].append(0)
    data[computer_name] = sorted(data[computer_name])

# Generate qubit usage plots
fig = mp.figure(figsize=(7.1, 2.0))

ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.grid(linestyle=':', color='grey', linewidth=0.5)


for c in range(len(COMPUTERS)):

	ax.plot(list(range(NUM_QUBITS)),
		   data[COMPUTERS[c]],
		   color=COLORS[COMPUTERS[c]],
		   label=COMPUTERS[c],
		   linewidth=1.5,
		   ls=LS[COMPUTERS[c]])

ax.set_xlim(0, NUM_QUBITS-1)
ax.set_ylim(0, 1.0)

ax.set_xticks(X_TICKS)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)

ax.set_xlabel('Qubit (Sorted According to Usage for Each Computer)', fontsize=14)
ax.set_ylabel('Normalized Qubit Usage', fontsize=14)

ax.legend(ncol=5, edgecolor='black', bbox_to_anchor=(0.0, 1.02, 1., .102), mode='expand',
          loc='lower left', borderaxespad=0.02, borderpad=0.3,
          fontsize=14, handletextpad=0.5)

mp.savefig('../plots/qubit_use_frequency.pdf', format='pdf', bbox_inches='tight',  pad_inches=0.01)
mp.close()


# Load and prepare data
average_vals = np.load('../data/average_vals.npy')
data0, data1 = zip(*average_vals)
data0 = np.array(data0)
data1 = np.array(data1)

# Define bins and group data
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
indices = np.digitize(data0, bins)
groups = []
for i in range(1, len(bins)):
    group = data1[indices == i]
    groups.append(group)

# Generate the violin plot
fig = mp.figure(figsize=(2.0, 2.0))
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.grid(linestyle=':', color='grey', linewidth=0.5)
ax.violinplot(groups, positions=range(len(groups)), showmeans=False, showmedians=True)

# Set x-ticks and labels
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])

# Adjust axis labels and limits
ax.set_xlim(-0.5, len(groups) - 0.5)
ax.set_ylim(0, 1.0)
mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)
ax.set_ylabel('Mean KLD', fontsize=14)
ax.set_xlabel('Qubit Usage', fontsize=14)

# Save and close the figure
mp.savefig('../plots/qubit_usage_vs_mean_KLD_violin.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
mp.close()


# Load and prepare data
average_vals = np.load('../data/std_vals.npy')
data0, data1 = zip(*average_vals)
data0 = np.array(data0)
data1 = np.array(data1)

# Define bins and group data
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
indices = np.digitize(data0, bins)
groups = []
for i in range(1, len(bins)):
    group = data1[indices == i]
    groups.append(group)

# Generate the violin plot
fig = mp.figure(figsize=(2.0, 2.0))
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.grid(linestyle=':', color='grey', linewidth=0.5)
ax.violinplot(groups, positions=range(len(groups)), showmeans=False, showmedians=True)

# Set x-ticks and labels
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])

# Adjust axis labels and limits
ax.set_xlim(-0.5, len(groups) - 0.5)
ax.set_ylim(0, 1.0)
mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)
ax.set_ylabel('Std. Dev.of KLD', fontsize=14)
ax.set_xlabel('Qubit Usage', fontsize=14)

# Save and close the figure
mp.savefig('../plots/qubit_usage_vs_std_KLD_violin.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
mp.close()


average_vals = np.load('../data/min_vals.npy')
data0, data1 = zip(*average_vals)
data0 = np.array(data0)
data1 = np.array(data1)

# Define bins and group data
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
indices = np.digitize(data0, bins)
groups = []
for i in range(1, len(bins)):
    group = data1[indices == i]
    groups.append(group)

# Generate the violin plot
fig = mp.figure(figsize=(2.0, 2.0))
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.grid(linestyle=':', color='grey', linewidth=0.5)
ax.violinplot(groups, positions=range(len(groups)), showmeans=False, showmedians=True)

# Set x-ticks and labels
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])

# Adjust axis labels and limits
ax.set_xlim(-0.5, len(groups) - 0.5)
ax.set_ylim(0, 1.0)
mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)
ax.set_ylabel('Minimum KLD', fontsize=14)
ax.set_xlabel('Qubit Usage', fontsize=14)

# Save and close the figure
mp.savefig('../plots/qubit_usage_vs_min_KLD_violin.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
mp.close()

average_vals = np.load('../data/max_vals.npy')
data0, data1 = zip(*average_vals)
data0 = np.array(data0)
data1 = np.array(data1)

# Define bins and group data
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
indices = np.digitize(data0, bins)
groups = []
for i in range(1, len(bins)):
    group = data1[indices == i]
    groups.append(group)

# Generate the violin plot
fig = mp.figure(figsize=(2.0, 2.0))
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.grid(linestyle=':', color='grey', linewidth=0.5)
ax.violinplot(groups, positions=range(len(groups)), showmeans=False, showmedians=True)

# Set x-ticks and labels
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])

# Adjust axis labels and limits
ax.set_xlim(-0.5, len(groups) - 0.5)
ax.set_ylim(0, 1.0)
mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)
ax.set_ylabel('Maximum KLD', fontsize=14)
ax.set_xlabel('Qubit Usage', fontsize=14)

# Save and close the figure
mp.savefig('../plots/qubit_usage_vs_max_KLD_violin.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
mp.close()

average_vals = np.load('../data/spread_vals.npy')
data0, data1 = zip(*average_vals)
data0 = np.array(data0)
data1 = np.array(data1)

# Define bins and group data
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
indices = np.digitize(data0, bins)
groups = []
for i in range(1, len(bins)):
    group = data1[indices == i]
    groups.append(group)

# Generate the violin plot
fig = mp.figure(figsize=(2.0, 2.0))
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.grid(linestyle=':', color='grey', linewidth=0.5)
ax.violinplot(groups, positions=range(len(groups)), showmeans=False, showmedians=True)

# Set x-ticks and labels
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])

# Adjust axis labels and limits
ax.set_xlim(-0.5, len(groups) - 0.5)
ax.set_ylim(0, 1.0)
mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)
ax.set_ylabel('Spread of KLD', fontsize=14)
ax.set_xlabel('Qubit Usage', fontsize=14)

# Save and close the figure
mp.savefig('../plots/qubit_usage_vs_spread_KLD_violin.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
mp.close()

X_TICK_LABELS = ['VQE',
				'HHL',
				'QPE1',
				'ADD1',
				'SAT',
				'SECA',
				'GCM',
				'MULT',
				'QPE2',
				'DNN',
				'QEC',
				'ADD2',
				'SQRT',
				'QRAM',
				'KNN1',
				'SWAP']
NUM_X_TICKS = len(X_TICK_LABELS)


with open('../data/kld_differences.json', 'r') as file:
    kld_differences = json.load(file)

data = list(kld_differences.values())

# Generate Number of Gates plots
fig = mp.figure(figsize=(7.0, 2.0))

ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)

ax.bar([i for i in range(NUM_X_TICKS)],
	   data,
	   color='aquamarine',
	   linewidth=1.0,
	   edgecolor='black',
	   hatch='////',
	   width=0.8)

ax.set_xlim(-0.6, NUM_X_TICKS-0.4)
ax.set_ylim(-10, 10)

ax.set_xticks(range(NUM_X_TICKS))
ax.set_xticklabels(X_TICK_LABELS)

ax.set_yticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])

# Draw a black horizontal line at 0
ax.axhline(y=0, color='black', linewidth=1.0)

mp.setp(ax.get_xticklabels(), fontsize=14, rotation=90)
mp.setp(ax.get_yticklabels(), fontsize=14)

ax.set_ylabel('KLD Degradation\nover Group-O3', fontsize=14)

mp.savefig('../plots/KLD_diff_O3_IM.pdf', format='pdf', bbox_inches='tight',  pad_inches=0.01)
mp.close()
