"""Generate all figures for the CoLM 2026 paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# Set style
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

# Data from results
datasets = ['AesBench', 'MM-Vet', 'WIT', 'COCO', 'Mind2Web', 'Conc.Cap.',
            'TextVQA', 'LLaVA-B.', 'VisitBench', 'ChartQA', 'ScienceQA',
            'MathVista', 'DiffusionDB', 'Infograph.']

llava_width = [2.08, 2.18, 2.38, 2.43, 2.69, 2.70, 2.81, 2.92, 2.96, 3.08, 3.27, 3.37, 3.41, 3.50]
phi4_width  = [1.98, 2.39, 2.48, 2.51, 2.74, 2.70, 3.10, 3.07, 2.90, 3.52, 3.25, 3.42, 3.22, 3.58]
gemini_width= [2.14, 2.29, 2.39, 2.40, 2.92, 2.80, 2.33, 2.91, 2.85, 2.83, 2.77, 3.39, 3.38, 2.93]

# Category colors
categories = ['Aesth.', 'Gen.VQA', 'Know.', 'Gen.VQA', 'Know.', 'Know.',
              'Vision', 'Gen.VQA', 'Gen.VQA', 'Vision', 'Vision',
              'Vision', 'Aesth.', 'Vision']
cat_colors = {'Aesth.': '#e74c3c', 'Gen.VQA': '#3498db', 'Know.': '#2ecc71', 'Vision': '#9b59b6'}
colors = [cat_colors[c] for c in categories]

# ============= Figure 1: Task width comparison =============
fig, ax = plt.subplots(figsize=(5.5, 3.0))

x = np.arange(len(datasets))
width = 0.25

bars1 = ax.bar(x - width, llava_width, width, label='LLaVA-Critic-7B', color='#3498db', alpha=0.85, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, phi4_width, width, label='Phi-4-15B', color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, gemini_width, width, label='Gemini 2.5 Flash', color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=0.5)

ax.set_ylabel('R2CCP Interval Width (raw)')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=7)
ax.legend(loc='upper left', framealpha=0.9, fontsize=7)
ax.set_ylim(1.5, 4.0)
ax.axhline(y=4.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
ax.text(13.5, 3.95, 'full range', fontsize=6, ha='right', color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add category color strips at bottom
for i, c in enumerate(categories):
    ax.bar(i, 0.05, 0.8, bottom=1.5, color=cat_colors[c], alpha=0.5)

# Category legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, alpha=0.6, label=k) for k, v in cat_colors.items()]
ax2 = ax.legend(handles=legend_elements, loc='lower right', fontsize=6, title='Task category', title_fontsize=6, framealpha=0.9)
ax.add_artist(ax.legend(loc='upper left', framealpha=0.9, fontsize=7))

plt.tight_layout()
plt.savefig('figures/task_width_comparison.pdf', bbox_inches='tight', dpi=300)
plt.close()

print("Figure 1: task_width_comparison.pdf generated")

# ============= Figure 2: Ranking-scoring decoupling scatter =============
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.2), sharey=True)

# LLaVA data
llava_pearson = [0.402, 0.260, 0.164, 0.362, 0.268, 0.362, 0.389, 0.244, 0.352, 0.507, 0.258, 0.376, 0.089, 0.411]
phi4_pearson  = [0.353, 0.296, 0.400, 0.274, 0.224, 0.273, 0.213, 0.138, 0.339, 0.261, 0.303, 0.339, 0.285, 0.162]
gemini_pearson= [0.256, 0.233, 0.294, 0.342, 0.065, 0.428, 0.600, 0.312, 0.524, 0.494, 0.530, 0.414, 0.320, 0.547]

judge_data = [
    ('LLaVA-Critic-7B', llava_pearson, llava_width, '#3498db'),
    ('Phi-4-15B', phi4_pearson, phi4_width, '#e74c3c'),
    ('Gemini 2.5 Flash', gemini_pearson, gemini_width, '#2ecc71'),
]

for ax, (name, pearson, widths, color) in zip(axes, judge_data):
    for i in range(len(datasets)):
        ax.scatter(pearson[i], widths[i], c=cat_colors[categories[i]], s=30, alpha=0.8, edgecolors='black', linewidth=0.3, zorder=3)
        # Label ChartQA specifically
        if datasets[i] == 'ChartQA':
            ax.annotate('ChartQA', (pearson[i], widths[i]), fontsize=5, xytext=(5, 5),
                       textcoords='offset points', color='black')
        elif datasets[i] == 'AesBench':
            ax.annotate('AesBench', (pearson[i], widths[i]), fontsize=5, xytext=(5, -8),
                       textcoords='offset points', color='black')
    ax.set_xlabel('Pearson $\\rho$')
    ax.set_title(name, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.05, 0.7)
    ax.set_ylim(1.5, 4.0)

axes[0].set_ylabel('R2CCP Width')

plt.tight_layout()
plt.savefig('figures/ranking_scoring_decoupling.pdf', bbox_inches='tight', dpi=300)
plt.close()

print("Figure 2: ranking_scoring_decoupling.pdf generated")

# ============= Figure 3: MLLM-Judge vs Polaris comparison =============
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

# Left: bar chart comparing key metrics
metrics = ['Pearson', 'Accuracy', '±1 Acc.', 'MAE']
mllm_vals = [0.402, 0.322, 0.751, 1.031]
polaris_vals = [0.906, 0.809, 0.954, 0.243]

x = np.arange(len(metrics))
w = 0.35
axes[0].bar(x - w/2, mllm_vals, w, label='MLLM-Judge', color='#e74c3c', alpha=0.85)
axes[0].bar(x + w/2, polaris_vals, w, label='Polaris', color='#3498db', alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics, fontsize=8)
axes[0].set_ylabel('Value')
axes[0].legend(fontsize=7)
axes[0].set_title('Point prediction metrics', fontsize=9)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Right: interval width comparison
labels = ['MLLM-Judge\n(VQA, 14 types)', 'Polaris\n(Captioning)']
widths_vals = [3.049, 0.678]
coverages = [0.900, 0.899]
bar_colors = ['#e74c3c', '#3498db']

bars = axes[1].bar(labels, widths_vals, color=bar_colors, alpha=0.85, width=0.5)
axes[1].set_ylabel('R2CCP Interval Width')
axes[1].set_title('Same judge, same method', fontsize=9)

# Add coverage annotation
for bar, cov, wid in zip(bars, coverages, widths_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, wid + 0.08,
                f'cov={cov:.1%}\nwidth={wid:.2f}', ha='center', fontsize=7)

# Arrow showing 4.5x
axes[1].annotate('', xy=(1, 0.678), xytext=(0, 3.049),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
axes[1].text(0.5, 1.8, '4.5× narrower', ha='center', fontsize=8, fontweight='bold', rotation=-60)

axes[1].set_ylim(0, 3.8)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/polaris_comparison.pdf', bbox_inches='tight', dpi=300)
plt.close()

print("Figure 3: polaris_comparison.pdf generated")

# ============= Figure 4: Error bin CP coverage =============
fig, ax = plt.subplots(figsize=(4.0, 2.5))

errors = ['Exact\n(32.6%)', '±1\n(43.0%)', '±2\n(16.0%)', '±3\n(6.3%)', '±4\n(2.0%)']
cov_raw = [99.8, 98.7, 84.3, 25.4, 5.7]
cov_adj = [100.0, 99.9, 99.4, 91.5, 43.3]

x = np.arange(len(errors))
w = 0.35

ax.bar(x - w/2, cov_raw, w, label='Raw', color='#3498db', alpha=0.85)
ax.bar(x + w/2, cov_adj, w, label='Boundary adj.', color='#2ecc71', alpha=0.85)
ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, linewidth=1, label='90% target')
ax.set_ylabel('CP Coverage (%)')
ax.set_xlabel('Judge error magnitude')
ax.set_xticks(x)
ax.set_xticklabels(errors, fontsize=7)
ax.legend(fontsize=7, loc='lower left')
ax.set_ylim(0, 105)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('CP coverage by error magnitude (R2CCP, LLaVA-Critic)', fontsize=8)

plt.tight_layout()
plt.savefig('figures/error_bin_coverage.pdf', bbox_inches='tight', dpi=300)
plt.close()

print("Figure 4: error_bin_coverage.pdf generated")

print("\nAll figures generated successfully!")
