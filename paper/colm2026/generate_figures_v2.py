"""Generate clean, publication-quality figures for CoLM 2026 paper.
Style: grayscale-friendly, minimal, serif fonts, no chartjunk.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

OUT = 'figures_v2'
os.makedirs(OUT, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Data
datasets = ['AesBench', 'MM-Vet', 'WIT', 'COCO', 'Mind2Web', 'Conc.Cap.',
            'TextVQA', 'LLaVA-B.', 'VisitBench', 'ChartQA', 'ScienceQA',
            'MathVista', 'DiffusionDB', 'Infograph.']

llava_width = [2.08, 2.18, 2.38, 2.43, 2.69, 2.70, 2.81, 2.92, 2.96, 3.08, 3.27, 3.37, 3.41, 3.50]
phi4_width  = [1.98, 2.39, 2.48, 2.51, 2.74, 2.70, 3.10, 3.07, 2.90, 3.52, 3.25, 3.42, 3.22, 3.58]
gemini_width= [2.14, 2.29, 2.39, 2.40, 2.92, 2.80, 2.33, 2.91, 2.85, 2.83, 2.77, 3.39, 3.38, 2.93]

categories = ['Aesth.', 'Gen.', 'Know.', 'Gen.', 'Know.', 'Know.',
              'Vision', 'Gen.', 'Gen.', 'Vision', 'Vision',
              'Vision', 'Aesth.', 'Vision']

llava_pearson = [0.402, 0.260, 0.164, 0.362, 0.268, 0.362, 0.389, 0.244, 0.352, 0.507, 0.258, 0.376, 0.089, 0.411]
phi4_pearson  = [0.353, 0.296, 0.400, 0.274, 0.224, 0.273, 0.213, 0.138, 0.339, 0.261, 0.303, 0.339, 0.285, 0.162]
gemini_pearson= [0.256, 0.233, 0.294, 0.342, 0.065, 0.428, 0.600, 0.312, 0.524, 0.494, 0.530, 0.414, 0.320, 0.547]

# Grayscale-friendly colors (distinguishable in B&W)
C_LLAVA = '#404040'   # dark gray
C_PHI4 = '#999999'    # medium gray
C_GEMINI = '#CCCCCC'  # light gray
C_ACCENT = '#000000'  # black for emphasis

# Category markers for scatter plots
cat_markers = {'Aesth.': 'o', 'Gen.': 's', 'Know.': '^', 'Vision': 'D'}
cat_fills = {'Aesth.': 'white', 'Gen.': '#999999', 'Know.': '#404040', 'Vision': 'black'}

# ============================================================================
# Figure 1: Task-dependent width (horizontal bar chart — cleaner than grouped bars)
# ============================================================================
fig, ax = plt.subplots(figsize=(3.3, 3.5))

# Sort by LLaVA width (already sorted)
y = np.arange(len(datasets))

ax.barh(y + 0.2, llava_width, 0.2, label='LLaVA-Critic-7B', color=C_LLAVA, edgecolor='white', linewidth=0.3)
ax.barh(y, phi4_width, 0.2, label='Phi-4-15B', color=C_PHI4, edgecolor='white', linewidth=0.3)
ax.barh(y - 0.2, gemini_width, 0.2, label='Gemini 2.5 Flash', color=C_GEMINI, edgecolor='black', linewidth=0.3)

ax.set_yticks(y)
ax.set_yticklabels(datasets, fontsize=6.5)
ax.set_xlabel('R2CCP Interval Width')
ax.set_xlim(1.5, 4.0)
ax.axvline(x=4.0, color='gray', linestyle=':', alpha=0.4, linewidth=0.5)
ax.text(3.92, -0.8, 'full\nrange', fontsize=5.5, ha='right', color='gray')
ax.invert_yaxis()
ax.legend(loc='lower right', fontsize=6, frameon=True, fancybox=False, edgecolor='gray', framealpha=0.9)

# Add category labels on right
for i, cat in enumerate(categories):
    ax.text(4.05, i, cat, fontsize=5, va='center', color='gray')

plt.savefig(f'{OUT}/fig1_task_width.pdf', dpi=300)
plt.close()
print("Fig 1: task_width done")

# ============================================================================
# Figure 2: Ranking-scoring decoupling (scatter, 3 panels)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0), sharey=True)

judge_data = [
    ('LLaVA-Critic-7B', llava_pearson, llava_width),
    ('Phi-4-15B', phi4_pearson, phi4_width),
    ('Gemini 2.5 Flash', gemini_pearson, gemini_width),
]

for ax, (name, pearson, widths) in zip(axes, judge_data):
    for i in range(len(datasets)):
        marker = cat_markers[categories[i]]
        fill = cat_fills[categories[i]]
        ax.scatter(pearson[i], widths[i], marker=marker, c=fill,
                   s=25, edgecolors='black', linewidth=0.4, zorder=3)

    # Highlight ChartQA and AesBench
    for i, ds in enumerate(datasets):
        if ds == 'ChartQA':
            ax.annotate(ds, (pearson[i], widths[i]), fontsize=5,
                        xytext=(3, 4), textcoords='offset points', style='italic')
        elif ds == 'AesBench':
            ax.annotate(ds, (pearson[i], widths[i]), fontsize=5,
                        xytext=(3, -6), textcoords='offset points', style='italic')
        elif ds == 'Infograph.' and name == 'LLaVA-Critic-7B':
            ax.annotate(ds, (pearson[i], widths[i]), fontsize=5,
                        xytext=(-35, 4), textcoords='offset points', style='italic')

    ax.set_xlabel('Pearson $\\rho$')
    ax.set_title(name, fontsize=7.5)
    ax.set_xlim(-0.05, 0.7)
    ax.set_ylim(1.5, 4.0)
    ax.grid(True, alpha=0.15, linewidth=0.3)

axes[0].set_ylabel('R2CCP Width')

# Category legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker=cat_markers[k], color='w', markerfacecolor=cat_fills[k],
           markeredgecolor='black', markersize=5, label=k, linewidth=0)
    for k in ['Aesth.', 'Gen.', 'Know.', 'Vision']
]
axes[2].legend(handles=legend_elements, loc='upper right', fontsize=5.5,
               frameon=True, fancybox=False, edgecolor='gray', handletextpad=0.3)

plt.tight_layout(w_pad=0.3)
plt.savefig(f'{OUT}/fig2_ranking_scoring.pdf', dpi=300)
plt.close()
print("Fig 2: ranking_scoring done")

# ============================================================================
# Figure 3: MLLM-Judge vs Polaris (clean paired comparison)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 2.0), gridspec_kw={'width_ratios': [1.2, 1]})

# Left: Key metrics side-by-side
metrics = ['Pearson $\\rho$', 'Accuracy', '$\\pm$1 Acc.']
mllm_vals = [0.402, 0.322, 0.751]
polaris_vals = [0.906, 0.809, 0.954]

x = np.arange(len(metrics))
w = 0.3
bars1 = ax1.bar(x - w/2, mllm_vals, w, label='MLLM-Judge', color=C_LLAVA, edgecolor='black', linewidth=0.3)
bars2 = ax1.bar(x + w/2, polaris_vals, w, label='Polaris', color=C_GEMINI, edgecolor='black', linewidth=0.3)

ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=7)
ax1.set_ylabel('Score')
ax1.set_ylim(0, 1.1)
ax1.legend(fontsize=6, loc='upper left', frameon=True, fancybox=False, edgecolor='gray')

# Add value labels
for bar, val in zip(bars1, mllm_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', fontsize=5.5)
for bar, val in zip(bars2, polaris_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', fontsize=5.5)

# Right: Width comparison (the 4.5x result)
labels = ['MLLM-Judge', 'Polaris']
widths_vals = [3.049, 0.678]

bars = ax2.bar(labels, widths_vals, color=[C_LLAVA, C_GEMINI], edgecolor='black', linewidth=0.3, width=0.5)
ax2.set_ylabel('R2CCP Width')
ax2.set_ylim(0, 3.8)

# Value + coverage labels
for bar, wid in zip(bars, widths_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, wid + 0.08,
             f'{wid:.2f}', ha='center', fontsize=7, fontweight='bold')

# 4.5x annotation
ax2.annotate('', xy=(1, 0.85), xytext=(0, 2.85),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.0, connectionstyle='arc3,rad=-0.2'))
ax2.text(0.5, 1.7, '$4.5\\times$\nnarrower', ha='center', fontsize=7, fontweight='bold')

plt.tight_layout(w_pad=0.8)
plt.savefig(f'{OUT}/fig3_polaris.pdf', dpi=300)
plt.close()
print("Fig 3: polaris done")

# ============================================================================
# Figure 4: Error-bin CP coverage (clean grouped bar)
# ============================================================================
fig, ax = plt.subplots(figsize=(3.3, 2.2))

errors = ['Exact', '$\\pm$1', '$\\pm$2', '$\\pm$3', '$\\pm$4']
pct_samples = ['32.6%', '43.0%', '16.0%', '6.3%', '2.0%']
cov_raw = [99.8, 98.7, 84.3, 25.4, 5.7]
cov_adj = [100.0, 99.9, 99.4, 91.5, 43.3]

x = np.arange(len(errors))
w = 0.3

ax.bar(x - w/2, cov_raw, w, label='Raw', color=C_LLAVA, edgecolor='black', linewidth=0.3)
ax.bar(x + w/2, cov_adj, w, label='Boundary adj.', color=C_GEMINI, edgecolor='black', linewidth=0.3)
ax.axhline(y=90, color='black', linestyle='--', alpha=0.4, linewidth=0.7)
ax.text(4.4, 91, '90%', fontsize=6, color='gray')

ax.set_ylabel('CP Coverage (%)')
ax.set_xlabel('Judge Error Magnitude')
ax.set_xticks(x)
ax.set_xticklabels([f'{e}\n({p})' for e, p in zip(errors, pct_samples)], fontsize=6.5)
ax.legend(fontsize=6, loc='lower left', frameon=True, fancybox=False, edgecolor='gray')
ax.set_ylim(0, 108)

plt.savefig(f'{OUT}/fig4_error_bins.pdf', dpi=300)
plt.close()
print("Fig 4: error_bins done")

# ============================================================================
# Figure 5: Mondrian CP comparison (NEW — not in original)
# ============================================================================
fig, ax = plt.subplots(figsize=(3.3, 2.0))

groups = ['Easy', 'Medium', 'Hard', 'Overall']
std_width = [3.55, 3.60, 3.63, 3.60]
mon_width = [2.96, 3.49, 3.78, 3.47]

x = np.arange(len(groups))
w = 0.3

ax.bar(x - w/2, std_width, w, label='Standard R2CCP', color=C_LLAVA, edgecolor='black', linewidth=0.3)
ax.bar(x + w/2, mon_width, w, label='Mondrian R2CCP', color=C_GEMINI, edgecolor='black', linewidth=0.3)

# Improvement annotations
improvements = ['-16.6%', '-3.1%', '+4.1%', '-3.6%']
for i, imp in enumerate(improvements):
    color = 'black'
    y_pos = max(std_width[i], mon_width[i]) + 0.05
    ax.text(i, y_pos, imp, ha='center', fontsize=6, fontweight='bold', color=color)

ax.set_ylabel('Width (adj.)')
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=7)
ax.set_ylim(2.5, 4.2)
ax.legend(fontsize=6, loc='upper right', frameon=True, fancybox=False, edgecolor='gray')

plt.savefig(f'{OUT}/fig5_mondrian.pdf', dpi=300)
plt.close()
print("Fig 5: mondrian done")

# ============================================================================
# Figure 6: Multi-judge comparison (NEW — shows negative result)
# ============================================================================
fig, ax = plt.subplots(figsize=(3.3, 2.0))

configs = ['Gemini\n(5 feat)', 'LLaVA+\nGemini\n(10)', 'LLaVA\n(5 feat)', 'All 3\n(15 feat)', 'LLaVA+\nPhi-4\n(10)', 'Phi-4\n(5 feat)']
widths_mj = [2.85, 3.02, 3.05, 3.14, 3.18, 3.13]

colors_mj = [C_GEMINI, '#B0B0B0', C_LLAVA, '#707070', '#B0B0B0', C_PHI4]
bars = ax.bar(range(len(configs)), widths_mj, color=colors_mj, edgecolor='black', linewidth=0.3, width=0.6)

# Highlight best (Gemini alone)
bars[0].set_edgecolor('black')
bars[0].set_linewidth(1.5)

ax.set_ylabel('R2CCP Width (raw)')
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(configs, fontsize=5.5)
ax.set_ylim(2.5, 3.4)

# Arrow from multi-judge to gemini showing it's worse
ax.annotate('Best: single judge', xy=(0, 2.87), fontsize=6, fontweight='bold',
            ha='center', va='bottom')

plt.savefig(f'{OUT}/fig6_multijudge.pdf', dpi=300)
plt.close()
print("Fig 6: multijudge done")

print(f"\nAll 6 figures generated in {OUT}/")
