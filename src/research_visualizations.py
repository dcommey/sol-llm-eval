"""
Publication-quality visualization generation for ICBC 2026 paper.
Professional research-grade figures with Nature/Science styling.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, MultipleLocator
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Nature/Science publication color palette (Wong palette - colorblind safe)
WONG_PALETTE = {
    'blue': '#0173B2',
    'orange': '#DE8F05', 
    'green': '#029E73',
    'red': '#D55E00',
    'purple': '#CC78BC',
    'brown': '#CA9161',
    'pink': '#FBAFE4',
    'gray': '#949494',
}

MODEL_COLORS = {
    'qwen': WONG_PALETTE['blue'],
    'deepseek': WONG_PALETTE['orange'],
    'codellama': WONG_PALETTE['green'],
    'mistral': WONG_PALETTE['purple'],
}

MODEL_ORDER = ['qwen', 'deepseek', 'codellama', 'mistral']
MODEL_DISPLAY = {
    'qwen': 'Qwen2.5-Coder',
    'deepseek': 'DeepSeek-Coder',
    'codellama': 'CodeLLaMA',
    'mistral': 'Mistral'
}


def setup_publication_style():
    """Configure matplotlib for high-impact journal figures."""
    plt.rcParams.update({
        # Font settings - use Computer Modern for LaTeX compatibility
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '0.8',
        
        # Figure settings
        'figure.dpi': 300,
        'figure.figsize': (3.5, 2.8),  # Single column width
        'figure.facecolor': 'white',
        'savefig.dpi': 600,  # High res for print
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        
        # Axes settings
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '0.2',
        'axes.labelcolor': '0.1',
        'axes.facecolor': 'white',
        
        # Grid
        'axes.grid': False,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.5,
        
        # Ticks
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.color': '0.2',
        'ytick.color': '0.2',
        
        # Lines and patches
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.5,
        'patch.edgecolor': '0.2',
        
        # Use LaTeX for text rendering if available
        'text.usetex': False,  # Set to True if LaTeX is installed
        'mathtext.fontset': 'cm',
    })


class PublicationVisualizer:
    """Generates high-impact journal quality figures for ICBC 2026."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.paper_figures_dir = Path(config['paper']['figures_dir'])
        self.paper_tables_dir = Path(config['paper']['tables_dir'])
        
        self.paper_figures_dir.mkdir(parents=True, exist_ok=True)
        self.paper_tables_dir.mkdir(parents=True, exist_ok=True)
        
        setup_publication_style()
    
    def _get_metric_range(self, results: Dict, metric: str) -> Tuple[float, float]:
        """Dynamically determine y-axis range for a metric."""
        values = [results[k]['overall_metrics'][metric] 
                  for k in MODEL_ORDER if k in results]
        if not values:
            return (0, 1)
        min_val = min(values)
        max_val = max(values)
        padding = (max_val - min_val) * 0.15 or 0.1
        return (max(0, min_val - padding), min(1.0, max_val + padding + 0.05))
    
    def plot_overall_performance(self, results: Dict) -> None:
        """Create publication-quality grouped bar chart."""
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        
        models = [MODEL_DISPLAY[k] for k in MODEL_ORDER if k in results]
        x = np.arange(len(models))
        width = 0.25
        
        # Extract metrics
        precision = [results[k]['overall_metrics']['precision'] for k in MODEL_ORDER if k in results]
        recall = [results[k]['overall_metrics']['recall'] for k in MODEL_ORDER if k in results]
        f1 = [results[k]['overall_metrics']['f1_score'] for k in MODEL_ORDER if k in results]
        
        # Professional color scheme
        colors = [WONG_PALETTE['blue'], WONG_PALETTE['orange'], WONG_PALETTE['green']]
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', 
                       color=colors[0], edgecolor='white', linewidth=0.5, zorder=3)
        bars2 = ax.bar(x, recall, width, label='Recall',
                       color=colors[1], edgecolor='white', linewidth=0.5, zorder=3)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score',
                       color=colors[2], edgecolor='white', linewidth=0.5, zorder=3)
        
        # Add value labels on F1 bars only
        for bar in bars3:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        # Highlight best F1
        best_idx = np.argmax(f1)
        ax.annotate('BEST', xy=(x[best_idx] + width, f1[best_idx] + 0.04),
                   fontsize=7, color=WONG_PALETTE['green'], ha='center', fontweight='bold')
        
        # Dynamic y-axis
        y_min, y_max = self._get_metric_range(results, 'f1_score')
        ax.set_ylim([0, min(1.05, max(f1) + 0.15)])
        
        ax.set_xlabel('Model', fontweight='medium')
        ax.set_ylabel('Score', fontweight='medium')
        ax.set_title('Vulnerability Detection Performance', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right')
        ax.legend(loc='upper right', frameon=True, fancybox=False)
        
        # Subtle grid
        ax.yaxis.grid(True, linestyle='-', alpha=0.2, zorder=0)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        fig.savefig(self.paper_figures_dir / 'overall_performance.pdf')
        plt.close(fig)
        logger.info("Saved publication-quality overall_performance.pdf")
    
    def plot_per_vulnerability_heatmap(self, results: Dict) -> None:
        """Create professional heatmap with proper color scaling."""
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        
        vuln_types = ['reentrancy', 'integer_overflow', 'unchecked_low_level_calls']
        vuln_display = ['Reentrancy', 'Integer\nOverflow', 'Unchecked\nCalls']
        
        # Build matrix
        matrix = []
        model_names = []
        for model_key in MODEL_ORDER:
            if model_key in results:
                model_names.append(MODEL_DISPLAY[model_key])
                row = []
                for vuln in vuln_types:
                    per_vuln = results[model_key].get('per_vulnerability_metrics', {})
                    f1 = per_vuln.get(vuln, {}).get('f1_score', 0)
                    row.append(f1)
                matrix.append(row)
        
        if not matrix:
            logger.warning("No data for heatmap")
            return
            
        matrix = np.array(matrix)
        
        # Use a standard professional sequential colormap
        cmap = 'YlGnBu' 
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
        cbar.set_label('F1-Score', fontsize=9, fontweight='medium')
        cbar.ax.tick_params(labelsize=8)
        
        # Ticks
        ax.set_xticks(np.arange(len(vuln_types)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels(vuln_display, fontsize=9)
        ax.set_yticklabels(model_names, fontsize=9)
        
        # Annotations with smart color contrast
        for i in range(len(model_names)):
            for j in range(len(vuln_types)):
                value = matrix[i, j]
                # YlGnBu is dark at high values (near 1.0) and light at low values
                # Threshold ~0.5 for black text, >0.5 for white text checks
                color = 'white' if value > 0.6 else 'black'
                ax.text(j, i, f'{value:.2f}',
                       ha='center', va='center', color=color,
                       fontsize=9, fontweight='bold')
        
        ax.set_title('Detection Performance by Vulnerability', pad=10, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(self.paper_figures_dir / 'per_vulnerability_heatmap.pdf')
        plt.close(fig)
        logger.info("Saved publication-quality per_vulnerability_heatmap.pdf")
    
    def plot_confusion_matrix_grid(self, results: Dict) -> None:
        """Create 2x2 confusion matrix grid for all models."""
        n_models = len([k for k in MODEL_ORDER if k in results])
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(5, 4.5))
        axes = axes.flatten()
        
        for idx, model_key in enumerate(MODEL_ORDER):
            if model_key not in results or idx >= 4:
                continue
            
            cm = np.array(results[model_key].get('confusion_matrix', [[0,0],[0,0]]))
            
            # Ensure 2x2
            if cm.shape != (2, 2):
                cm = np.zeros((2, 2))
            
            ax = axes[idx]
            
            # Normalize for display
            cm_display = cm.astype(int)
            
            # Use professional colormap
            im = ax.imshow(cm, cmap='Blues', aspect='equal')
            
            # Annotations
            for i in range(2):
                for j in range(2):
                    val = cm_display[i, j]
                    color = 'white' if val > cm.max() * 0.5 else 'black'
                    ax.text(j, i, f'{val}', ha='center', va='center', 
                           fontsize=10, fontweight='bold', color=color)
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Clean', 'Vuln'], fontsize=8)
            ax.set_yticklabels(['Clean', 'Vuln'], fontsize=8)
            ax.set_xlabel('Predicted', fontsize=8)
            ax.set_ylabel('Actual', fontsize=8)
            ax.set_title(MODEL_DISPLAY[model_key], fontsize=9, fontweight='bold')
        
        # Hide unused axes
        for idx in range(n_models, 4):
            axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices', fontsize=11, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(self.paper_figures_dir / 'confusion_matrices.pdf')
        plt.close(fig)
        logger.info("Saved publication-quality confusion_matrices.pdf")
    
    def plot_model_comparison_radar(self, results: Dict) -> None:
        """Create radar/spider chart for multi-dimensional comparison."""
        # Skip if too few metrics
        categories = ['Precision', 'Recall', 'F1-Score']
        N = len(categories)
        
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop
        
        for idx, model_key in enumerate(MODEL_ORDER):
            if model_key not in results:
                continue
            
            metrics = results[model_key]['overall_metrics']
            values = [metrics['precision'], metrics['recall'], metrics['f1_score']]
            values += values[:1]  # Complete the loop
            
            color = list(MODEL_COLORS.values())[idx]
            ax.plot(angles, values, 'o-', linewidth=1.5, label=MODEL_DISPLAY[model_key],
                   color=color, markersize=4)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        plt.tight_layout()
        fig.savefig(self.paper_figures_dir / 'model_comparison_radar.pdf')
        plt.close(fig)
        logger.info("Saved publication-quality model_comparison_radar.pdf")

    def plot_precision_recall_tradeoff(self, results: Dict) -> None:
        """Create PR scatter plot with F1 isolines for professional finish."""
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Generate F1 isoline grid
        x = np.linspace(0.01, 1, 100)
        y = np.linspace(0.01, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = 2 * X * Y / (X + Y)
        
        # Plot isolines
        cs = ax.contour(X, Y, Z, levels=[0.2, 0.4, 0.6, 0.8], colors='gray', linestyles='--', alpha=0.3, linewidths=0.8)
        ax.clabel(cs, inline=1, fontsize=8, fmt='F1=%.1f')
        
        # Plot models
        for model_key in MODEL_ORDER:
            if model_key not in results: continue
            
            p = results[model_key]['overall_metrics']['precision']
            r = results[model_key]['overall_metrics']['recall']
            color = MODEL_COLORS.get(model_key, 'black')
            name = MODEL_DISPLAY.get(model_key, model_key)
            
            ax.plot(r, p, marker='o', color=color, markersize=10, label=name, alpha=0.9, markeredgecolor='white', markeredgewidth=1)
            
            # Label points
            ax.annotate(name, (r, p), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='medium')

        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title('Precision-Recall Trade-off', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.legend(loc='lower left', fontsize=8)
        
        plt.tight_layout()
        fig.savefig(self.paper_figures_dir / 'precision_recall_tradeoff.pdf')
        plt.close(fig)
        logger.info("Saved detailed precision_recall_tradeoff.pdf")

    def plot_efficiency_tradeoff(self, results: Dict) -> None:
        """Create Effectiveness vs Efficiency scatter plot with Pareto styled annotations."""
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Hardcoded Latency Data (Mean inference time in seconds)
        latencies = {
            'qwen': 10.50,
            'mistral': 18.72,
            'deepseek': 17.55,
            'codellama': 52.55
        }
        
        for model_key in MODEL_ORDER:
            if model_key not in results or model_key not in latencies: continue
            
            f1 = results[model_key]['overall_metrics']['f1_score']
            latency = latencies[model_key]
            color = MODEL_COLORS.get(model_key, 'black')
            name = MODEL_DISPLAY.get(model_key, model_key)
            
            # Plot point
            ax.plot(latency, f1, marker='D', color=color, markersize=12, label=name, markeredgecolor='white', markeredgewidth=1.2)
            
            # Annotate
            ax.annotate(name, (latency, f1), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

        # Formatting
        ax.set_xlabel('Mean Inference Latency (s)', fontsize=10)
        ax.set_ylabel('F1-Score', fontsize=10)
        ax.set_title('Effectiveness vs. Efficiency Frontier', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add 'Better' arrow annotation
        ax.annotate('Better (Fast & Accurate)', xy=(0.05, 0.95), xycoords='axes fraction', 
                    xytext=(0.2, 0.8), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                    horizontalalignment='left', verticalalignment='top', fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        fig.savefig(self.paper_figures_dir / 'efficiency_tradeoff.pdf')
        plt.close(fig)
        logger.info("Saved professional efficiency_tradeoff.pdf")
    
    def generate_latex_tables(self, results: Dict) -> None:
        """Generate publication-quality LaTeX tables."""
        # Overall metrics table
        latex = []
        latex.append(r"\begin{table}[t]")
        latex.append(r"\centering")
        latex.append(r"\caption{Overall vulnerability detection performance on combined dataset (98 vulnerable + 43 clean contracts). Best values in \textbf{bold}.}")
        latex.append(r"\label{tab:overall}")
        latex.append(r"\small")
        latex.append(r"\begin{tabular}{@{}lcccccc@{}}")
        latex.append(r"\toprule")
        latex.append(r"\textbf{Model} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{TP} & \textbf{FP} & \textbf{TN} \\")
        latex.append(r"\midrule")
        
        # Find best values
        metrics_list = [(k, results[k]['overall_metrics']) for k in MODEL_ORDER if k in results]
        if metrics_list:
            best_p = max(m['precision'] for _, m in metrics_list)
            best_r = max(m['recall'] for _, m in metrics_list)
            best_f1 = max(m['f1_score'] for _, m in metrics_list)
            
            # Sort by F1
            sorted_models = sorted(metrics_list, key=lambda x: x[1]['f1_score'], reverse=True)
            
            for model_key, metrics in sorted_models:
                name = MODEL_DISPLAY[model_key]
                p = metrics['precision']
                r = metrics['recall']
                f1 = metrics['f1_score']
                tp = metrics.get('true_positives', 0)
                fp = metrics.get('false_positives', 0)
                tn = metrics.get('true_negatives', 0)
                
                p_str = f"\\textbf{{{p:.3f}}}" if abs(p - best_p) < 0.001 else f"{p:.3f}"
                r_str = f"\\textbf{{{r:.3f}}}" if abs(r - best_r) < 0.001 else f"{r:.3f}"
                f1_str = f"\\textbf{{{f1:.3f}}}" if abs(f1 - best_f1) < 0.001 else f"{f1:.3f}"
                
                latex.append(f"{name} & {p_str} & {r_str} & {f1_str} & {tp} & {fp} & {tn} \\\\")
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        with open(self.paper_tables_dir / 'overall_metrics.tex', 'w') as f:
            f.write('\n'.join(latex))
        
        # Per-vulnerability table
        latex2 = []
        latex2.append(r"\begin{table}[t]")
        latex2.append(r"\centering")
        latex2.append(r"\caption{F1-scores by vulnerability type. Values $\geq 0.8$ in \textbf{bold}.}")
        latex2.append(r"\label{tab:per_vuln}")
        latex2.append(r"\small")
        latex2.append(r"\begin{tabular}{@{}lccc@{}}")
        latex2.append(r"\toprule")
        latex2.append(r"\textbf{Model} & \textbf{Reentrancy} & \textbf{Int. Overflow} & \textbf{Unchecked} \\")
        latex2.append(r"\midrule")
        
        vuln_types = ['reentrancy', 'integer_overflow', 'unchecked_low_level_calls']
        
        for model_key in MODEL_ORDER:
            if model_key not in results:
                continue
            name = MODEL_DISPLAY[model_key]
            scores = []
            per_vuln = results[model_key].get('per_vulnerability_metrics', {})
            for vuln in vuln_types:
                f1 = per_vuln.get(vuln, {}).get('f1_score', 0)
                if f1 >= 0.8:
                    scores.append(f"\\textbf{{{f1:.2f}}}")
                else:
                    scores.append(f"{f1:.2f}")
            
            latex2.append(f"{name} & {scores[0]} & {scores[1]} & {scores[2]} \\\\")
        
        latex2.append(r"\bottomrule")
        latex2.append(r"\end{tabular}")
        latex2.append(r"\end{table}")
        
        with open(self.paper_tables_dir / 'per_vulnerability_metrics.tex', 'w') as f:
            f.write('\n'.join(latex2))
        
        logger.info("Saved publication-quality LaTeX tables")
    
    def generate_all(self, results: Dict) -> None:
        """Generate all publication-quality visualizations."""
        logger.info("Generating ICBC 2026 publication-quality figures...")
        
        self.plot_overall_performance(results)
        self.plot_per_vulnerability_heatmap(results)
        self.plot_confusion_matrix_grid(results)
        self.plot_model_comparison_radar(results)
        self.plot_precision_recall_tradeoff(results)
        self.plot_efficiency_tradeoff(results)
        self.generate_latex_tables(results)
        
        logger.info("All publication-quality visualizations generated!")
