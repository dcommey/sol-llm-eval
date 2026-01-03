"""
Visualization generation for paper figures and tables.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Generates publication-quality figures and tables."""
    
    def __init__(self, config: Dict):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.viz_config = config['visualization']
        self.output_dir = Path(config['output']['figures_dir'])
        self.paper_figures_dir = Path(config['paper']['figures_dir'])
        self.paper_tables_dir = Path(config['paper']['tables_dir'])
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paper_figures_dir.mkdir(parents=True, exist_ok=True)
        self.paper_tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(self.viz_config['style'])
        sns.set_palette("husl")
    
    def _save_figure(self, fig, filename: str, for_paper: bool = True):
        """Save figure to both results and paper directories."""
        dpi = self.viz_config['dpi']
        fmt = self.viz_config['format']
        
        # Save to results
        results_path = self.output_dir / f"{filename}.{fmt}"
        fig.savefig(results_path, dpi=dpi, bbox_inches='tight')
        
        # Save to paper if requested
        if for_paper:
            paper_path = self.paper_figures_dir / f"{filename}.{fmt}"
            fig.savefig(paper_path, dpi=dpi, bbox_inches='tight')
        
        plt.close(fig)
        logger.info(f"Saved figure: {filename}")
    
    def plot_overall_performance(self, evaluation_results: Dict) -> None:
        """Create bar chart comparing overall performance across models."""
        if not self.viz_config['figures']['overall_performance']['enabled']:
            return
        
        metrics_to_plot = self.viz_config['figures']['overall_performance']['metrics']
        
        # Extract data
        models = []
        precision_vals = []
        recall_vals = []
        f1_vals = []
        
        for model_key, results in evaluation_results.items():
            models.append(results['display_name'])
            metrics = results['overall_metrics']
            precision_vals.append(metrics['precision'])
            recall_vals.append(metrics['recall'])
            f1_vals.append(metrics['f1_score'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(models))
        width = 0.25
        
        ax.bar(x - width, precision_vals, width, label='Precision', alpha=0.8)
        ax.bar(x, recall_vals, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        self._save_figure(fig, 'overall_performance')
    
    def plot_per_vulnerability_performance(self, evaluation_results: Dict) -> None:
        """Create grouped bar chart for per-vulnerability-type performance."""
        if not self.viz_config['figures']['per_vulnerability']['enabled']:
            return
        
        # Collect data
        vuln_types = []
        model_names = []
        f1_scores = []
        
        for model_key, results in evaluation_results.items():
            model_name = results['display_name']
            per_vuln = results['per_vulnerability_metrics']
            
            for vuln_type, metrics in per_vuln.items():
                vuln_types.append(vuln_type.replace('_', ' ').title())
                model_names.append(model_name)
                f1_scores.append(metrics['f1_score'])
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Vulnerability Type': vuln_types,
            'Model': model_names,
            'F1-Score': f1_scores
        })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Grouped bar chart
        unique_vulns = df['Vulnerability Type'].unique()
        x = np.arange(len(unique_vulns))
        width = 0.2
        models = df['Model'].unique()
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            offset = width * (i - len(models)/2 + 0.5)
            ax.bar(x + offset, model_data['F1-Score'].values, width, 
                   label=model, alpha=0.8)
        
        ax.set_xlabel('Vulnerability Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance by Vulnerability Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(unique_vulns, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=9)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        self._save_figure(fig, 'per_vulnerability_performance')
    
    def plot_confusion_matrices(self, evaluation_results: Dict) -> None:
        """Create confusion matrix heatmaps for all models."""
        if not self.viz_config['figures']['confusion_matrix']['enabled']:
            return
        
        n_models = len(evaluation_results)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5*rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_key, results) in enumerate(evaluation_results.items()):
            cm = np.array(results['confusion_matrix'])
            
            # Normalize if configured (with zero-division handling)
            if self.viz_config['figures']['confusion_matrix']['normalize']:
                row_sums = cm.sum(axis=1)[:, np.newaxis]
                # Avoid divide by zero: replace zeros with 1 (will result in 0/1=0)
                row_sums = np.where(row_sums == 0, 1, row_sums)
                cm = cm.astype('float') / row_sums
                fmt = '.2f'
            else:
                fmt = 'd'
            
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                       ax=axes[idx], cbar=True,
                       xticklabels=['No Vuln', 'Has Vuln'],
                       yticklabels=['No Vuln', 'Has Vuln'])
            axes[idx].set_title(f"{results['display_name']}", fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        self._save_figure(fig, 'confusion_matrices')
    
    def generate_latex_overall_metrics_table(self, evaluation_results: Dict) -> str:
        """Generate LaTeX table for overall metrics."""
        latex = []
        latex.append(r"\begin{table}[t]")
        latex.append(r"\centering")
        latex.append(r"\caption{Overall Performance Metrics Across All Models}")
        latex.append(r"\label{tab:overall_metrics}")
        latex.append(r"\begin{tabular}{lcccc}")
        latex.append(r"\toprule")
        latex.append(r"Model & Precision & Recall & F1-Score & FPR \\")
        latex.append(r"\midrule")
        
        for model_key, results in evaluation_results.items():
            model_name = results['display_name']
            metrics = results['overall_metrics']
            
            precision = f"{metrics['precision']:.3f}"
            recall = f"{metrics['recall']:.3f}"
            f1 = f"{metrics['f1_score']:.3f}"
            fpr = f"{metrics['false_positive_rate']:.3f}"
            
            latex.append(f"{model_name} & {precision} & {recall} & {f1} & {fpr} \\\\")
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        return "\n".join(latex)
    
    def generate_latex_per_vuln_table(self, evaluation_results: Dict) -> str:
        """Generate LaTeX table for per-vulnerability metrics."""
        latex = []
        latex.append(r"\begin{table}[t]")
        latex.append(r"\centering")
        latex.append(r"\caption{Performance Breakdown by Vulnerability Type}")
        latex.append(r"\label{tab:per_vuln_metrics}")
        latex.append(r"\begin{tabular}{llccc}")
        latex.append(r"\toprule")
        latex.append(r"Model & Vulnerability Type & Precision & Recall & F1-Score \\")
        latex.append(r"\midrule")
        
        for model_key, results in evaluation_results.items():
            model_name = results['display_name']
            per_vuln = results['per_vulnerability_metrics']
            
            for vuln_type, metrics in per_vuln.items():
                vuln_display = vuln_type.replace('_', ' ').title()
                precision = f"{metrics['precision']:.3f}"
                recall = f"{metrics['recall']:.3f}"
                f1 = f"{metrics['f1_score']:.3f}"
                
                latex.append(f"{model_name} & {vuln_display} & {precision} & {recall} & {f1} \\\\")
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        return "\n".join(latex)
    
    def save_all_latex_tables(self, evaluation_results: Dict) -> None:
        """Save all LaTeX tables to files."""
        # Overall metrics table
        overall_table = self.generate_latex_overall_metrics_table(evaluation_results)
        overall_path = self.paper_tables_dir / "overall_metrics.tex"
        with open(overall_path, 'w') as f:
            f.write(overall_table)
        logger.info(f"Saved LaTeX table: overall_metrics.tex")
        
        # Per-vulnerability table
        per_vuln_table = self.generate_latex_per_vuln_table(evaluation_results)
        per_vuln_path = self.paper_tables_dir / "per_vulnerability_metrics.tex"
        with open(per_vuln_path, 'w') as f:
            f.write(per_vuln_table)
        logger.info(f"Saved LaTeX table: per_vulnerability_metrics.tex")
    
    def generate_all_visualizations(self, evaluation_results: Dict) -> None:
        """Generate all figures and tables for the paper."""
        logger.info("Generating all visualizations")
        
        # Figures
        self.plot_overall_performance(evaluation_results)
        self.plot_per_vulnerability_performance(evaluation_results)
        self.plot_confusion_matrices(evaluation_results)
        
        # LaTeX tables
        self.save_all_latex_tables(evaluation_results)
        
        logger.info("All visualizations generated successfully")
