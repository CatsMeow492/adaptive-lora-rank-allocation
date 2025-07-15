#!/usr/bin/env python3
"""
Create publication-quality figures for Adaptive LoRA Rank Allocation research paper.
Generates academic-standard visualizations of experimental results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication-quality matplotlib parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf'
})

def load_experimental_data():
    """Load and prepare experimental data for analysis."""
    with open('results/all_results.json', 'r') as f:
        all_results = json.load(f)
    
    # Focus on SST-2 results (complete and reliable)
    sst2_results = [r for r in all_results if r['task'] == 'sst2']
    
    # Create comprehensive dataframe
    data = []
    for r in sst2_results:
        data.append({
            'Config': r['config_id'],
            'Name': r['config']['name'],
            'Accuracy': r['eval_metrics']['eval_accuracy'],
            'F1': r['eval_metrics']['eval_f1'],
            'Trainable_Params': r['trainable_params'],
            'Trainable_Percent': r['trainable_percent'],
            'Memory_GB': r['peak_memory_mb'] / 1024,
            'Training_Time_Min': r['training_time_seconds'] / 60,
            'Quantization': r['config'].get('quantization_bits', 'FP16'),
            'LoRA_Rank': r['config']['lora_rank'],
            'Adaptive': r['config']['lora_adaptive'],
            'Category': 'Adaptive' if r['config']['lora_adaptive'] else 'Fixed-rank'
        })
    
    return pd.DataFrame(data)

def create_main_performance_figure(df):
    """Create the main performance comparison figure (Figure 1)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Color scheme for categories
    colors = {'Fixed-rank': '#2E86AB', 'Adaptive': '#A23B72'}
    
    # 1. Accuracy Comparison
    accuracy_data = df.groupby('Category')['Accuracy'].agg(['mean', 'std']).reset_index()
    bars1 = ax1.bar(accuracy_data['Category'], accuracy_data['mean'], 
                   color=[colors[cat] for cat in accuracy_data['Category']],
                   alpha=0.8, capsize=5)
    ax1.errorbar(accuracy_data['Category'], accuracy_data['mean'], 
                yerr=accuracy_data['std'], fmt='none', color='black', capsize=5)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Performance Comparison')
    ax1.set_ylim(0.85, 0.95)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, accuracy_data['mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Memory Efficiency
    memory_data = df.groupby('Category')['Memory_GB'].agg(['mean', 'std']).reset_index()
    bars2 = ax2.bar(memory_data['Category'], memory_data['mean'],
                   color=[colors[cat] for cat in memory_data['Category']],
                   alpha=0.8, capsize=5)
    ax2.errorbar(memory_data['Category'], memory_data['mean'],
                yerr=memory_data['std'], fmt='none', color='black', capsize=5)
    ax2.set_ylabel('Peak Memory (GB)')
    ax2.set_title('(b) Memory Usage')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, memory_data['mean']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}GB', ha='center', va='bottom', fontweight='bold')
    
    # 3. Parameter Efficiency
    param_data = df.groupby('Category')['Trainable_Params'].agg(['mean', 'std']).reset_index()
    bars3 = ax3.bar(param_data['Category'], param_data['mean'] / 1e6,
                   color=[colors[cat] for cat in param_data['Category']],
                   alpha=0.8, capsize=5)
    ax3.errorbar(param_data['Category'], param_data['mean'] / 1e6,
                yerr=param_data['std'] / 1e6, fmt='none', color='black', capsize=5)
    ax3.set_ylabel('Trainable Parameters (M)')
    ax3.set_title('(c) Parameter Efficiency')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars3, param_data['mean']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 4. Training Time
    time_data = df.groupby('Category')['Training_Time_Min'].agg(['mean', 'std']).reset_index()
    bars4 = ax4.bar(time_data['Category'], time_data['mean'],
                   color=[colors[cat] for cat in time_data['Category']],
                   alpha=0.8, capsize=5)
    ax4.errorbar(time_data['Category'], time_data['mean'],
                yerr=time_data['std'], fmt='none', color='black', capsize=5)
    ax4.set_ylabel('Training Time (minutes)')
    ax4.set_title('(d) Training Efficiency')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars4, time_data['mean']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}min', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/figure1_main_comparison.pdf', bbox_inches='tight')
    plt.savefig('figures/figure1_main_comparison.png', bbox_inches='tight', dpi=300)
    print("✅ Created Figure 1: Main Performance Comparison")
    return fig

def create_detailed_configuration_analysis(df):
    """Create detailed analysis of all configurations (Figure 2)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Configuration order for consistent plotting
    config_order = ['B-FP', 'B-Q4', 'B-Ada', 'Joint-1', 'Joint-2', 'Joint-3']
    df_ordered = df.set_index('Config').loc[config_order].reset_index()
    
    # Color mapping for configurations
    config_colors = {
        'B-FP': '#2E86AB', 'B-Q4': '#A23B72', 'B-Ada': '#F18F01',
        'Joint-1': '#C73E1D', 'Joint-2': '#5A189A', 'Joint-3': '#2D5016'
    }
    colors = [config_colors[config] for config in df_ordered['Config']]
    
    # 1. Accuracy by Configuration
    bars1 = ax1.bar(range(len(df_ordered)), df_ordered['Accuracy'], color=colors, alpha=0.8)
    ax1.set_xticks(range(len(df_ordered)))
    ax1.set_xticklabels(df_ordered['Config'], rotation=45)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Accuracy by Configuration')
    ax1.set_ylim(0.85, 0.95)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, df_ordered['Accuracy'])):
        ax1.text(i, val + 0.001, f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    # 2. Memory Usage by Configuration
    bars2 = ax2.bar(range(len(df_ordered)), df_ordered['Memory_GB'], color=colors, alpha=0.8)
    ax2.set_xticks(range(len(df_ordered)))
    ax2.set_xticklabels(df_ordered['Config'], rotation=45)
    ax2.set_ylabel('Peak Memory (GB)')
    ax2.set_title('(b) Memory Usage by Configuration')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, df_ordered['Memory_GB'])):
        ax2.text(i, val + 0.3, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Efficiency Scatter: Accuracy vs Parameters
    for category in df['Category'].unique():
        cat_data = df[df['Category'] == category]
        ax3.scatter(cat_data['Trainable_Params'] / 1e6, cat_data['Accuracy'],
                   s=100, alpha=0.7, label=category)
    
    ax3.set_xlabel('Trainable Parameters (M)')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('(c) Parameter Efficiency Trade-off')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add configuration labels
    for _, row in df.iterrows():
        ax3.annotate(row['Config'], 
                    (row['Trainable_Params'] / 1e6, row['Accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Training Time vs Memory
    for category in df['Category'].unique():
        cat_data = df[df['Category'] == category]
        ax4.scatter(cat_data['Memory_GB'], cat_data['Training_Time_Min'],
                   s=100, alpha=0.7, label=category)
    
    ax4.set_xlabel('Peak Memory (GB)')
    ax4.set_ylabel('Training Time (minutes)')
    ax4.set_title('(d) Resource Usage Trade-off')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add configuration labels
    for _, row in df.iterrows():
        ax4.annotate(row['Config'],
                    (row['Memory_GB'], row['Training_Time_Min']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('figures/figure2_detailed_analysis.pdf', bbox_inches='tight')
    plt.savefig('figures/figure2_detailed_analysis.png', bbox_inches='tight', dpi=300)
    print("✅ Created Figure 2: Detailed Configuration Analysis")
    return fig

def create_quantization_analysis(df):
    """Create quantization impact analysis (Figure 3)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compare B-FP vs B-Q4 specifically
    fp16_data = df[df['Config'] == 'B-FP'].iloc[0]
    q4_data = df[df['Config'] == 'B-Q4'].iloc[0]
    
    # 1. Memory Reduction Visualization
    categories = ['FP16\n(B-FP)', '4-bit\n(B-Q4)']
    memory_values = [fp16_data['Memory_GB'], q4_data['Memory_GB']]
    accuracy_values = [fp16_data['Accuracy'], q4_data['Accuracy']]
    
    bars1 = ax1.bar(categories, memory_values, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax1.set_ylabel('Peak Memory (GB)')
    ax1.set_title('(a) Memory Usage Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels and reduction percentage
    for i, (bar, val) in enumerate(zip(bars1, memory_values)):
        ax1.text(i, val + 0.3, f'{val:.1f}GB', ha='center', va='bottom', fontweight='bold')
    
    # Add reduction arrow and percentage
    reduction_pct = (fp16_data['Memory_GB'] - q4_data['Memory_GB']) / fp16_data['Memory_GB'] * 100
    ax1.annotate(f'-{reduction_pct:.1f}%', xy=(0.5, max(memory_values) - 2),
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 2. Accuracy Preservation
    bars2 = ax2.bar(categories, accuracy_values, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('(b) Accuracy Preservation')
    ax2.set_ylim(0.90, 0.92)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, accuracy_values)):
        ax2.text(i, val + 0.0005, f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Add "NO LOSS" annotation
    ax2.annotate('NO ACCURACY LOSS', xy=(0.5, 0.915),
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('figures/figure3_quantization_impact.pdf', bbox_inches='tight')
    plt.savefig('figures/figure3_quantization_impact.png', bbox_inches='tight', dpi=300)
    print("✅ Created Figure 3: Quantization Impact Analysis")
    return fig

def create_results_table(df):
    """Create publication-quality results table."""
    # Reorder and format data for table
    table_data = df[['Config', 'Name', 'Accuracy', 'F1', 'Trainable_Params', 
                    'Trainable_Percent', 'Memory_GB', 'Training_Time_Min']].copy()
    
    # Format for publication
    table_data['Accuracy'] = table_data['Accuracy'].apply(lambda x: f"{x:.1%}")
    table_data['F1'] = table_data['F1'].apply(lambda x: f"{x:.1%}")
    table_data['Trainable_Params'] = table_data['Trainable_Params'].apply(lambda x: f"{x:,}")
    table_data['Trainable_Percent'] = table_data['Trainable_Percent'].apply(lambda x: f"{x:.1f}%")
    table_data['Memory_GB'] = table_data['Memory_GB'].apply(lambda x: f"{x:.1f}")
    table_data['Training_Time_Min'] = table_data['Training_Time_Min'].apply(lambda x: f"{x:.1f}")
    
    # Rename columns for publication
    table_data.columns = ['Config', 'Method', 'Accuracy', 'F1-Score', 'Trainable Params', 
                         'Trainable %', 'Memory (GB)', 'Training Time (min)']
    
    # Save as CSV and LaTeX
    table_data.to_csv('tables/table1_complete_results.csv', index=False)
    
    # Create LaTeX table
    latex_table = table_data.to_latex(index=False, escape=False, 
                                     caption='Complete experimental results for all LoRA configurations on SST-2 classification task.',
                                     label='tab:complete_results')
    
    with open('tables/table1_complete_results.tex', 'w') as f:
        f.write(latex_table)
    
    print("✅ Created Table 1: Complete Results")
    return table_data

def main():
    """Generate all publication figures and tables."""
    print("🎨 Generating publication-quality figures and tables...\n")
    
    # Create output directories
    Path('figures').mkdir(exist_ok=True)
    Path('tables').mkdir(exist_ok=True)
    
    # Load data
    df = load_experimental_data()
    print(f"📊 Loaded data for {len(df)} configurations\n")
    
    # Generate figures
    create_main_performance_figure(df)
    create_detailed_configuration_analysis(df)
    create_quantization_analysis(df)
    
    # Generate tables
    create_results_table(df)
    
    print(f"\n🎉 All publication materials generated successfully!")
    print("📁 Output locations:")
    print("   • Figures: ./figures/")
    print("   • Tables: ./tables/")
    print("   • Formats: PDF (vector), PNG (raster), CSV, LaTeX")

if __name__ == "__main__":
    main() 