
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_academic_plots(en_csv, vn_csv, output_path):
    # Load data
    df_en = pd.read_csv(en_csv)
    df_vn = pd.read_csv(vn_csv)
    
    # Common X-axis
    x_en = df_en['threshold']
    x_vn = df_vn['threshold']
    
    # Setup Figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Styles
    styles = {
        'VN': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'EN': {'color': 'red', 'marker': '^', 'linestyle': '--'}
    }
    
    # Subplot 1: Accuracy
    axes[0].plot(x_vn, df_vn['accuracy'], **styles['VN'], label='Vietnamese (ViHOS)')
    axes[0].plot(x_en, df_en['accuracy'], **styles['EN'], label='English (HateXplain)')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xlabel('Gating Threshold ($\\tau$)', fontsize=12)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: Macro-F1
    axes[1].plot(x_vn, df_vn['f1_macro'], **styles['VN'], label='Vietnamese (ViHOS)')
    axes[1].plot(x_en, df_en['f1_macro'], **styles['EN'], label='English (HateXplain)')
    axes[1].set_ylabel('Macro-F1', fontsize=12)
    axes[1].set_xlabel('Gating Threshold ($\\tau$)', fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 3: Explainability (Span Metric)
    # Using Span mF1 for VN and Span IoU-F1 for EN
    axes[2].plot(x_vn, df_vn['span_mf1'], **styles['VN'], label='VN (Span mF1)')
    axes[2].plot(x_en, df_en['iou_f1'], **styles['EN'], label='EN (Span IoU)')
    axes[2].set_ylabel('Explainability Score', fontsize=12)
    axes[2].set_xlabel('Gating Threshold ($\\tau$)', fontsize=12)
    axes[2].grid(True, linestyle=':', alpha=0.6)
    
    # Global Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated academic multi-pane plot at: {output_path}")

if __name__ == "__main__":
    vn_csv = r"c:\Projects\RATeD-V\reports\figures\sensitivity_data_vn.csv"
    en_csv = r"c:\Projects\RATeD-V\reports\figures\sensitivity_data_en.csv"
    output_fig = r"c:\Projects\RATeD-V\reports\figures\sensitivity_multi_pane.png"
    
    generate_academic_plots(en_csv, vn_csv, output_fig)
