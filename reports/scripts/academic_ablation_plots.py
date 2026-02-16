
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_ablation_plot(output_path):
    # Data from RATeD-V results
    labels = ['Vietnamese', 'English']
    
    # Explainability Values (Token-level F1 / Span WF1)
    # Stage 1: RATeD E1 (Backbone only)
    # Stage 2: LLM Zero-shot only
    # Both: Proposed Cascaded RATeD-V (E11)
    # VN (Gemini): Stage1=0.8933, Stage2=0.8451, Both=0.8938
    # EN (Gemini): Stage1=0.4400, Stage2=0.4014, Both=0.5110
    # Updated Values (Local Qwen Specialist + XLM-R)
    # Stage 1 (Backbone): 0.8933
    # Stage 2 (LLM Only): 0.9394
    # Cascaded (E11): 0.9385
    # English (Gemini) remains unchanged for now
    stage1 = [0.8933, 0.4400]  
    stage2 = [0.9394, 0.4014]  
    both   = [0.9385, 0.5110]  

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    # Patterned bars like the reference
    rects1 = ax.bar(x - width, stage1, width, label='Only Stage-1 (Backbone)', color='white', edgecolor='blue', hatch='///')
    rects2 = ax.bar(x, stage2, width, label='Only Stage-2 (LLM)', color='white', edgecolor='green', hatch='...')
    rects3 = ax.bar(x + width, both, width, label='Both (Proposed Cascaded)', color='white', edgecolor='red', hatch='\\\\\\')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Token-level F1 (Explainability)', fontsize=12)
    ax.set_title('', fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0.3, 1.0)  # Adjusted to show EN range clearly
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True)

    ax.grid(axis='y', linestyle=':', alpha=0.6)

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated ablation bar chart at: {output_path}")

if __name__ == "__main__":
    output_fig = r"c:\Projects\RATeD-V\reports\figures\ablation_study.png"
    generate_ablation_plot(output_fig)
