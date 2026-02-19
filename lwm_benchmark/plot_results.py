import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Plot benchmark results from CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to benchmark CSV file")
    parser.add_argument("--out", type=str, default=None, help="Output image file path (optional)")
    parser.add_argument("--title", type=str, default="Benchmark Results", help="Plot title")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check required columns
    required_cols = {'model', 'input_type', 'split_ratio', 'f1_mean', 'f1_std'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV missing required columns: {required_cols - set(df.columns)}")
        return

    # Set style manually
    plt.style.use('ggplot') 
    
    input_types = df['input_type'].unique()
    models = df['model'].unique() 
    
    # Label Mappings
    model_labels = {
        'base': 'Base LWM',
        'ca': 'CA-LWM',
        'axial': 'Axial LWM',
        'physics': 'Physics LWM'
    }
    input_labels = {
        'cls_emb': 'CLS Token',
        'channel_emb': 'Channel Embeddings',
        'raw': 'Raw Channels'
    }

    # Define colors and markers manually
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    model_color_map = {model: colors[i % len(colors)] for i, model in enumerate(models)}
    
    # Create subplots: 1 row, N columns (where N = number of input types)
    num_inputs = len(input_types)
    fig, axes = plt.subplots(1, num_inputs, figsize=(6 * num_inputs, 6), sharey=True)
    if num_inputs == 1: axes = [axes] # Handle single subplot case
    
    for i, input_type in enumerate(input_types):
        ax = axes[i]
        subset = df[df['input_type'] == input_type]
        
        for model in models:
            model_subset = subset[subset['model'] == model].sort_values('split_ratio')
            if model_subset.empty:
                continue
            
            # Label based on model only (Input type is in title)
            label_text = model_labels.get(model, model)
            
            ax.errorbar(
                model_subset['split_ratio'], 
                model_subset['f1_mean'], 
                yerr=model_subset['f1_std'], 
                label=label_text,
                color=model_color_map[model],
                linestyle='-',
                marker=markers[list(models).index(model) % len(markers)],
                capsize=5,
                linewidth=2,
                markeredgewidth=1,
                markeredgecolor='black'
            )
        
        ax.set_title(input_labels.get(input_type, input_type), fontsize=14)
        ax.set_xlabel("Split Ratio", fontsize=12)
        if i == 0:
            ax.set_ylabel("F1 Score (Weighted)", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Legend only in first plot (or shared?)
        # Adding to all is fine if not too crowded, but adding to first is cleaner if shared.
        ax.legend(loc='lower right', fontsize=10)

    plt.suptitle(args.title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle

    if args.out:
        print(f"Saving plot to {args.out}")
        plt.savefig(args.out, dpi=300, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    main()
