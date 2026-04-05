import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths
PCA_PATH = "/Users/vyshakathreya/Documents/subliminal_learning_Vyshak/outputs/real_mgnify_hierarchical/samples_pca2.csv"
OUTPUT_DIR = "/Users/vyshakathreya/Documents/subliminal_learning_Vyshak/outputs/real_mgnify_hierarchical"
SAVE_PATH = os.path.join(OUTPUT_DIR, "pca_plot.png")

def main():
    if not os.path.exists(PCA_PATH):
        print(f"Error: {PCA_PATH} not found.")
        return

    # Load data
    df = pd.read_csv(PCA_PATH)
    
    # Determine environment from sample_id
    def get_env(sid):
        if sid.startswith("marine"):
            return "Marine"
        elif sid.startswith("freshwater"):
            return "Freshwater"
        return "Unknown"

    df['environment'] = df['sample_id'].apply(get_env)
    
    # Define colors
    colors = {"Marine": "royalblue", "Freshwater": "forestgreen", "Unknown": "gray"}
    
    # Create plot
    plt.figure(figsize=(10, 7), dpi=150)
    
    # Use dark theme aesthetics
    plt.style.use('dark_background')
    
    for env, group in df.groupby('environment'):
        plt.scatter(group['pc1'], group['pc2'], 
                    label=env, 
                    color=colors.get(env, "gray"), 
                    s=150, edgecolors='white', alpha=0.9, zorder=3)
        
        # Label points
        for _, row in group.iterrows():
            plt.text(row['pc1'], row['pc2'] + 0.05, 
                     row['sample_id'].split("__")[-1], 
                     fontsize=8, ha='center', color='white', zorder=5)

    # Styling
    plt.title("Latent Space Visualization: Marine vs Freshwater", fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("Principal Component 1 (PC1)", fontsize=12)
    plt.ylabel("Principal Component 2 (PC2)", fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
    plt.legend(title="Environment", loc='upper right', frameon=True, facecolor='black', edgecolor='white')
    
    # Add subtle annotations for Subliminal Learning
    plt.annotate("Zero-Shot Adaptation Clusters", 
                 xy=(0.02, 0.02), xycoords='axes fraction', 
                 fontsize=10, fontstyle='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"Plot saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
