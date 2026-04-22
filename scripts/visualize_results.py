import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output", default="latent_viz.png")
    args = parser.parse_args()

    # Load data
    pca_df = pd.read_csv(args.pca_csv)
    # Extract category
    pca_df["category"] = pca_df["sample_id"].apply(lambda x: x.split("__")[0])

    # Aesthetics
    sns.set_theme(style="darkgrid", palette="viridis")
    plt.figure(figsize=(10, 8))

    # Scatter plot
    scatter = sns.scatterplot(
        data=pca_df,
        x="pc1",
        y="pc2",
        hue="category",
        style="category",
        s=100,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5
    )

    # Title and labels
    plt.title("Latent Space Visualization (PCA)", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Principal Component 1 (PC1)", fontsize=12)
    plt.ylabel("Principal Component 2 (PC2)", fontsize=12)
    
    # Legend
    plt.legend(title="Environment", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(args.output, dpi=300)
    print(f"[OK] Saved plot to: {args.output}")

if __name__ == "__main__":
    main()
