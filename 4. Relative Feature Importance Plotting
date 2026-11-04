import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Extract feature importances from trained Decision Tree model
importances = dt_classifier.feature_importances_
feature_names = X.columns

# Create Pandas Series sorted in ascending order for horizontal plotting
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

# Plotting Configuration
plt.figure(figsize=(9, 5), dpi=300)
colors = sns.color_palette("viridis", len(feat_imp))
bars = plt.barh(feat_imp.index, feat_imp.values * 100, color=colors, edgecolor='black', height=0.6)

# Annotate exact percentage values on each bar
for bar in bars:
    width = bar.get_width()
    if width > 0:
        plt.text(
            width + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.1f}%',
            va='center',
            ha='left',
            fontsize=9,
            fontweight='bold'
        )

# Styling details
plt.title("Relative Feature Importance Distribution (Decision Tree)", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Importance Weight (%)", fontsize=11, fontweight='bold')
plt.ylabel("Selected Attributes", fontsize=11, fontweight='bold')
plt.xlim(0, max(feat_imp.values * 100) + 10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save image locally for inclusion in report
plt.savefig("feature_importance_chart.png", bbox_inches='tight')
plt.show()

print("✓ Feature Importance plot saved as 'feature_importance_chart.png'.")
