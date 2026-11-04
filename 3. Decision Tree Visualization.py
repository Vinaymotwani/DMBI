import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Configure high-resolution plot window
plt.figure(figsize=(18, 9), dpi=300)

# Render Decision Tree Graph
plot_tree(
    dt_classifier,
    feature_names=X.columns,
    class_names=['At Risk', 'Pass'],
    filled=True,
    rounded=True,
    impurity=True,
    fontsize=9
)

plt.title(
    "Decision Tree Classifier Flowchart (Criterion: Entropy | Max Depth: 4)",
    fontsize=14,
    fontweight='bold',
    pad=15
)
plt.tight_layout()

# Save image locally for easy access
plt.savefig("decision_tree_structure.png", bbox_inches='tight')
plt.show()

print("✓ Decision Tree structure plot saved as 'decision_tree_structure.png'.")
