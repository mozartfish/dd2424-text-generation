import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# with tokens
model_a = {
    "Temperature": [0.2, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0],
    "Word Diversity": [0.413, 0.488, 0.539, 0.641, 0.729, 0.786, 0.851],
    "Repetition Rate": [0.587, 0.512, 0.461, 0.359, 0.271, 0.214, 0.149],
    "Avg Sentence Length": [23.720, 16.066, 17.995, 16.393, 15.699, 23.353, 19.952],
    "Spelling Accuracy": [99.52, 98.70, 98.95, 95.90, 92.93, 90.89, 88.04]
}

# one layer lstm with characters
model_b = {
    "Temperature": [0.2, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0],
    "Word Diversity": [0.822, 0.826, 0.893, 0.887, 0.902, 0.907, 0.936],
    "Repetition Rate": [0.178, 0.174, 0.107, 0.113, 0.098, 0.093, 0.064],
    "Avg Sentence Length": [23.187, 20.967, 12.921, 13.824, 12.593, 13.000, 11.061],
    "Spelling Accuracy": [94.79, 96.64, 93.09, 91.34, 86.53, 81.96, 71.14]
}

# with tokens, checkpoint 40
model_c = {
    "Temperature": [0.2, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0],
    "Word Diversity": [0.252, 0.411, 0.531, 0.648 ,0.773,0.858, 0.897],
    "Repetition Rate": [0.748,  0.589, 0.469, 0.352,  0.227,0.142,0.103 ],

    "Avg Sentence Length": [39.954, 16.846, 15.419,  17.165,14.752, 20.631, 27.187],
    
    "Spelling Accuracy": [100.00, 98.80,97.53, 95.52, 91.00,85.38, 85.56]
}

# Convert to DataFrames
df_a = pd.DataFrame(model_a)
df_b = pd.DataFrame(model_b)
df_c = pd.DataFrame(model_c)

# Add model labels
df_a["Model"] = "LSTM with tokens, checkpoint 80"
df_b["Model"] = "Base LSTM with characters"
df_c["Model"] = "LSTM with tokens, checkpoint 40"

# Combine both datasets
df = pd.concat([df_a, df_b, df_c])

# Set seaborn theme
sns.set(style="whitegrid")

# Metrics to plot
metrics = ["Word Diversity", "Repetition Rate", "Avg Sentence Length", "Spelling Accuracy"]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    sns.lineplot(
        data=df,
        x="Temperature",
        y=metric,
        hue="Model",
        marker="o",
        ax=axes[i]
    )
    axes[i].set_title(f"{metric} vs Temperature")
    axes[i].set_xlabel("Temperature")
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.show()
