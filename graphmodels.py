import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def all_LSTM_sum():

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

def grammar_results_token():

    import matplotlib.pyplot as plt
    import numpy as np

    # Temperature settings
    temps = [0.2, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0]
    x = np.arange(len(temps))
    bar_width = 0.35

    # Averaged error data for each checkpoint
    ckpt80_avg_errors = [
        {'PUNCTUATION': 2.20, 'REPETITIONS_STYLE': 0.40},
        {'PUNCTUATION': 1.80, 'GRAMMAR': 0.40, 'COLLOCATIONS': 0.20},
        { 'PUNCTUATION': 2.80, 'GRAMMAR': 1.00, 'TYPOGRAPHY': 0.20},
        { 'PUNCTUATION': 4.20, 'GRAMMAR': 1.40, 'MISC': 0.40, 'CONFUSED_WORDS': 0.20, 'TYPOGRAPHY': 0.20},
        { 'PUNCTUATION': 5.40, 'MISC': 1.20, 'GRAMMAR': 1.20, 'CASING': 0.60, 'TYPOGRAPHY': 0.20},
        { 'PUNCTUATION': 5.40, 'GRAMMAR': 2.80, 'CASING': 1.60, 'MISC': 0.80, 'TYPOGRAPHY': 0.60, 'CONFUSED_WORDS': 0.20, 'AMERICAN_ENGLISH': 0.20},
        { 'PUNCTUATION': 4.00, 'GRAMMAR': 2.40, 'CASING': 1.40, 'MISC': 0.40, 'STYLE': 0.20, 'TYPOGRAPHY': 0.20},
    ]

    ckpt40_avg_errors = [
        {'PUNCTUATION': 2.20,  'GRAMMAR': 0.20},
        {'PUNCTUATION': 3.20,  'GRAMMAR': 0.60},
        { 'PUNCTUATION': 2.40, 'GRAMMAR': 1.20, 'MISC': 0.80, 'CONFUSED_WORDS': 0.40, 'AMERICAN_ENGLISH': 0.20},
        { 'PUNCTUATION': 4.20, 'GRAMMAR': 2.00, 'MISC': 0.80, 'CASING': 0.40, 'REDUNDANCY': 0.20, 'CONFUSED_WORDS': 0.20, 'BRE_STYLE_OXFORD_SPELLING': 0.20},
        { 'PUNCTUATION': 4.20, 'GRAMMAR': 2.20, 'CASING': 0.60, 'MISC': 0.60, 'AMERICAN_ENGLISH': 0.40, 'TYPOGRAPHY': 0.40, 'CONFUSED_WORDS': 0.20, 'COLLOCATIONS': 0.20},
        { 'PUNCTUATION': 4.00, 'MISC': 1.80, 'GRAMMAR': 1.20, 'CASING': 1.00, 'TYPOGRAPHY': 0.60, 'STYLE': 0.20},
        { 'PUNCTUATION': 3.80, 'GRAMMAR': 1.60, 'CASING': 1.20, 'MISC': 0.60, 'TYPOGRAPHY': 0.20, 'COLLOCATIONS': 0.20, 'CONFUSED_WORDS': 0.20},
    ]

    # Get all unique categories from both checkpoints
    all_categories = sorted(set(k for d in ckpt80_avg_errors + ckpt40_avg_errors for k in d.keys()))

    # Assign colors to categories using a consistent color map
    color_map = plt.cm.get_cmap('tab20', len(all_categories))
    category_colors = {cat: color_map(i) for i, cat in enumerate(all_categories)}

    # Build matrix for plotting
    def build_matrix(errors_list):
        matrix = {cat: [] for cat in all_categories}
        for entry in errors_list:
            for cat in all_categories:
                matrix[cat].append(entry.get(cat, 0))
        return matrix

    matrix_80 = build_matrix(ckpt80_avg_errors)
    matrix_40 = build_matrix(ckpt40_avg_errors)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    bottom80 = np.zeros(len(temps))
    bottom40 = np.zeros(len(temps))

    for cat in all_categories:
        vals80 = matrix_80[cat]
        vals40 = matrix_40[cat]
        color = category_colors[cat]
        
        ax.bar(x - bar_width/2, vals80, bar_width, label=cat if cat not in ax.get_legend_handles_labels()[1] else "", color=color, bottom=bottom80)
        ax.bar(x + bar_width/2, vals40, bar_width, color=color, bottom=bottom40)
        
        bottom80 += np.array(vals80)
        bottom40 += np.array(vals40)

    ax.set_title("Average Error Categories by Temperature (Stacked Bars)")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Average Number of Errors")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in temps])
    ax.legend(title="Error Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    plt.tight_layout()
    plt.show()

grammar_results_token()