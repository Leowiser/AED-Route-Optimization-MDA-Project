import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def line_plot(time, prop, maxresponder, aed_style, opentypes, all_data):
    plt.figure(figsize=(15, 6))  # Make room for external legend

    for i in time:
        for maxresp in maxresponder:
            for opentype in opentypes:
                x_vals = []
                y_vals = []

                for j in prop:
                    key = (i, j, maxresp, aed_style, opentype)
                    if key not in all_data:
                        print(f"⚠️ Missing key: {key}")
                        continue
                    df = all_data[key]
                    mean = df["avg_resp_prob"].mean()
                    proportion = df["proportion_of_CFR_run1"].mean()
                    x_vals.append(proportion)
                    y_vals.append(mean)

                label = f'Time: {i}h, MaxResp: {maxresp}, Open: {opentype}'
                plt.plot(x_vals, y_vals, linestyle='-', marker='o', label=label)

    plt.xlabel("Proportion of CFRs")
    plt.ylabel("Average Survival Probability")
    plt.title("Line Plot of Mean vs Proportion")

    # Put legend outside the plot on the right
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # leave space on the right
    plt.show()


def line_plot_seaborn(time, prop, maxresponder, aed_style, opentypes, all_data):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))

    cmap = plt.colormaps["Blues"]
    total_lines = len(time) * len(maxresponder) * len(opentypes)
    colors = [cmap(0.4 + 0.6 * (i / max(1, total_lines - 1))) for i in range(total_lines)]

    color_idx = 0

    for i in time:
        for maxresp in maxresponder:
            for opentype in opentypes:
                x_vals = []
                y_vals = []

                for j in prop:
                    key = (i, j, maxresp, aed_style, opentype)
                    if key not in all_data:
                        print(f"⚠️ Missing key: {key}")
                        continue
                    df = all_data[key]
                    mean = df["avg_resp_prob"].mean()
                    proportion = df["proportion_of_CFR_run1"].mean()
                    x_vals.append(proportion)
                    y_vals.append(mean)

                label = f'Time: {i}h, MaxResp: {maxresp}, Open: {opentype}'
                sns.lineplot(x=x_vals, y=y_vals, label=label, marker='o', color=colors[color_idx])
                color_idx += 1

    plt.xlabel("Proportion of CFRs", fontsize=12)
    plt.ylabel("Average Survival Probability", fontsize=12)
    plt.title("Average Survival Probability vs CFR Proportion", fontsize=14)
    plt.legend(title="Scenario", fontsize=10, title_fontsize=11, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.grid(True)
    plt.show()