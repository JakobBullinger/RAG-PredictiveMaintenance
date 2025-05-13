import pandas as pd
import matplotlib.pyplot as plt

lat = pd.read_csv("eval/latency_raw.csv", header=None)[0]

# Font and layout settings for thesis
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11
})

# Plot histogram
plt.figure(figsize=(6, 4))
plt.hist(lat, bins=20, edgecolor="black")

# Mean line
plt.axvline(lat.mean(), linestyle="--", color='blue',
            label=f"Mean: {lat.mean():.2f}s")

# 95th percentile line
plt.axvline(lat.quantile(0.95), linestyle=":", color='deepskyblue',
            label=f"95th percentile: {lat.quantile(0.95):.2f}s")

# Target line
plt.axvline(3.0, linestyle="-", color="red", label="Target: 3.00s")

# Labels and legend
plt.xlabel("Latency (seconds)")
plt.ylabel("Request count")
plt.legend(loc='upper right')

# No title inside plot (use caption in thesis)
plt.tight_layout()
plt.savefig("eval/latency_histogram_thesis.png", dpi=300, bbox_inches='tight')
plt.show()
