import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Performance Test Visualization')
parser.add_argument( '-f', '--file', type=str, default='mesasurements.csv')
args = parser.parse_args()

data = pd.read_csv(f"data/{args.file}", header=None)

frameworks = data[1].unique()
anzahl_frameworks = len(frameworks)

fig, axes = plt.subplots(nrows=1, ncols=anzahl_frameworks, figsize=(6 * anzahl_frameworks, 6))

if anzahl_frameworks == 1:
    axes = [axes]

y_min = data[2].min()
y_max = data[2].max()

for ax, framework in zip(axes, frameworks):
    df_framework = data[data[1] == framework]
    ax.bar(df_framework[0], df_framework[2])
    ax.set_title(framework)
    ax.set_xlabel('Modell')
    ax.set_ylabel('Messung')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim([y_min, y_max])

plt.tight_layout()
plt.show()

