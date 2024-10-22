import csv
import math
import glob
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
#matplotlib.use('svg')


#     Q1-1.5IQR   Q1   median  Q3   Q3+1.5IQR
#                  |-----:-----|
#  o      |--------|     :     |--------|    o  o
#                  |-----:-----|
#flier             <----------->            fliers
#                       IQR

def time_plot(path='measurement/ml-pipline.csv'):
    data = {}
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            method, duration = row
            duration = float(duration)
            if method in data:
                data[method].append(duration)
            else:
                data[method] = [duration]

    methods = list(data.keys())
    runtimes = [data[m] for m in methods]

    # Boxplot erstellen
    plt.boxplot(runtimes, label=methods, tick_labels=methods)
    
    plt.xlabel('Methoden')
    plt.ylabel('Laufzeit (Sekunden)')
    plt.title('Verteilung der Laufzeiten pro Methode')
    plt.suptitle(f'Batch-Größe: ??', fontsize=10, y=0.95)
    plt.show()


def plot_pollinator_inference_scatter(csv_filename):
    # Read the CSV data
    data = pd.read_csv(csv_filename)

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data['n_flowers'], data['pollinator_inference'])
    plt.xlabel('Number of Flowers (n_flowers)')
    plt.ylabel('Pollinator Inference Time (seconds)')
    plt.title('Scatter Plot of Pollinator Inference Time vs. Number of Flowers')
    plt.show()


def plot_pollinator_inference_boxplot(filename):
    # Read the data from the string
    
    # Read the data into a DataFrame
    df = pd.read_csv(filename, header=0, usecols=['pollinator_inference', 'n_flowers'])
    
    # Ensure 'n_flowers' is treated as a categorical variable
    df['n_flowers'] = df['n_flowers'].astype(str)
    
    # Group the data by 'n_flowers'
    grouped = df.groupby('n_flowers')['pollinator_inference']
    
    # Prepare data for the box plot
    boxplot_data = [group.values for name, group in grouped]
    
    # Get the labels for each group
    labels = [name for name, group in grouped]
    
    # Create the box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(boxplot_data, labels=labels)
    plt.xlabel('Number of Flowers (n_flowers)')
    plt.ylabel('Pollinator Inference Time (seconds)')
    plt.title('Box Plot of Pollinator Inference Times by Number of Flowers')
    plt.show()


def plot_pollinator_inference_scatter(csv_filenames, titles=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    import math

    num_files = len(csv_filenames)
    cols = math.ceil(math.sqrt(num_files))
    rows = math.ceil(num_files / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, csv_filename in enumerate(csv_filenames):
        data = pd.read_csv("./measurement/" + csv_filename)
        axes[idx].scatter(data['n_flowers'], data['pollinator_inference'])
        axes[idx].set_xlabel('Number of Flowers (n_flowers)')
        axes[idx].set_ylabel('Pollinator Inference Time (seconds)')
        title = data.columns[-1].split(":")[-1].strip()
        axes[idx].set_title(title)

    for idx in range(num_files, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()



def plot_multiple_csv_scatter(csv_filenames):
    plt.figure(figsize=(10, 6))

    data_list = []
    labels = []

    # Read and store data and labels
    for csv_filename in csv_filenames:
        data = pd.read_csv("./measurement/" +csv_filename)
        x = data['n_flowers']
        y = data['pollinator_inference']
        sorted_indices = x.argsort()
        x = x.iloc[sorted_indices]
        y = y.iloc[sorted_indices]
        data_list.append((x, y))
        labels.append(data.columns[-1].split(":")[-1].strip())

    # Sort labels and data if a sorting function is provided

    # Plot each dataset
    for (x, y), label in zip(data_list, labels):
        plt.scatter(x, y)
        plt.plot(x, y)

        # Label the line directly at the last data point
        label_x = x.iloc[-1]
        label_y = y.iloc[-1]
        plt.text(label_x, label_y, label, fontsize=9, ha='left', va='center')

    plt.xlabel('Number of Flowers (n_flowers)')
    plt.ylabel('Pollinator Inference Time (seconds)')
    plt.title('Pollinator Inference Time vs. Number of Flowers')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    csv_files = glob.glob("*.csv", root_dir="./measurement")
    #plot_pollinator_inference_scatter("cpu_inference_20241022_112815.csv")
    #plot_pollinator_inference_boxplot("cpu_inference_20241022_112815.csv")
    #plot_pollinator_inference_scatter(csv_files)
    plot_multiple_csv_scatter(csv_files)
