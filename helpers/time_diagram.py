import csv
import matplotlib
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    time_plot()
