import matplotlib.pyplot as plt

def plot_histogram(values, pollutant):
    plt.figure(1);
    plt.hist(values, bins=50);
    plt.title(pollutant + ' values histogram');
    plt.show();
