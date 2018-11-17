import csv
import math
from matplotlib import rc
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import seaborn as sns

LATEX_FLAG = True


def pandas_main():
    proj3_df = pd.read_csv('data/proj_3.csv', header=None)
    proj3_df.columns = ['Timestamp', 'Hostname', 'DiskNumber', 'Type', 'Offset', 'Size', 'ResponseTime']
    start_series = proj3_df['Timestamp'] * 1e-4
    ia_series = proj3_df['Timestamp'].diff() * 1e-4
    service_series = proj3_df['ResponseTime'] * 1e-4

    print(start_series[:10])
    print(ia_series[:10])

    metrics_df = pd.read_csv('data/output.csv')
    sns.pairplot(metrics_df)
    plt.savefig('test.png')

    if LATEX_FLAG:
        plt.style.use('ggplot')

        rc('font', **{'family': 'serif'})
        rc('text', usetex=True)

    # plot_acf(service_series, lags=10)
    # plt.savefig('s_autocorr.png')


def csv_main():
    with open("data/proj_3.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        process_arr = []
        ia_mean_arr = []
        ia_variance_arr = []
        ia_std_dev_arr = []
        ia_cv_arr = []

        s_mean_arr = []
        s_variance_arr = []
        s_std_dev_arr = []
        s_cv_arr = []

        ia_mean = 0
        ia_variance = 0
        ia_std_dev = 0
        ia_cv = 0

        s_mean = 0
        s_variance = 0
        s_std_dev = 0
        s_cv = 0

        ia_arr = []
        s_arr = []

        service_zero_count = 0

        previous_start = 0

        for index, row in enumerate(csv_reader):
            if index == 0:
                previous_start = int(row[0])

            ia = (int(row[0]) - previous_start) * 1e-4
            previous_start = int(row[0]) * 1e-4
            service = int(row[6]) * 1e-4

            ia_arr.append(ia)
            s_arr.append(service)

            ia_variance = ia_variance + (index / (index + 1) * (ia - ia_mean) ** 2)
            s_variance = s_variance + (index / (index + 1) * (service - s_mean) ** 2)
            ia_variance_arr.append(ia_variance)
            s_variance_arr.append(s_variance)

            ia_mean = ia_mean + (ia - ia_mean) / (index + 1)
            s_mean = s_mean + (service - s_mean) / (index + 1)
            ia_mean_arr.append(ia_mean)
            s_mean_arr.append(s_mean)

            ia_std_dev = math.sqrt(ia_variance)
            s_std_dev = math.sqrt(s_std_dev)
            ia_std_dev_arr.append(ia_std_dev)
            s_std_dev_arr.append(s_std_dev)

            if ia_mean == 0:
                ia_cv = 0
            else:
                ia_cv = ia_std_dev / ia_mean
            s_cv = s_std_dev / s_mean
            ia_cv_arr.append(ia_cv)
            s_cv_arr.append(s_cv)

            process_arr.append(index + 1)

    # with open('no_metrics.csv', 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile, delimiter=',')
    #     csv_writer.writerow(['Process', 'Interarrival', 'Service'])
    #     for i, process in enumerate(process_arr):
    #         csv_writer.writerow([process, ia_arr[i], s_arr[i]])

    print("Total rows processed: {}".format(len(process_arr)))
    print("Length of Service Array: {}".format(len(s_arr)))

    if LATEX_FLAG:
        plt.style.use('ggplot')

        rc('font', **{'family': 'serif'})
        rc('text', usetex=True)

    # plt.title("Variance Vs. Time")
    # plt.scatter(process_arr, ia_variance_arr, c='g', label="Interarrival")
    # # plt.scatter(process_arr, s_variance_arr, c='r', label="Service")
    # plt.xlabel('Process')
    # plt.ylabel('Variance')
    # plt.legend(loc=9)
    #
    # plt.savefig('output/variance_ia.png')

    mean_process = pd.DataFrame(
        {"Process": process_arr,
         "Interarrival Mean": ia_mean_arr,
         "Service Mean": s_mean_arr
         })

    sns.scatterplot(x='Process', y='Interarrival Mean', data=mean_process[:1000])
    plt.savefig('output/ia_mean_seaborn.png')

    # plt.title("CDF of Interarrival Times")
    # plt.hist(ia_arr, 10000, density=True, cumulative=True)
    # plt.xlabel('Interarrival Time')
    # plt.ylabel('Cumulative Probability')
    #
    # plt.savefig('output/hist_ia_10000_cum.png')

    # plt.title("Autocorrelation of Service Times")
    # plt.acorr(s_arr[:10000])
    # plt.xlabel('Lag')
    # plt.ylabel('Autocorrelation')
    # plt.savefig('output/autocorr/s_autocorr_sub.png')

    print("Figure saved.")


if __name__ == "__main__":
    pandas_main()
