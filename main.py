import csv
import math
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import seaborn as sns
import warnings
import numpy as np

LATEX_FLAG = True
USE_MS_FLAG = True
PLOT_AUTOCORRELATION = False
PLOT_DISTRIBUTIONS = False
PLOT_TRANSIENT = True
PRINT_STATS = True


def get_transient_metrics(df):
    mean = []
    std = []
    var = []
    cv = []

    for index, item in enumerate(df):
        mean.append(df[:index].mean())
        std.append(df[:index].std())
        var.append(df[:index].var())
        cv_val = float(df[:index].std / df[:index].mean())
        cv.append(cv_val)

    return np.array([mean, std, var, cv])


def pandas_main():
    proj3_df = pd.read_csv('data/proj_3.csv', header=None)
    proj3_df.columns = ['Timestamp', 'Hostname', 'DiskNumber', 'Type', 'Offset', 'Size', 'ResponseTime']

    ias_df = proj3_df[['Timestamp', 'ResponseTime']].copy()
    ias_df['Timestamp'] = ias_df['Timestamp'].diff()
    ias_df.columns = ['Interarrival', 'Service']

    if USE_MS_FLAG:
        multiplier = 1e-4
        scale = '(ms)'
    else:
        multiplier = 1e-7
        scale = '(s)'

    ias_df *= multiplier

    ia_series = ias_df['Interarrival']
    service_series = ias_df['Service']

    if PRINT_STATS:
        print('Interarrival {}: \n'.format(scale) +
              '   Mean..........: {}\n'.format(ia_series.mean()) +
              '   Std. Dev......: {}\n'.format(ia_series.std()) +
              '   Variance......: {}\n'.format(ia_series.var()) +
              '   CV............: {}'.format(ia_series.std() / ia_series.mean()))
        print('Service {}: \n'.format(scale) +
              '   Mean..........: {}\n'.format(service_series.mean()) +
              '   Std. Dev......: {}\n'.format(service_series.std()) +
              '   Variance......: {}\n'.format(service_series.var()) +
              '   CV............: {}'.format(service_series.std() / service_series.mean()))

    f_scale = scale[1:-1]

    if LATEX_FLAG:
        plt.style.use('ggplot')

        rc('font', **{'family': 'serif'})
        rc('text', usetex=True)
        pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
        rcParams.update(pgf_with_rc_fonts)

    # Plot Autocorrelations
    if PLOT_AUTOCORRELATION:
        ia_auto_fig, ia_auto_ax = plt.subplots()
        plot_acf(ia_series, ax=ia_auto_ax, lags=50, fft=True)
        ia_auto_ax.set_title('Autocorrelation of Interarrival Times {}'.format(scale))
        ia_auto_ax.set_ylabel('Correlation')
        ia_auto_ax.set_xlabel('Lag')
        ia_auto_fig.savefig('output/autocorr/ia_autocorr_{}.png'.format(f_scale))

        s_auto_fig, s_auto_ax = plt.subplots()
        plot_acf(service_series, ax=s_auto_ax, lags=50, fft=True)
        s_auto_ax.set_title('Autocorrelation of Service Times {}'.format(scale))
        s_auto_ax.set_xlabel('Lag')
        s_auto_ax.set_ylabel('Correlation')
        s_auto_fig.savefig('output/autocorr/s_autocorr_{}.png'.format(f_scale))

    # Plot Distributions
    if PLOT_DISTRIBUTIONS:
        with warnings.catch_warnings():
            bins = 100

            # Histograms
            fig, ax = plt.subplots()
            sns.distplot(service_series, bins=bins, kde=False, ax=ax)
            ax.set_title('Distribution of Service Times {}'.format(scale))
            ax.set_xlabel('Service Time {}'.format(scale))
            ax.set_ylabel('Count')
            fig.savefig('output/histogram/service_hist_{}_{}.png'.format(bins, f_scale))

            fig, ax = plt.subplots()
            sns.distplot(ia_series, bins=bins, kde=False, ax=ax)
            ax.set_title('Distribution of Interarrival Times {}'.format(scale))
            ax.set_xlabel('Interarrival Time {}'.format(scale))
            ax.set_ylabel('Count')
            fig.savefig('output/histogram/ia_hist_{}_{}.png'.format(bins, f_scale))

            fig, ax = plt.subplots()
            sns.distplot(service_series, kde=False, ax=ax)
            ax.set_title('Distribution of Service Times {}'.format(scale))
            ax.set_xlabel('Service Time {}'.format(scale))
            ax.set_ylabel('Count')
            fig.savefig('output/histogram/service_hist_{}.png'.format(f_scale))

            fig, ax = plt.subplots()
            sns.distplot(ia_series[1:], kde=False, ax=ax)
            ax.set_title('Distribution of Interarrival Times {}'.format(scale))
            ax.set_xlabel('Interarrival Time {}'.format(scale))
            ax.set_ylabel('Count')
            fig.savefig('output/histogram/ia_hist_{}.png'.format(f_scale))

            # Cumulative Distributions
            fig, ax = plt.subplots()
            sns.distplot(service_series, bins=bins, hist_kws=dict(cumulative=True),
                         kde_kws=dict(cumulative=True), ax=ax)
            ax.set_title('CDF of Service Times {}'.format(scale))
            ax.set_xlabel('Service Time {}'.format(scale))
            ax.set_ylabel('Cumulative Probability')
            fig.savefig('output/cdf/service_cdf_{}_{}.png'.format(bins, f_scale))

            fig, ax = plt.subplots()
            sns.distplot(ia_series[1:], bins=bins, hist_kws=dict(cumulative=True),
                         kde_kws=dict(cumulative=True), ax=ax)
            ax.set_title('CDF of Interarrival Times {}'.format(scale))
            ax.set_xlabel('Interarrival Time {}'.format(scale))
            ax.set_ylabel('Cumulative Probability')
            fig.savefig('output/cdf/ia_cdf_{}_{}.png'.format(bins, f_scale))

            fig, ax = plt.subplots()
            sns.distplot(service_series, hist_kws=dict(cumulative=True),
                         kde_kws=dict(cumulative=True), ax=ax)
            ax.set_title('CDF of Service Times {}'.format(scale))
            ax.set_xlabel('Service Time {}'.format(scale))
            ax.set_ylabel('Cumulative Probability')
            fig.savefig('output/cdf/service_cdf_{}.png'.format(f_scale))

            fig, ax = plt.subplots()
            sns.distplot(ia_series[1:], hist_kws=dict(cumulative=True),
                         kde_kws=dict(cumulative=True), ax=ax)
            ax.set_title('CDF of Interarrival Times {}'.format(scale))
            ax.set_xlabel('Interarrival Time {}'.format(scale))
            ax.set_ylabel('Cumulative Probability')
            fig.savefig('output/cdf/ia_cdf_{}.png'.format(f_scale))

    if PLOT_TRANSIENT:
        print(get_transient_metrics(ia_series[:100]))


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
         "Service Mean (ms)": s_mean_arr
         })

    sns.scatterplot(x='Process', y='Service Mean (ms)', data=mean_process)
    plt.savefig('output/s_mean_seaborn_ms.png')

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
