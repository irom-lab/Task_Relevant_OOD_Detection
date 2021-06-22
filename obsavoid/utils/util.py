import matplotlib.pyplot as plt
import numpy as np

XLABEL_FONTSIZE = 20
YLABEL_FONTSIZE = 20
XTICKS_FONTSIZE = 15
YTICKS_FONTSIZE = 15
LEGEND_FONTSIZE = 15
YLABEL = "OOD indicator"
MARKERS = ['^', 'o', 's', 'X']
MARKER_SIZE = 9
LINE_COLORS = ['b', 'r', 'g', 'c', 'k', 'm']


def plot_compare_precision(x, ys, legend, xlabel=r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", ylabel=YLABEL, app=""):
    fig, ax = plt.subplots()
    for i in range(len(ys)):
        y = ys[i]
        ax.plot(x, y, marker=MARKERS[i], markersize=MARKER_SIZE, linestyle='dashed', linewidth=1, label=legend[i], color=LINE_COLORS[i])
    ax.legend(loc='lower right', fontsize=LEGEND_FONTSIZE)
    # ax.set_xticklabels(['0','-0.2', r'$C_\mathcal{D}(\pi)$', '+0.2', '+0.4', '+0.6'])

    plt.xlabel(xlabel, fontsize=XLABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=YLABEL_FONTSIZE)
    plt.xticks(fontsize=XTICKS_FONTSIZE)
    plt.yticks(fontsize=YTICKS_FONTSIZE)
    plot_setup()
    plt.savefig('obsavoid/plots/swingsim_compare' + app + '.pdf')


def plot_wind(ys, legend):
    fig, ax = plt.subplots()
    x = [0, 25, 50, 75, 100]
    ax.plot(x, [0.95 for _ in x], color='black', label='OOD threshold')
    ax.plot(x, ys[0], marker=MARKERS[0],markersize=MARKER_SIZE, linestyle='dashed', linewidth=1, color=LINE_COLORS[0], label=legend[0])
    ax.plot(x, np.array(ys[1])+0.95, marker=MARKERS[1],markersize=MARKER_SIZE, linestyle='dashed', linewidth=1, color=LINE_COLORS[1], label=legend[1])
    ax.legend(loc='lower right', fontsize=LEGEND_FONTSIZE)
    plt.xlabel("Percentage of max wind disturbance", fontsize=XLABEL_FONTSIZE)
    plt.ylabel(YLABEL, fontsize=YLABEL_FONTSIZE)
    plt.xticks(ticks=x, fontsize=XTICKS_FONTSIZE)
    plt.yticks(fontsize=YTICKS_FONTSIZE)
    plot_setup()
    plt.savefig('obsavoid/plots/swinghardware_wind.pdf')


def plot_cardinality(cdpmcd, y):
    fig, ax = plt.subplots()
    line_color = 'red'
    fill_color = 'lightcoral'
    mean = np.mean(y, -1)
    max = np.max(y, -1)
    percentile = np.percentile(y, 90, -1)
    # std = np.std(y, -1)

    x = [i for i in range(1, len(mean) + 1)]
    c = [cdpmcd for _ in x]
    ax.plot(x, c, label=r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", color='black')

    ax.plot(x, mean, label=r'Mean $\Delta C$', color=line_color)
    ax.fill_between(x, max, mean, alpha=0.4, label='100th percentile', color=fill_color)
    ax.fill_between(x, percentile, mean, alpha=0.7, label='90th percentile', color=fill_color)
    ax.legend(loc='lower right', fontsize=LEGEND_FONTSIZE)
    plt.xlabel("Cardinality of Test Data", fontsize=XLABEL_FONTSIZE)
    plt.ylabel(YLABEL, fontsize=YLABEL_FONTSIZE)
    plt.xticks(fontsize=XTICKS_FONTSIZE)
    plt.yticks(fontsize=YTICKS_FONTSIZE)
    plot_setup()
    plt.savefig('obsavoid/plots/swingsim_lowerbound.pdf')

def plot_setup():
    plt.tick_params(top=False, right=False, direction='out')
    plt.tight_layout()
    plt.minorticks_off()