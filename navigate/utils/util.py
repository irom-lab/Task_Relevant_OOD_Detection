import matplotlib.pyplot as plt
import numpy as np

XLABEL_FONTSIZE = 16
YLABEL_FONTSIZE = 16
XTICKS_FONTSIZE = 14
YTICKS_FONTSIZE = 14
LEGEND_FONTSIZE = 12
YLABEL = "OOD-adverse indicator"
MARKERS = ['^', 'o', 's', 'X', '.', 'o', '.', 'x']
MARKER_SIZE = 6
LINE_COLORS = ['b', 'r', 'g', 'c', 'k', 'm', 'y', 'gray', 'indigo']


def plot_compare_precision(x, ys, legend, xlabel=r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", ylabel=YLABEL, app="", loc = 'lower right'):
    fig, ax = plt.subplots()
    for i in range(len(ys)):
        y = ys[i]
        # only one x array for all y values
        if isinstance(x[0], np.floating): 
            new_x, new_y = zip(*sorted(zip(x, y)))
        # one x array for every y array
        else:
            x_cur = x[i]
            new_x, new_y = zip(*sorted(zip(x_cur, y)))
        ax.plot(new_x, new_y, marker=MARKERS[i], markersize=MARKER_SIZE, linestyle='dashed', linewidth=1, label=legend[i], color=LINE_COLORS[i])
    ax.legend(loc=loc, fontsize=LEGEND_FONTSIZE)
    # ax.set_xticklabels(['0','-0.2', r'$C_\mathcal{D}(\pi)$', '+0.2', '+0.4', '+0.6'])

    plt.xlabel(xlabel, fontsize=XLABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=YLABEL_FONTSIZE)
    plt.xticks(fontsize=XTICKS_FONTSIZE)
    plt.yticks(fontsize=YTICKS_FONTSIZE)
    plot_setup()
    plt.savefig('plots/swingsim_compare' + app + '.png')

def plot_wind(ys, legend):
    fig, ax = plt.subplots()
    x = [0, 25, 50, 75, 100]
    ax.plot(x, [0.95 for _ in x], color='black', label='$OOD_A$ threshold')
    ax.plot(x, ys[0], marker=MARKERS[0],markersize=MARKER_SIZE+3, linestyle='dashed', linewidth=1, color=LINE_COLORS[0], label=legend[0])
    ax.plot(x, np.array(ys[1])+0.95, marker=MARKERS[1],markersize=MARKER_SIZE+3, linestyle='dashed', linewidth=1, color=LINE_COLORS[1], label=legend[1])
    # ax.plot(x, ys+0.95, marker=MARKERS[1],markersize=MARKER_SIZE, linestyle='dashed', linewidth=1, color=LINE_COLORS[1], label=legend)
    ax.legend(loc='lower right', fontsize=LEGEND_FONTSIZE)
    plt.xlabel("Percentage of max wind disturbance", fontsize=XLABEL_FONTSIZE)
    plt.ylabel(YLABEL, fontsize=YLABEL_FONTSIZE)
    plt.xticks(fontsize=XTICKS_FONTSIZE)
    plt.yticks(fontsize=YTICKS_FONTSIZE)
    plot_setup()
    plt.savefig('plots/swinghardware_wind.png')

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
    plt.savefig('plots/swingsim_lowerbound.png')

def plot_combined_detector(x, ys, legend, xlabel= r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", ylabel=YLABEL, app="_combined_detector_CI", loc = 'upper left', figtext=""): 
    fig, ax = plt.subplots() 
    
    # ax.bar(x, ys[0], color = 'r', label = legend[0], width = 0.01)
    # ax.bar(x, ys[1], bottom = ys[0], color = 'g', label = legend[1], width = 0.01)
    # ax.bar(x, ys[2], bottom = np.array(ys[0])+np.array(ys[1]), color = 'y', label = legend[2], width = 0.01)

    ax.bar(x, ys[0], color = 'none', edgecolor = 'r', hatch = '////', label = legend[0], width = 0.01)
    ax.bar(x, ys[1], bottom = ys[0], color = 'none', edgecolor = 'g', hatch = '\\\\\\\\', label = legend[1], width = 0.01)
    ax.bar(x, ys[2], bottom = np.array(ys[0])+np.array(ys[1]), color = 'none', edgecolor = 'y', hatch = '----', label = legend[2], width = 0.01)
    
    ax.legend(loc = loc, fontsize = LEGEND_FONTSIZE, ncol = 1)
    # plt.figtext(.725,.8,figtext, fontsize = 12, bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round,pad=0.4'))
    plt.xlabel(xlabel, fontsize=XLABEL_FONTSIZE-6)
    plt.ylabel(ylabel, fontsize=YLABEL_FONTSIZE-6)
    plt.xticks(fontsize=XTICKS_FONTSIZE)
    plt.yticks(fontsize=YTICKS_FONTSIZE)
    
    plt.ylim(0,1.4)
    plot_setup()

    plt.savefig('plots/swingsim_' + app + '.png')

def plot_setup():
    plt.tick_params(top=False, right=False, direction='out')
    plt.tight_layout()
    plt.minorticks_off()
    