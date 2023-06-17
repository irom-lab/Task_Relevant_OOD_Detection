#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

XLABEL_FONTSIZE = 24
YLABEL_FONTSIZE = 24
XTICKS_FONTSIZE = 15
YTICKS_FONTSIZE = 15
TITLE_FONTSIZE = 25
YLABEL = "OOD-adverse indicator"


def plot_compare_methods(x, ys, legend, output_file, title="", xlabel=r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", ylabel=YLABEL, app=""):
    fig, ax = plt.subplots()
    line_colors = ['b', 'r']
    fill_colors = ['lightblue', 'lightcoral']
    ax.plot(x, [0.95 for _ in x], color='black')

    for i in range(len(ys)):
        y = ys[i]
        meansi = np.mean(y, axis=0)
        sdi = np.std(y, axis=0)
        if i==0:
            ax.plot(x, meansi, marker='^', markersize = 12, linestyle='dashed', label=legend[i], color=line_colors[i])
        else:
            ax.plot(x, meansi, marker='o', markersize = 12, linestyle='dashed', label=legend[i], color=line_colors[i])
        ax.fill_between(x, meansi-sdi, meansi+sdi, alpha=0.5, label='_nolegend_', color=fill_colors[i])
    plt.legend(('$OOD_A$ threshold', *legend),loc='lower right', fontsize=18)
    plt.xlabel(xlabel, fontsize=XLABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=YLABEL_FONTSIZE)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    plt.xticks(fontsize=XTICKS_FONTSIZE)
    plt.yticks(fontsize=YTICKS_FONTSIZE)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def plot_object_type(x, ys_p, ys_con, legend, output_file_name, title=""):
    fig, ax = plt.subplots()
    plt.ylim([-2.5,1.55])
    thrs = ax.plot(x, [0.95 for _ in x], color='black')

    mug_p = ax.plot(x, ys_p[0], marker='^', markersize = 10, linestyle='dashed', linewidth=1.5, color='b')
    mug_CI = ax.plot(x, np.array(ys_con[0])+0.95, marker='o', markersize = 10, linestyle='dashed', linewidth=1.5, color='b')
    bowl_p = ax.plot(x, ys_p[1], marker='s', markersize = 10, linestyle='dashed', linewidth=1.5, color='r')
    bowl_CI = ax.plot(x, np.array(ys_con[1])+0.95, marker='X', markersize = 10, linestyle='dashed', linewidth=1.5, color='r')

    ax.legend(["$OOD_A$ threshold", *legend], loc="lower right", fontsize=18)#, frameon=False)
    plt.xlabel("Cardinality of Test Data", fontsize=XLABEL_FONTSIZE)
    plt.ylabel(YLABEL, fontsize=YLABEL_FONTSIZE)
    plt.xticks(ticks=x, fontsize=XTICKS_FONTSIZE)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    plt.yticks(fontsize=YTICKS_FONTSIZE)
    plt.tight_layout()
    plt.savefig(output_file_name)
    plt.show()