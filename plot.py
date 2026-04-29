import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# set some parameters for plt 

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 8
plt.rcParams["font.weight"] = "normal"


def plot(col_one, col_two, col_one_title, col_two_title, xlabel, ylabel, plotlabel, show_legend):

    plt.figure(dpi=500)
    ax = plt.gca()
    ax.set_facecolor("#fff4f4")

    plt.plot(col_one, color="#C45714", linestyle= '--', linewidth=2, label=str(col_one_title))
    plt.plot(col_two, color="#D5A5D5", linewidth=1.5, label=str(col_two_title))

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.xlabel(xlabel, fontsize=8, fontweight='normal', fontname='serif')
    plt.ylabel(ylabel, fontsize=8, fontweight='normal', fontname='serif')
    plt.title(plotlabel, fontsize=8, fontweight='normal', fontname='serif')
    if show_legend: 
        plt.legend()
    plt.savefig(plotlabel+'.png')