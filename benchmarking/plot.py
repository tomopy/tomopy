"""Functions for creating a plot of image quality vs reconstruction time."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re


def image_quality_vs_time_plot(
    plot_name, json_filename, algo_folder_dir,
):
    """Create a lineplot with errorbars of image quality vs time.

    The vertical axis is the MS-SSIM metric from 0 to 1 and the horizontal axis
    is the reconstruction wall time for each algorithm in seconds. The plot is
    saved to the disk.

    Parameters
    ----------
    plot_name : file path
        The output path and filename including the file extension.
    json_filename : file path
        The name of the timemory json to scrape for wall time data.
    algo_folder_dir: folder path
        The folder to look in for each of the directories named after each
        algorithm. There should be one folder for each algorithm named in the
        timemory JSON. Each of these folders contains a series of numbered
        npz files.

    """
    results = scrape_algorithm_times(json_filename)

    for algo in results.keys():
        algo_folder = os.path.join(algo_folder_dir, algo)
        results[algo].update(scrape_image_quality(algo_folder))

    plt.figure(dpi=600)

    for algo in results.keys():
        # Normalize the iterations to the range [0, total_wall_time]
        time_steps = (
            np.array(results[algo]["num_iter"])
            / results[algo]["num_iter"][-1]
            * results[algo]["wall time"]
        )
        plt.errorbar(
            x=time_steps,
            y=results[algo]["quality"],
            yerr=results[algo]["error"],
            fmt='-o'
        )

    plt.ylim([0, 1])

    plt.legend(results.keys())
    plt.xlabel('time [s]')
    plt.ylabel('MS-SSIM Index')

    plt.savefig(plot_name, dpi=600, pad_inches=0.0)


def scrape_image_quality(algo_folder):
    """Scrape the quality std error and iteration numbers from the files.

    {algo_folder} is a folder containing files {num_iter}.npz with keyword
    "mssim" pointing to an array of quality values.

    Return a dictionary for the folder containing three lists for the
    concatenated image quality at each iteration, std error at each iteration,
    and the number of each iteration. The length of the lists is the number of
    files in {algo_folder}.
    """
    quality = list()
    error = list()
    num_iter = list()

    for file in glob.glob(os.path.join(algo_folder, "*.npz")):
        data = np.load(file)
        # get the iteration number from the filename sans extension
        num_iter.append(int(os.path.basename(file).split(".")[0]))
        quality.append(np.mean(data['msssim']))
        error.append(np.std(data['msssim']))

    return {
        "quality": quality,
        "error": error,
        "num_iter": num_iter,
    }


def scrape_algorithm_times(json_filename):
    """Scrape wall times from the timemory json.

    Search for timer tags containing "algorithm='{algorithm}'" then extract
    the wall times. Return a new dictionary of dictionaries
    where the first key is the algorihm name and the second key is "wall time".
    """
    with open(json_filename, "r") as file:
        data = json.load(file)

    results = {}

    for timer in data["ranks"][0]["manager"]["timers"]:
        # only choose timer with "algorithm in the tag
        if "algorithm" in timer["timer.tag"]:
            # find the part that contains the algorithm name
            m = re.search("'.*'", timer["timer.tag"])
            # strip off the single quotes
            clean_tag = m.group(0).strip("'")
            # convert microseconds to seconds
            wtime = timer["timer.ref"]["wall_elapsed"] * 1e-6

            print("{tag:>10} had a wall time of {wt:10.3g} s".format(
                tag=clean_tag,
                wt=wtime,
            ))

            results[clean_tag] = {
                "wall time": wtime,
            }

    return results
