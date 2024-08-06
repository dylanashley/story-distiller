#!/usr/bin/env python
# -*- coding: ascii -*-

import argparse
import functools
import json
import logging
import multiprocessing
import numpy as np
import os
import scipy.interpolate
import sys


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["fma", "mad"],
        default="fma",
        help="dataset to extract templates from",
        required=True,
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="use pca value rather than narrative essence; " "requires fma dataset",
    )
    parser.add_argument(
        "--num-templates",
        type=int,
        default=4,
        help="number of templates",
        required=True,
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=120,
        help="number of optimization iterations",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=12,
        help="size of each generation",
    )
    parser.add_argument(
        "--fertility",
        type=int,
        default=60,
        help="number of offspring made by each generation",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help=".npy file to write resulting templates to",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed to control for stochasticity",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="don't plot the final narrative arc template curves",
    )
    args = vars(parser.parse_args(args))
    if not args["no_plot"]:
        global plt, sns
        import matplotlib.pyplot as plt
        import seaborn as sns
    if args["pca"]:
        assert args["dataset"] == "fma"
    assert args["num_templates"] > 0
    assert args["num_generations"] > 0
    assert args["population_size"] > 0
    if args["outfile"] is not None and os.path.exists(args["outfile"]):
        logging.warning("{} exists; overwriting".format(args["outfile"]))
    return args


def scale(
    value: float,
    start_min: float,
    start_max: float,
    end_min: float,
    end_max: float,
) -> float:
    """Returns the result of scaling value from the range
    [start_min, start_max] to [end_min, end_max].
    """
    return end_min + (end_max - end_min) * (value - start_min) / (start_max - start_min)


def load_fma_data(pca: bool = False):
    filename = "data/fma_albums.xz"
    with lzma.open(filename) as infile:
        raw_data = pickle.load(infile)
    logging.info("successfully parsed {}".format(filename))
    training, validation, test = list(), list(), list()
    stats = {"min_track_number": float("inf"), "max_track_number": 0}
    while raw_data:
        raw_album = raw_data.pop()
        min_album_track, max_album_track = float("inf"), 0
        album = list()
        for raw_track in raw_album:
            album.append(
                (
                    raw_track["track number"],
                    raw_track["pca"] if pca else raw_track["learned scalar feature"],
                )
            )
            min_album_track = min(min_album_track, raw_track["track number"])
            max_album_track = max(max_album_track, raw_track["track number"])
        for i, (track_number, value) in enumerate(album):
            album[i] = (
                scale(
                    track_number,
                    min(1, min_album_track),
                    max_album_track,
                    0,
                    1,
                ),
                (value - np.mean([v[1] for v in album]))
                / np.std([v[1] for v in album]),
            )
        if raw_album[0]["set split"] == "training":
            training.append(album)
        elif raw_album[0]["set split"] == "validation":
            validation.append(album)
        else:
            assert raw_album[0]["set split"] == "test"
            test.append(album)
        stats["min_track_number"] = min(stats["min_track_number"], min_album_track)
        stats["max_track_number"] = max(stats["max_track_number"], max_album_track)
    return (training, validation, test), stats


def load_mad_data():
    training_filename = "data/mad_shot_narrative_essence_training.json"
    validation_filename = "data/mad_shot_narrative_essence_validation.json"
    test_filename = "data/mad_shot_narrative_essence_test.json"
    raw_data = dict()
    with open(training_filename, "r") as infile:
        raw_data["training"] = json.load(infile)
    logging.info("successfully parsed {}".format(training_filename))
    with open(validation_filename, "r") as infile:
        raw_data["validation"] = json.load(infile)
    logging.info("successfully parsed {}".format(validation_filename))
    with open(test_filename, "r") as infile:
        raw_data["test"] = json.load(infile)
    logging.info("successfully parsed {}".format(test_filename))

    training, validation, test = list(), list(), list()
    stats = {"min_frame_count": float("inf"), "max_frame_count": 0}
    for split_raw_data, split_data in [
        (raw_data["training"], training),
        (raw_data["validation"], validation),
        (raw_data["test"], test),
    ]:
        for raw_entry in split_raw_data:
            entry = list()
            narrative_essence_mean = np.mean(raw_entry["narrative_essence"])
            narrative_essence_std = np.std(raw_entry["narrative_essence"])
            min_frame_count = min(raw_entry["indices"])
            max_frame_count = max(raw_entry["indices"])
            assert len(raw_entry["indices"]) == len(raw_entry["narrative_essence"])
            for i in range(len(raw_entry["narrative_essence"])):
                entry.append(
                    (
                        scale(
                            raw_entry["indices"][i],
                            min_frame_count,
                            max_frame_count,
                            0,
                            1,
                        ),
                        (raw_entry["narrative_essence"][i] - narrative_essence_mean)
                        / narrative_essence_std,
                    )
                )
            split_data.append(entry)
            stats["min_frame_count"] = min(stats["min_frame_count"], min_frame_count)
            stats["max_frame_count"] = max(stats["max_frame_count"], max_frame_count)
    return (training, validation, test), stats


def init_templates():
    return np.zeros((args["num_templates"], 7))


def expand_templates(tpls):
    return [
        scipy.interpolate.interp1d(
            [0.0, 0.2, 0.3, 0.5, 0.65, 0.8, 1.0],
            tpls[i, :],
            kind="cubic",
        )
        for i in range(tpls.shape[0])
    ]


def fitness(tpls, dataset):
    expanded_tpls = expand_templates(tpls)
    rv = 0
    for entry in dataset:
        best = float("inf")
        for tpl in expanded_tpls:
            error = 0
            for index, value in entry:
                error += np.square(tpl(index) - value)
            error /= len(entry)
            if error < best:
                best = error
        rv += best
    return rv / len(dataset)


def crossover(population):
    children = list()
    for _ in range(args["fertility"]):
        father_idx, mother_idx = np.random.choice(
            len(population), size=2, replace=False
        )
        father, mother = population[father_idx], population[mother_idx]
        child = np.zeros_like(father)
        mask = np.transpose(
            np.tile(np.random.randint(2, size=child.shape[0]), (child.shape[1], 1))
        )
        child += father * mask
        child += mother * (1 - mask)
        children.append(child)
    return children


def mutate(tpls):
    return tpls + np.random.normal(0, np.random.rand(), size=tpls.shape)


def optimize(population, dataset):
    population_size = len(population)
    with multiprocessing.Pool(
        min(
            multiprocessing.cpu_count(),
            args["population_size"] + args["fertility"],
        )
    ) as pool:
        if args["num_generations"] == 1:
            fitnesses = pool.map(
                functools.partial(fitness, dataset=dataset), population
            )
            rv = population[np.argmin(fitnesses)]
        else:
            try:
                for i in range(args["num_generations"] - 1):
                    population += [mutate(tpls) for tpls in crossover(population)]
                    fitnesses = pool.map(
                        functools.partial(fitness, dataset=dataset), population
                    )
                    rv = population[np.argmin(fitnesses)]
                    logging.info(
                        "generation {} has min fitness = {}".format(
                            i + 1, np.min(fitnesses)
                        )
                    )
                    selection = np.argsort(fitnesses)[:population_size]
                    population = [v for i, v in enumerate(population) if i in selection]
            except KeyboardInterrupt:
                pass
    return rv


def plot(tpls):
    sns.set_theme(style="ticks", palette="colorblind")
    x = np.linspace(0, 1, 100)
    plt.cla()
    for tpl in expand_templates(tpls):
        plt.plot(x, [tpl(v) for v in x])
    plt.title("Story Template Curves")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()


def main():
    global args, training, validation, test, stats
    args = parse_args()
    if args["seed"] is not None:
        np.random.seed(args["seed"])
    if args["dataset"] == "fma":
        (training, validation, test), stats = load_fma_data(pca=args["pca"])
    else:
        assert args["dataset"] == "mad"
        (training, validation, test), stats = load_mad_data()
    tpls = optimize(
        [init_templates() for _ in range(args["population_size"])], training
    )
    logging.info(
        "final templates have training fitness = {} and validation fitness = {}".format(
            fitness(tpls, training), fitness(tpls, validation)
        )
    )
    if args["outfile"] is not None:
        np.save(args["outfile"], tpls)
    if not args["no_plot"]:
        plot(tpls)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
