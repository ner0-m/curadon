from types import FunctionType
from pathlib import Path
import pathlib
import argparse
import math
import time
import sys
import csv
import numpy as np
from math import floor, log10, inf
import torch

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.box import Box, HEAVY_HEAD, MARKDOWN
from rich.live import Live


def format_change_percentage(change, speedup=None, flip=False):
    if flip:
        change *= -1
    if change < 0:
        if change > 2:
            col = "yellow1"
        elif change > 10:
            col = "medium_spring_green"
        elif 10 >= change < 20:
            col = "spring_green1"
        elif 20 >= change < 40:
            col = "green1"
        else:
            col = "green3"
        if flip:
            change *= -1

        if speedup:
            return Text(f"{change:>+6.1f}%, {speedup:>+5.1f}x", style=col)
        else:
            return Text(f"{change:>+6.1f}%", style=col)

    # undo flip, but suuuper ugly
    if flip:
        change *= -1
    if speedup:
        return Text(f"{change:>+6.1f}%, {speedup:>+5.1f}x", style="red")
    else:
        return Text(f"{change:>+6.1f}%", style="red")


def benchmark_function(func: FunctionType, repeat: int = 50, warmup: int = 3, framework=None, **kwargs):
    for _ in range(warmup):
        func(**kwargs)

    times = []
    for _ in range(repeat):
        start_time = time.time()

        res = func(**kwargs)
        torch.cuda.synchronize()

        end_time = time.time()
        times.append(end_time - start_time)

    # if framework == "curadon":
    #     import matplotlib.pyplot as plt
    #     plt.imshow(res.cpu().numpy(), cmap="gray")
    #     plt.show()

    return {
        "mean": np.mean(times),
        "min": np.min(times),
        "max": np.max(times),
        "stddev": np.std(times),
    }


def count_frameworks(d: dict):
    for desc in d.keys():
        if not d[desc]:
            continue
        n = list(d[desc].keys())[0]
        frameworks = d[desc][n].keys()
        return len(frameworks), list(frameworks)


def header_str(framework, baseline=None):
    if framework == "curadon":
        if baseline:
            fstr = f"{framework} (± vs baseline)"
        else:
            fstr = f"{framework}"
    else:
        fstr = f"{framework} (± vs curadon)"
    return fstr


def print_table(results, title, markdown=True, baseline=None):
    box = MARKDOWN if markdown else HEAVY_HEAD
    table = Table(title=title, box=box)

    table.add_column("Benchmark (mean op / s)", justify="right",
                     style="cyan", no_wrap=True)

    n_frameworks, frameworks = count_frameworks(results)

    if "curadon" in frameworks:
        table.add_column(header_str("curadon", baseline))
    if "astra" in frameworks:
        table.add_column(header_str("astra"))
    if "torch-radon" in frameworks:
        table.add_column(header_str("torch-radon"))
    if "tigre" in frameworks:
        table.add_column(header_str("tigre"))

    for desc, v in results.items():
        for n, benchmarks in v.items():
            row = [f"{desc} {n:>4}"]

            # First print curadon (if there)
            curadon_mean = None
            if "curadon" in benchmarks:
                tmp = benchmarks["curadon"]
                curadon_mean = tmp["mean"]
                change = None

                if baseline and row[0] in baseline:
                    base_mean = baseline[row[0]]["benchmark"]["mean"]
                    change_perc = (curadon_mean - base_mean) / base_mean * 100
                    change_speedup = math.copysign(
                        curadon_mean / base_mean, change_perc)
                    change = format_change_percentage(
                        change_perc, speedup=change_speedup, flip=True)
                elif baseline:
                    change = "---".center(7)

                row_content = Text()
                row_content.append(f"{curadon_mean:>12.2f}")
                if change:
                    row_content.append(" (")
                    row_content.append(change)
                    row_content.append(")")
                row.append(row_content)

            def format_other_framework(framework, data):
                tmp = benchmarks[framework]
                mean = tmp["mean"]

                change = None
                if curadon_mean:
                    change_perc = (mean - curadon_mean) / curadon_mean * 100
                    change_speedup = - \
                        math.copysign(curadon_mean / mean, change_perc)
                    change = format_change_percentage(
                        change_perc, speedup=change_speedup)

                row_content = Text()
                row_content.append(f"{mean:>12.2f}")
                if change:
                    row_content.append("(")
                    row_content.append(change)
                    row_content.append(")")
                row.append(row_content)

            if "astra" in benchmarks:
                format_other_framework("astra", benchmarks["astra"])
            if "torch-radon" in benchmarks:
                format_other_framework(
                    "torch-radon", benchmarks["torch-radon"])
            if "tigre" in benchmarks:
                format_other_framework("tigre", benchmarks["tigre"])

            table.add_row(*row)

    console = Console(width=200)
    console.print(table)


def write_csv_curadon(output, results):
    if results:
        with output.open("w", encoding="utf-8") as out:
            # description;mean;min;max;stddev;vol_shape;vol_spacing;vol_offset;det_shape;det_spacing;det_offset;det_rotation;arc;nangles;DSO;DSD;COR
            fieldnames = ["framework", "description",
                          "mean", "min", "max", "stddev"]

            # ufff ugly...
            some_desc = list(results.keys())[0]
            some_n = list(results[some_desc].keys())[0]
            some_framework = list(results[some_desc][some_n].keys())[0]
            some_config = results[some_desc][some_n][some_framework]["config"]
            fieldnames += list(some_config.keys())

            writer = csv.DictWriter(out, delimiter=';', fieldnames=fieldnames)
            writer.writeheader()

            for desc, v in results.items():
                for n, benchmarks in v.items():
                    for framework, run in benchmarks.items():
                        if framework != "curadon":
                            continue
                        row = {
                            "framework": framework,
                            "description": f"{desc} {n:>4}",
                            "mean": run["mean"],
                            "min": run["min"],
                            "max": run["max"],
                            "stddev": run["stddev"],
                        }

                        for k, v in run["config"].items():
                            row[k] = v
                        writer.writerow(row)


def read_csv_baseline(path):
    if not path:
        return None

    baseline = {}
    with path.open("r", encoding="utf-8") as b:
        reader = csv.DictReader(b, delimiter=';')
        for line in reader:
            if line['framework'] != "curadon":
                continue
            # framework;description;mean;min;max;stddev;vol_shape;vol_spacing;vol_offset;det_shape;det_spacing;det_offset;det_rotation;arc;nangles;DSO;DSD;COR
            baseline[line['description']] = {
                "framework": line['framework'],
                "benchmark": {
                    # TODO: Currently this is still stored in seconds...
                    # "mean": 1. / float(line['mean']),
                    # "min": 1. / float(line['min']),
                    # "max": 1. / float(line['max']),
                    # "stddev": 1. / float(line['stddev']),
                    "mean": float(line['mean']),
                    "min": float(line['min']),
                    "max": float(line['max']),
                    "stddev": float(line['stddev']),
                },
                "config": {
                    "vol_shape": line['vol_shape'],
                    "vol_spacing": line['vol_spacing'],
                    "vol_offset": line['vol_offset'],
                    "det_shape": line['det_shape'],
                    "det_offset": line['det_offset'],
                    "det_rotation": line['det_rotation'],
                    "arc": line['arc'],
                    "nangles": line['nangles'],
                    "DSO": line['DSO'],
                    "DSD": line['DSD'],
                    "COR": line['COR'],
                }
            }
    return baseline


def main(target, repeat=50, warmup=3, output=None, baseline_path=None, filter_benchmark=None, filter_benchmark_desc=None, markdown=True,
         rmin=7, rmax=10, launch_config=None
         ):

    bench_dir = pathlib.Path(target)

    baseline = read_csv_baseline(baseline_path)

    benchmark_results = {}
    for file in reversed(list(bench_dir.glob("bench_*.py"))):
        if filter_benchmark and filter_benchmark not in str(file):
            continue

        # Okay we import that file, and check it's attributes
        sys.path.append(str(bench_dir.absolute()))
        i = __import__(file.stem, globals(), locals(), level=0)
        if hasattr(i, "__benchmarks__"):
            for benchmark in i.__benchmarks__:
                fn, params, desc, framework, config = benchmark

                if desc not in benchmark_results:
                    benchmark_results[desc] = {}

                if filter_benchmark_desc and filter_benchmark_desc not in desc:
                    continue

                # TODO pass as argument
                for i in range(rmin, rmax):
                    n = 2**i

                    run_params = params(n)
                    if framework == "curadon" and launch_config:
                        run_params["launch_config"] = launch_config

                    result = benchmark_function(
                        fn, repeat=repeat, warmup=warmup, framework=framework, **run_params)
                    del run_params

                    res_min = 1. / result["min"]
                    res_max = 1. / result["max"]
                    res_mean = 1. / result["mean"]
                    res_stddev = result["stddev"]

                    if n not in benchmark_results[desc]:
                        benchmark_results[desc][n] = {}

                    benchmark_results[desc][n][framework] = {
                        "mean": res_mean,
                        "min": res_max,  # swap due to division
                        "max": res_min,
                        "stddev": res_stddev,
                        "config": config(n),
                    }

    title = f"Benchmarks, repeat={repeat}, warumup={warmup}"
    print_table(benchmark_results, title, markdown, baseline=baseline)

    if output:
        write_csv_curadon(output, benchmark_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark a function")
    parser.add_argument("target", type=Path, help="")
    parser.add_argument("--repeat", type=int, default=100,
                        help="Number of times to repeat")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup runs")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--markdown", default=False, action='store_true')
    parser.add_argument("--rmin", type=int, default=7)
    parser.add_argument("--rmax", type=int, default=10)
    parser.add_argument("--filter", nargs='?', default=None,
                        help="Filter by benchmark")
    parser.add_argument("--filter-desc", nargs='?',
                        default=None, help="Filter by benchmark description")
    parser.add_argument("--launch_config", nargs="+",
                        type=int, help="Launch config for curadon")

    args = parser.parse_args()

    launch_config = tuple(args.launch_config) if args.launch_config else None
    main(args.target, repeat=args.repeat, warmup=args.warmup,
         output=args.output, baseline_path=args.baseline, filter_benchmark=args.filter, filter_benchmark_desc=args.filter_desc, markdown=args.markdown,
         rmin=args.rmin, rmax=args.rmax, launch_config=launch_config
         )
