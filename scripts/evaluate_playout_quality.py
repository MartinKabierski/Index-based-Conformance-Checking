import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from itertools import product
from pathlib import Path

import numpy as np

from pm4py.objects.log.obj import Trace, Event
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.importer import importer as pnml_importer

from enhanced_random_playout import beam_playout_unique_variants
from enhanced_random_playout import basic_playout_unique_variants
from diverse_beam import dbs_beam_playout_unique_variants

# local imports
from alignment_params_factory import generate_valid_combinations as params_factory
from alignment_txt_fitness_wrapper import alignment_txt_calculate_fitness
from subprocess_wrapper import xes_to_txt


# ---------------- Parameters ----------------

log_name = "BPI_Challenge_2012"
noise_thresholds = [0.8, 0.2]
repetitions = 8

target_variants_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
playout_modes = ["random", "dbs"]  # "random", "beam", "gumbel", "dbs"

beam_widths = [64]
numbers_of_groups = [16]
max_diversity_ngrams = [1, 2, 3, 4]

weight_activity = 0.0
ngram_weights = {2: 1.0, 3: 1.0, 4: 1.0}
max_traces_per_depth = 4
max_traces_per_round = 2000
lambda_div = 0.5

BASE_SEED = 13

# ---------------- IBF settings ----------------

# One fixed parameter set for all IBF comparisons.
# If left at None, the first combination returned by
# params_factory() is used, which guarantees a valid set.
IBF_PARAMS = None

# The playout TXT files are only needed as input for the IBF run.
# With 10 repetitions and many parameter combinations they add up,
# so they are removed again by default.
KEEP_PLAYOUT_TXT = False

# The IBF tool expects the shortest visible trace as the LAST trace
# of the model trace file, same convention as in the other scripts.
APPEND_SHORTEST_TRACE = True

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
OPT_ALIGNMENTS_DIR = BASE_DIR / "opt_alignments"
OUTPUT_DIR = BASE_DIR / "output"

# Intermediate files: playout TXT files and the per-run IBF output
# CSVs. Only the final result CSV goes into OUTPUT_DIR.
TEMP_DIR = OUTPUT_DIR / "temp"

ORIGINAL_LOG_XES = LOGS_DIR / f"{log_name}.xes"
ORIGINAL_LOG_TXT = OUTPUT_DIR / f"{log_name}.txt"

# Written by calculate_fitness_baseline. Holds one AVG row per noise
# level with the A* total cost and the A* total variant cost.
BASELINE_CSV = OUTPUT_DIR / f"fitness_baseline_all_noise_{log_name}.csv"

OUTPUT_CSV = OUTPUT_DIR / f"playout_vs_opt_alignments_{log_name}.csv"


# ---------------- Helpers ----------------

def noise_to_token(noise):
    return str(noise).replace(".", "p")


def parse_trace_line(line):
    return tuple(activity.strip() for activity in line.strip().split(" - ") if activity.strip())


def load_optimal_alignment_traces(path):
    traces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            trace = parse_trace_line(line)
            if trace:
                traces.append(trace)
    return traces


def count_traces_in_txt(path):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if parse_trace_line(line):
                count += 1
    return count


def trace_to_tuple(trace, activity_key="concept:name"):
    return tuple(event[activity_key] for event in trace if activity_key in event)


def log_to_variant_set(log):
    return {trace_to_tuple(trace) for trace in log}


def log_to_ordered_variants(log):
    # Deduplicate while keeping the order in which the playout
    # produced the variants, so the TXT file is deterministic.
    seen = set()
    ordered = []

    for trace in log:
        variant = trace_to_tuple(trace)

        if not variant or variant in seen:
            continue

        seen.add(variant)
        ordered.append(variant)

    return ordered


def write_traces_txt(variants, shortest_trace_tuple, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        for variant in variants:
            f.write(" - ".join(variant) + "\n")

        if APPEND_SHORTEST_TRACE and shortest_trace_tuple:
            f.write(" - ".join(shortest_trace_tuple) + "\n")


def is_silent(t):
    name = (getattr(t, "name", "") or "")
    return (
        t.label is None
        or t.label == ""
        or getattr(t, "invisible", False)
        or name.startswith(("tau", "skip", "tauSplit", "tauJoin"))
    )


def _mkey(marking):
    return tuple(sorted((id(p), v) for p, v in marking.items()))


def shortest_visible_trace(net, initial_marking, final_marking, activity_key="concept:name"):
    import heapq
    from math import inf

    start_k = _mkey(initial_marking)
    goal_k = _mkey(final_marking)

    pq = []
    tie = 0
    heapq.heappush(pq, (0, tie, initial_marking, 0))

    dist = {(start_k, 0): 0}
    prev = {(start_k, 0): (None, None, None)}

    while pq:
        cost, _, marking, seen_visible = heapq.heappop(pq)
        mk = _mkey(marking)

        if cost > dist.get((mk, seen_visible), inf):
            continue

        if mk == goal_k and seen_visible == 1 and cost > 0:
            path = []
            cur = (mk, seen_visible)

            while prev[cur][0] is not None:
                pkey, pvis, t = prev[cur]
                path.append(t)
                cur = (pkey, pvis)

            path.reverse()

            tr = Trace()
            for t in path:
                if not is_silent(t):
                    tr.append(Event({activity_key: t.label}))

            return tr, cost

        for t in semantics.enabled_transitions(net, marking):
            m2 = semantics.execute(t, net, marking)
            step = 0 if is_silent(t) else 1
            new_cost = cost + step
            new_seen = 1 if seen_visible == 1 or step == 1 else 0
            key2 = (_mkey(m2), new_seen)

            if new_cost < dist.get(key2, inf):
                dist[key2] = new_cost
                prev[key2] = (mk, seen_visible, t)
                tie += 1
                heapq.heappush(pq, (new_cost, tie, m2, new_seen))

    return None, None


# ---------------- A* baseline ----------------

def parse_optional_float(value):
    # None for missing or empty cells, e.g. when the baseline CSV was
    # written before the total_variant_cost column existed.
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_astar_total_costs():
    """
    Reads the A* total cost and the A* total variant cost per noise
    level from the baseline CSV.

    Only the AVG rows are used, so the values are the mean over all
    repetitions of the A* run. Returns
    {noise: {"total_cost": ..., "total_variant_cost": ...}}.
    A value is None if its column is missing or empty.
    """
    costs = {}

    if not BASELINE_CSV.exists():
        print(
            f"WARNING: baseline CSV not found: {BASELINE_CSV}\n"
            f"         MAE will be empty for all rows."
        )
        return costs

    with open(BASELINE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("repetition")).strip() != "AVG":
                continue

            noise = parse_optional_float(row.get("noise_threshold"))

            if noise is None:
                continue

            costs[noise] = {
                "total_cost": parse_optional_float(row.get("total_cost")),
                "total_variant_cost": parse_optional_float(
                    row.get("total_variant_cost")
                ),
            }

    if costs:
        print("A* costs per noise level (from baseline CSV):")
        for noise in sorted(costs):
            print(
                f"  noise={noise}: "
                f"total_cost={costs[noise]['total_cost']} "
                f"| total_variant_cost={costs[noise]['total_variant_cost']}"
            )

    return costs


def resolve_ibf_params():
    if IBF_PARAMS is not None:
        return IBF_PARAMS

    first = next(iter(params_factory()), None)

    if first is None:
        raise RuntimeError(
            "params_factory() returned no parameter combination and "
            "IBF_PARAMS is not set."
        )

    return first


def iter_playout_configs():
    #Yields every valid playout parameter combination once.

    for mode in playout_modes:
        if mode == "random":
            for target in target_variants_list:
                yield {
                    "mode": mode,
                    "target": target,
                    "beam_width": 0,
                    "groups": 0,
                    "max_div_ngram": 0,
                }

        elif mode == "dbs":
            for target, beam_width, groups, max_div_ngram in product(
                target_variants_list,
                beam_widths,
                numbers_of_groups,
                max_diversity_ngrams,
            ):
                yield {
                    "mode": mode,
                    "target": target,
                    "beam_width": beam_width,
                    "groups": groups,
                    "max_div_ngram": max_div_ngram,
                }

        else:
            for target, beam_width in product(
                target_variants_list,
                beam_widths,
            ):
                yield {
                    "mode": mode,
                    "target": target,
                    "beam_width": beam_width,
                    "groups": 0,
                    "max_div_ngram": 0,
                }


def config_token(config):
    return (
        f"{config['mode']}"
        f"_t{config['target']}"
        f"_bw{config['beam_width']}"
        f"_g{config['groups']}"
        f"_d{config['max_div_ngram']}"
    )


def run_playout(config, net, im, fm, max_trace_length, rnd_seed):
    mode = config["mode"]

    if mode == "random":
        return basic_playout_unique_variants(
            petri_net=net,
            initial_marking=im,
            final_marking=fm,
            max_trace_length=max_trace_length,
            target_unique=config["target"],
            batch_size=max(1, int(config["target"] / 10)),
            max_rounds=config["target"],
        )

    if mode == "dbs":
        return dbs_beam_playout_unique_variants(
            petri_net=net,
            initial_marking=im,
            final_marking=fm,
            max_trace_length=max_trace_length,
            target_unique=config["target"],
            beam_width=config["beam_width"],
            max_rounds=config["target"],
            rnd_seed=rnd_seed,
            num_groups=config["groups"],
            lambda_div=lambda_div,
            max_traces_per_depth=max_traces_per_depth,
            max_traces_per_round=max_traces_per_round,
            weight_activity=weight_activity,
            ngram_weights=ngram_weights,
            max_diversity_ngram=config["max_div_ngram"],
            csv_path=None,
        )

    return beam_playout_unique_variants(
        petri_net=net,
        initial_marking=im,
        final_marking=fm,
        max_trace_length=max_trace_length,
        target_unique=config["target"],
        beam_width=config["beam_width"],
        selection_mode=mode,
        max_rounds=config["target"],
        rnd_seed=rnd_seed,
        max_traces_per_depth=max_traces_per_depth,
        max_traces_per_round=max_traces_per_round,
        weight_activity=weight_activity,
        ngram_weights=ngram_weights,
        csv_path=None,
    )


def run_ibf_comparison(
    playout_variants,
    shortest_trace_tuple,
    shortest_path,
    log_length,
    ibf_params,
    noise,
    repetition,
    config,
):
    """
    Writes the playout variants as TXT and runs the IBF comparison
    against the original log. Returns the raw wrapper result.
    """
    token = noise_to_token(noise)
    name = f"{log_name}_noise_{token}_rep_{repetition}_{config_token(config)}"

    playout_txt = TEMP_DIR / f"playout_{name}.txt"
    alignment_csv = TEMP_DIR / f"alignment_{name}.csv"

    write_traces_txt(playout_variants, shortest_trace_tuple, playout_txt)

    start = time.time()

    result = alignment_txt_calculate_fitness(
        params=ibf_params,
        log_txt=ORIGINAL_LOG_TXT,
        model_traces_txt=playout_txt,
        output_csv=alignment_csv,
        shortest_path=shortest_path,
        log_length=log_length,
    )

    ibf_runtime = timedelta(seconds=time.time() - start) / timedelta(milliseconds=1)

    if not KEEP_PLAYOUT_TXT:
        try:
            playout_txt.unlink()
        except OSError:
            pass

    return result, ibf_runtime


def run_parameter_combination(
    noise,
    repetition,
    config,
    net,
    im,
    fm,
    max_trace_length,
    opt_traces,
    opt_variants,
    rnd_seed,
    shortest_trace_tuple,
    shortest_path,
    log_length,
    ibf_params,
    astar_total_cost,
    astar_total_variant_cost,
):
    # Every parameter combination inside one repetition starts from the
    # same repetition-specific seed. This makes repetitions independent
    # while keeping parameter combinations paired by repetition.
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)

    start = time.time()

    log = run_playout(
        config=config,
        net=net,
        im=im,
        fm=fm,
        max_trace_length=max_trace_length,
        rnd_seed=rnd_seed,
    )

    runtime = timedelta(seconds=time.time() - start) / timedelta(milliseconds=1)

    ordered_variants = log_to_ordered_variants(log)
    playout_set = set(ordered_variants)
    matches = len(playout_set & opt_variants)

    matches_to_opt_variants = matches / len(opt_variants) if opt_variants else 0
    matches_to_playout_size = matches / len(playout_set) if playout_set else 0

    # IBF comparison: original log against this playout.
    ibf_result, ibf_runtime = run_ibf_comparison(
        playout_variants=ordered_variants,
        shortest_trace_tuple=shortest_trace_tuple,
        shortest_path=shortest_path,
        log_length=log_length,
        ibf_params=ibf_params,
        noise=noise,
        repetition=repetition,
        config=config,
    )

    # Indexed access, so the script also works with wrapper versions
    # that return additional variant-based values after these eight.
    ibf_fitness = ibf_result[0]
    ibf_trace_calculation_time = ibf_result[1]
    ibf_creation_time = ibf_result[2]
    ibf_search_time = ibf_result[3]
    ibf_total_cost = ibf_result[4]
    ibf_total_cost_adjusted = ibf_result[5]
    ibf_total_length_all_traces = ibf_result[6]
    ibf_number_traces = ibf_result[7]
    ibf_total_variant_cost = ibf_result[8]
    #ibf_mean_variant_cost = ibf_result[9]
    ibf_number_trace_variants = ibf_result[10]
    #variant_cost_inconsistencies = ibf_result[11]


    # Mean Alignment Error against the A* baseline.
    # Both MAEs are computed independently, so one missing A* value
    # does not blank out the other MAE.
    # Denominator: number of trace variants of the original log
    variant_count = ibf_number_trace_variants

    if astar_total_variant_cost is None or not variant_count:
        mae_by_variant_count = None
    else:
        mae_by_variant_count = (float(ibf_total_variant_cost) - float(astar_total_variant_cost)) / variant_count

    # Denominator: number of traces of the original log
    if astar_total_cost is None or not ibf_number_traces:
        mae_by_trace_count = None
    else:
        mae_by_trace_count = (float(ibf_total_cost) - float(astar_total_cost)) / ibf_number_traces

    return {
        "noise": noise,
        "repetition": repetition,
        "playout_mode": config["mode"],
        "beam_width": config["beam_width"],
        "groups": config["groups"],
        "max_div_ngram": config["max_div_ngram"],
        "target_variants": config["target"],
        "playout_variants": len(playout_set),
        "opt_alignment_traces": len(opt_traces),
        "opt_alignment_variants": len(opt_variants),
        "matches": matches,
        "matches_to_opt_variants": matches_to_opt_variants,
        "matches_to_playout_size": matches_to_playout_size,
        "time_ms": runtime,

        "ibf_fitness": ibf_fitness,
        "ibf_total_cost": ibf_total_cost,
        "ibf_total_cost_adjusted": ibf_total_cost_adjusted,
        "ibf_total_variant_cost": ibf_total_variant_cost,
        "ibf_total_length_all_traces": ibf_total_length_all_traces,
        "ibf_number_traces": ibf_number_traces,
        "ibf_trace_calculation_time": ibf_trace_calculation_time,
        "ibf_creation_time": ibf_creation_time,
        "ibf_search_time": ibf_search_time,
        "ibf_run_time_ms": ibf_runtime,

        "astar_total_cost": astar_total_cost,
        "astar_total_variant_cost": astar_total_variant_cost,
        "variant_count_for_mae": variant_count,
        "mae_by_variant_count": mae_by_variant_count,
        "mae_by_trace_count": mae_by_trace_count,
    }


# One repetition

def run_repetition(
    noise,
    repetition,
    model_path,
    opt_path,
    max_trace_length,
    shortest_trace_tuple,
    shortest_path,
    log_length,
    ibf_params,
    astar_total_cost,
    astar_total_variant_cost,
):
    """
    Executes one complete repetition in one worker process.

    All playout parameter combinations are processed serially inside this
    repetition. The worker returns all CSV rows to the main process.
    """
    process_id = os.getpid()
    rnd_seed = BASE_SEED + repetition

    print(
        f"[PID {process_id}] Starting repetition {repetition} "
        f"for noise={noise} with seed={rnd_seed}"
    )

    # Each repetition loads its own Petri net once and reuses it for all
    # parameter combinations in that repetition.
    net, im, fm = pnml_importer.apply(str(model_path))

    opt_traces = load_optimal_alignment_traces(opt_path)
    opt_variants = set(opt_traces)

    rows = []

    for config in iter_playout_configs():
        rows.append(
            run_parameter_combination(
                noise=noise,
                repetition=repetition,
                config=config,
                net=net,
                im=im,
                fm=fm,
                max_trace_length=max_trace_length,
                opt_traces=opt_traces,
                opt_variants=opt_variants,
                rnd_seed=rnd_seed,
                shortest_trace_tuple=shortest_trace_tuple,
                shortest_path=shortest_path,
                log_length=log_length,
                ibf_params=ibf_params,
                astar_total_cost=astar_total_cost,
                astar_total_variant_cost=astar_total_variant_cost,
            )
        )

    print(
        f"[PID {process_id}] Finished repetition {repetition} "
        f"for noise={noise} with {len(rows)} result rows"
    )

    return rows


def print_result_row(row):
    mae_variant = row["mae_by_variant_count"]
    mae_trace = row["mae_by_trace_count"]
    mae_variant_text = "n/a" if mae_variant is None else f"{mae_variant:.4f}"
    mae_trace_text = "n/a" if mae_trace is None else f"{mae_trace:.4f}"

    print(
        f"{row['playout_mode']} | target={row['target_variants']} "
        f"| bw={row['beam_width']} | groups={row['groups']} "
        f"| div={row['max_div_ngram']} | rep={row['repetition']} "
        f"| matches={row['matches']} "
        f"| matches_to_opt_variants={row['matches_to_opt_variants']:.4f} "
        f"| matches_to_playout_size={row['matches_to_playout_size']:.4f} "
        f"| ibf_total_cost={row['ibf_total_cost']} "
        f"| mae_by_variant_count={mae_variant_text} "
        f"| mae_by_trace_count={mae_trace_text}"
    )


# ---------------- Main ----------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    header = [
        "noise",
        "repetition",
        "playout_mode",
        "beam_width",
        "groups",
        "max_div_ngram",
        "target_variants",
        "playout_variants",
        "opt_alignment_traces",
        "opt_alignment_variants",
        "matches",
        "matches_to_opt_variants",
        "matches_to_playout_size",
        "time_ms",

        "ibf_fitness",
        "ibf_total_cost",
        "ibf_total_cost_adjusted",
        "ibf_total_variant_cost",
        "ibf_total_length_all_traces",
        "ibf_number_traces",
        "ibf_trace_calculation_time",
        "ibf_creation_time",
        "ibf_search_time",
        "ibf_run_time_ms",

        "astar_total_cost",
        "astar_total_variant_cost",
        "variant_count_for_mae",
        "mae_by_variant_count",
        "mae_by_trace_count",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=header).writeheader()

    if not ORIGINAL_LOG_XES.exists():
        raise FileNotFoundError(
            f"Original log is missing: {ORIGINAL_LOG_XES}"
        )

    # The original log TXT is the IBF input and is created once,
    # before any worker process is started.
    if not ORIGINAL_LOG_TXT.exists():
        print(f"Converting original log to TXT: {ORIGINAL_LOG_TXT}")
        conversion_time = xes_to_txt(
            input=str(ORIGINAL_LOG_XES),
            output=str(ORIGINAL_LOG_TXT),
        )
        print(f"  done in {conversion_time} ms")

    log_length = count_traces_in_txt(ORIGINAL_LOG_TXT)

    if log_length <= 0:
        raise RuntimeError(
            f"Original log TXT contains no valid traces: {ORIGINAL_LOG_TXT}"
        )

    print(f"Original log: {log_length} traces")

    ibf_params = resolve_ibf_params()
    print(f"IBF parameters: {ibf_params}")

    astar_total_costs = load_astar_total_costs()

    max_workers = min(repetitions, os.cpu_count() or 1)

    print(f"Available CPUs: {os.cpu_count() or 1}")
    print(f"Parallel repetition workers: {max_workers}")

    for noise in noise_thresholds:
        token = noise_to_token(noise)

        model_path = MODELS_DIR / f"{log_name}_noise_{token}.pnml"
        opt_path = OPT_ALIGNMENTS_DIR / f"{log_name}_noise_{token}_optimal_model_traces.txt"

        missing = []
        if not model_path.exists():
            missing.append("model")
        if not opt_path.exists():
            missing.append("opt_alignments")

        if missing:
            print(f"Skip noise {noise}: missing {', '.join(missing)}")
            continue

        print(f"\n--- Noise {noise} ---")

        # Model information needed for max_trace_length is calculated once
        # in the main process. Each repetition then loads its own model once.
        net, im, fm = pnml_importer.apply(str(model_path))

        opt_traces = load_optimal_alignment_traces(opt_path)
        opt_variants = set(opt_traces)

        print(
            f"Opt alignment traces: {len(opt_traces)} "
            f"| opt alignment variants: {len(opt_variants)}"
        )

        max_len = max((len(t) for t in opt_variants), default=0)
        shortest_trace, shortest = shortest_visible_trace(net, im, fm)
        max_trace_length = max_len + (shortest or 0)

        shortest_trace_tuple = (
            trace_to_tuple(shortest_trace) if shortest_trace is not None else ()
        )

        if APPEND_SHORTEST_TRACE and not shortest_trace_tuple:
            raise RuntimeError(
                f"No complete visible path found for {model_path}, "
                f"but APPEND_SHORTEST_TRACE is enabled."
            )

        print(f"Max trace length: {max_trace_length}")
        print(f"Shortest visible trace: {' - '.join(shortest_trace_tuple)}")

        astar_costs = astar_total_costs.get(float(noise), {})
        astar_total_cost = astar_costs.get("total_cost")
        astar_total_variant_cost = astar_costs.get("total_variant_cost")

        if astar_total_cost is None:
            print(
                f"WARNING: no A* total cost for noise={noise} in the "
                f"baseline CSV. mae_by_trace_count stays empty for "
                f"this noise level."
            )

        if astar_total_variant_cost is None:
            print(
                f"WARNING: no A* total variant cost for noise={noise} in "
                f"the baseline CSV. mae_by_variant_count stays empty for "
                f"this noise level."
            )

        # These objects are no longer needed in the main process. Workers
        # load one private model copy per repetition. Releasing them here
        # also avoids needlessly inheriting the main-process model on
        # platforms that create workers via fork.
        del net, im, fm, opt_traces, opt_variants, shortest_trace

        print(
            f"Starting {repetitions} repetitions with up to "
            f"{max_workers} parallel processes"
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    run_repetition,
                    noise,
                    repetition,
                    str(model_path),
                    str(opt_path),
                    max_trace_length,
                    shortest_trace_tuple,
                    shortest or 0,
                    log_length,
                    ibf_params,
                    astar_total_cost,
                    astar_total_variant_cost,
                )
                for repetition in range(repetitions)
            ]

            # Read results in repetition order. The computations themselves
            # still run in parallel; this only keeps CSV output grouped as
            # repetition 0, 1, 2, ...
            for repetition, future in enumerate(futures):
                rows = future.result()

                with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=header).writerows(rows)

                for row in rows:
                    print_result_row(row)

                print(
                    f"Results for repetition {repetition}, noise={noise} "
                    f"written to CSV"
                )

    print(f"\nFinished. Results written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
