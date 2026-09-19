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


# ---------------- Parameters ----------------

log_name = "BPI_Challenge_2012"
noise_thresholds = [0.8, 0.5, 0.2]
repetitions = 10

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

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OPT_ALIGNMENTS_DIR = BASE_DIR / "opt_alignments"
OUTPUT_DIR = BASE_DIR / "output"

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


def trace_to_tuple(trace, activity_key="concept:name"):
    return tuple(event[activity_key] for event in trace if activity_key in event)


def log_to_variant_set(log):
    return {trace_to_tuple(trace) for trace in log}


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

    playout_set = log_to_variant_set(log)
    matches = len(playout_set & opt_variants)

    matches_to_opt_variants = matches / len(opt_variants) if opt_variants else 0
    matches_to_playout_size = matches / len(playout_set) if playout_set else 0

    return [
        noise,
        repetition,
        config["mode"],
        config["beam_width"],
        config["groups"],
        config["max_div_ngram"],
        config["target"],
        len(playout_set),
        len(opt_traces),
        len(opt_variants),
        matches,
        matches_to_opt_variants,
        matches_to_playout_size,
        runtime,
    ]


# One repetition

def run_repetition(noise, repetition, model_path, opt_path, max_trace_length):
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
            )
        )

    print(
        f"[PID {process_id}] Finished repetition {repetition} "
        f"for noise={noise} with {len(rows)} result rows"
    )

    return rows


def print_result_row(row):
    (
        _,
        repetition,
        mode,
        beam_width,
        groups,
        max_div_ngram,
        target,
        _,
        _,
        _,
        matches,
        matches_to_opt_variants,
        matches_to_playout_size,
        _,
    ) = row

    print(
        f"{mode} | target={target} "
        f"| bw={beam_width} | groups={groups} "
        f"| div={max_div_ngram} | rep={repetition} "
        f"| matches={matches} | matches_to_opt_variants={matches_to_opt_variants:.4f} "
        f"| matches_to_playout_size={matches_to_playout_size:.4f}"
    )


# ---------------- Main ----------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

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
        _, shortest = shortest_visible_trace(net, im, fm)
        max_trace_length = max_len + (shortest or 0)

        print(f"Max trace length: {max_trace_length}")

        # These objects are no longer needed in the main process. Workers
        # load one private model copy per repetition. Releasing them here
        # also avoids needlessly inheriting the main-process model on
        # platforms that create workers via fork.
        del net, im, fm, opt_traces, opt_variants

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
                )
                for repetition in range(repetitions)
            ]

            # Read results in repetition order. The computations themselves
            # still run in parallel; this only keeps CSV output grouped as
            # repetition 0, 1, 2, ...
            for repetition, future in enumerate(futures):
                rows = future.result()

                with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(rows)

                for row in rows:
                    print_result_row(row)

                print(
                    f"Results for repetition {repetition}, noise={noise} "
                    f"written to CSV"
                )

    print(f"\nFinished. Results written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
