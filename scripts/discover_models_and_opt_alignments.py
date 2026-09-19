import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm
from pm4py.objects.log import obj
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.petri_net.importer import importer as pnml_importer
from tqdm import tqdm

LOG_NAME = "BPI_Challenge_2019"
NOISE_THRESHOLDS = [0.8, 0.2]
REPETITIONS = 8

# None:
# automatically use as many workers as repetitions / available CPUs
MAX_PARALLEL_REPETITIONS = 4

LOG_DIR = "../logs"
MODEL_DIR = "../models"
OPT_ALIGNMENTS_DIR = "../opt_alignments"
OUTPUT_DIR = "../output"

# Suffix for the filtered outputs (CSV and XES).
FILTERED_SUFFIX = "_without_unsolved"

# -------------------------------------
#  Alignment settings
# VERSION_DIJKSTRA_LESS_MEMORY:
# Memory-saving variant: does not keep the complete
# A* search tree in memory. Computes exactly (Dijkstra without heuristic).
#
# VERSION_STATE_EQUATION_A_STAR:
# pm4py default a* alignment

#ALIGN_VARIANT = algorithm.Variants.VERSION_DIJKSTRA_LESS_MEMORY
ALIGN_VARIANT = algorithm.Variants.VERSION_STATE_EQUATION_A_STAR

# Seconds per trace variant. If the time is exceeded, the
# variant counts as unsolved and the run continues.
ALIGN_TIMEOUT = 60
# ----------------------------------------------


def build_align_parameters():
    # The parameter name differs between PM4Py versions.
    # If it is missing, the script keeps running without a timeout
    # instead of aborting with an AttributeError.
    params = {}

    try:
        key = ALIGN_VARIANT.value.Parameters.PARAM_MAX_ALIGN_TIME_TRACE
        params[key] = ALIGN_TIMEOUT
    except AttributeError:
        print(
            "WARNUNG: PARAM_MAX_ALIGN_TIME_TRACE ist in dieser "
            "PM4Py-Version nicht verfügbar. Es läuft KEIN Timeout."
        )

    return params


ALIGN_PARAMETERS = build_align_parameters()


def safe_noise_name(noise):
    return str(noise).replace(".", "p")


def trace_to_string(trace):
    return " - ".join(event["concept:name"] for event in trace)


def variant_to_trace(variant_string):
    trace = Trace()
    for activity in variant_string.split(" - "):
        event = Event()
        event["concept:name"] = activity
        trace.append(event)
    return trace


def get_alignment_cost(alignment_result):
    return sum(
        1
        for step in alignment_result["alignment"]
        if step[1] == ">>" or (step[0] == ">>" and step[1] is not None)
    )


def extract_optimal_model_trace(alignment_result):
    model_trace = []

    for log_move, model_move in alignment_result["alignment"]:
        if model_move != ">>" and model_move is not None:
            model_trace.append(str(model_move))

    return " - ".join(model_trace)


# worker state / worker function

_WORKER_LOG = None
_WORKER_MODELS = {}


def _init_worker(log_path):
    # Is executed once per worker process.
    global _WORKER_LOG
    _WORKER_LOG = pm4py.read_xes(log_path, return_legacy_log_object=True)


def _get_model(model_path):
    # Load the Petri net once per worker
    if model_path not in _WORKER_MODELS:
        _WORKER_MODELS[model_path] = pnml_importer.apply(model_path)
    return _WORKER_MODELS[model_path]


def run_repetition(repetition, noise, model_path, shortest_path):
    process_id = os.getpid()

    print(
        f"[PID {process_id}] "
        f"Starting repetition {repetition}, "
        f"noise={noise}",
        flush=True
    )

    log = _WORKER_LOG
    net, im, fm = _get_model(model_path)

    variants = {}
    values = []

    unsolved_count = 0

    for idx, trace in enumerate(tqdm(
            log,
            desc=f"noise={noise} rep={repetition}",
            position=repetition,
            leave=False
    )):
        trace_string = trace_to_string(trace)

        if trace_string in variants:
            variants[trace_string]["count"] += 1
            continue

        variants[trace_string] = {
            "count": 1,
            "optimal_model_trace": None
        }

        t_start = time.time()

        try:
            conf = algorithm.apply_trace(
                trace,
                net,
                im,
                fm,
                variant=ALIGN_VARIANT,
                parameters=ALIGN_PARAMETERS
            )
        except Exception as exc:
            conf = None
            print(
                f"[PID {process_id}] Alignment-Fehler: "
                f"{type(exc).__name__}: {exc}",
                flush=True
            )

        calculation_time = time.time() - t_start

        trace_length = len(trace)
        denominator = trace_length + shortest_path

        # Timeout or error: the row is written anyway,
        # so that unsolved variants remain evaluable.
        if conf is None or conf.get("alignment") is None:
            unsolved_count += 1

            values.append({
                "repetition": repetition,
                "trace": trace_string,
                "optimal_model_trace": None,
                "time": calculation_time,
                "cost": None,
                "cost_pm4py": None,
                "cost_pm4py_normalized": None,
                "trace_length": trace_length,
                "shortest_path": shortest_path,
                "fitness": None,
                "fitness_pm4py": None,
                "noise_threshold": noise,
                "status": "unsolved"
            })
            continue

        cost = get_alignment_cost(conf)
        cost_pm4py = conf["cost"]
        cost_pm4py_normalized = cost_pm4py % 10000

        optimal_model_trace = extract_optimal_model_trace(conf)

        variants[trace_string]["optimal_model_trace"] = optimal_model_trace

        values.append({
            "repetition": repetition,
            "trace": trace_string,
            "optimal_model_trace": optimal_model_trace,
            "time": calculation_time,
            "cost": cost,
            "cost_pm4py": cost_pm4py,
            "cost_pm4py_normalized": cost_pm4py_normalized,
            "trace_length": trace_length,
            "shortest_path": shortest_path,
            "fitness": (
                1 - (cost / denominator) if denominator else None
            ),
            "fitness_pm4py": (
                1 - (cost_pm4py_normalized / denominator)
                if denominator else None
            ),
            "noise_threshold": noise,
            "status": "ok"
        })

    for v in values:
        v["count"] = variants[v["trace"]]["count"]

    optimal_model_traces = [
        data["optimal_model_trace"] for data in variants.values()
    ]

    print(
        f"[PID {process_id}] "
        f"Finished repetition {repetition}, "
        f"noise={noise} | "
        f"Varianten: {len(variants)} | "
        f"ungelöst: {unsolved_count}",
        flush=True
    )

    return values, optimal_model_traces


def append_results_csv(values, csv_path):
    # Writes the result rows of a finished repetition.
    # Only the main process calls this function, therefore
    # multiple repetitions never write to the CSV concurrently.

    df = pd.DataFrame(values)

    write_header = not os.path.exists(csv_path)

    df.to_csv(
        csv_path,
        mode="a",
        header=write_header,
        index=False
    )


def write_optimal_model_traces(optimal_model_traces, path):
    # Unsolved variants have no optimal model trace and are skipped.
    missing = sum(1 for t in optimal_model_traces if t is None)

    if missing:
        print(
            f"Hinweis: {missing} ungelöste Varianten werden in "
            f"{os.path.basename(path)} ausgelassen. Passend dazu "
            f"das gefilterte Log ({FILTERED_SUFFIX}.xes) verwenden."
        )

    with open(path, "w", encoding="utf-8") as f:
        for optimal_model_trace in optimal_model_traces:
            if optimal_model_trace is None:
                continue
            f.write(optimal_model_trace + "\n")


# ---------- Post-processing: filter out unsolved variants ----------


def collect_unsolved(df):
    # A variant counts as unsolved as soon as it is unsolved in ANY
    # row, that is across all noise values and repetitions.
    # Only this way the filtered log stays the same basis for
    # all evaluations.
    if "status" in df.columns:
        mask = df["status"] != "ok"
    else:
        mask = df["cost"].isna()

    return set(df.loc[mask, "trace"]), mask


def unique_trace_count(df):
    # Each variant appears once per noise value and repetition in
    # the CSV. For the number of traces in the log, each variant
    # may only be counted once.
    unique = df.drop_duplicates(subset=["trace"])
    return int(unique["count"].sum())


def write_unsolved_variants(df, mask, path):
    unsolved_rows = df.loc[mask].copy()
    unsolved_rows.to_csv(path, index=False)

    print(f"Geschrieben: {path}")
    print(f"  Zeilen:               {len(unsolved_rows)}")
    print(f"  eindeutige Varianten: {unsolved_rows['trace'].nunique()}")

    if len(unsolved_rows):
        print(
            f"  betroffene Traces:    "
            f"{unique_trace_count(unsolved_rows)}"
        )
        print(
            f"  Tracelänge: "
            f"min={int(unsolved_rows['trace_length'].min())} "
            f"median={int(unsolved_rows['trace_length'].median())} "
            f"max={int(unsolved_rows['trace_length'].max())}"
        )

        print("  je noise-Wert:")
        for noise, grp in unsolved_rows.groupby("noise_threshold"):
            print(
                f"    noise={noise}: "
                f"{grp['trace'].nunique()} Varianten"
            )


def write_filtered_csv(df, unsolved_variants, path):
    # ALL rows of these variants are removed, not only the
    # unsolved ones. This way the CSV matches the filtered log exactly.
    keep = df[~df["trace"].isin(unsolved_variants)]
    keep.to_csv(path, index=False)

    print(f"\nGeschrieben: {path}")
    print(f"  Zeilen: {len(keep)} von {len(df)}")
    print(
        f"  Traces: {unique_trace_count(keep)} von "
        f"{unique_trace_count(df)}"
    )

    return keep


def write_filtered_log(log, unsolved_variants, path, expected_traces):
    filtered_log = EventLog(
        attributes=log.attributes,
        extensions=log.extensions,
        classifiers=log.classifiers,
        omni_present=log.omni_present
    )

    removed = 0

    for trace in log:
        if trace_to_string(trace) in unsolved_variants:
            removed += 1
            continue
        filtered_log.append(trace)

    print(f"\nGefiltertes Log:")
    print(f"  entfernt:    {removed}")
    print(f"  verbleibend: {len(filtered_log)}")

    if len(filtered_log) != expected_traces:
        print(
            f"  WARNUNG: erwartet {expected_traces} Traces, "
            f"gefunden {len(filtered_log)}."
        )
    else:
        print(f"  OK: passt zur gefilterten CSV.")

    print(f"  schreibe nach {path} ...")

    t_start = time.time()
    pm4py.write_xes(filtered_log, path)
    print(f"  fertig in {time.time() - t_start:.1f} s")


def finalize_outputs(log, combined_output_csv_path):
    print(f"\n==============================")
    print(f"Nachbereitung")
    print(f"==============================")

    if not os.path.exists(combined_output_csv_path):
        print(f"Ergebnis-CSV fehlt: {combined_output_csv_path}")
        return

    df = pd.read_csv(combined_output_csv_path)

    unsolved_variants, unsolved_mask = collect_unsolved(df)

    print(f"Zeilen gesamt:        {len(df)}")
    print(f"Varianten gesamt:     {df['trace'].nunique()}")
    print(f"Ungelöste Varianten: {len(unsolved_variants)}")

    unsolved_csv_path = os.path.join(
        OUTPUT_DIR,
        f"{LOG_NAME}_unsolved_variants.csv"
    )

    filtered_csv_path = os.path.join(
        OUTPUT_DIR,
        f"{LOG_NAME}_all_noise_conformance_variant_runtimes"
        f"{FILTERED_SUFFIX}.csv"
    )

    filtered_log_path = os.path.join(
        LOG_DIR,
        f"{LOG_NAME}{FILTERED_SUFFIX}.xes"
    )

    write_unsolved_variants(df, unsolved_mask, unsolved_csv_path)

    keep = write_filtered_csv(df, unsolved_variants, filtered_csv_path)

    write_filtered_log(
        log,
        unsolved_variants,
        filtered_log_path,
        unique_trace_count(keep)
    )


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OPT_ALIGNMENTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    combined_output_csv_path = os.path.join(
        OUTPUT_DIR,
        f"{LOG_NAME}_all_noise_conformance_variant_runtimes.csv"
    )

    # Fresh combined CSV for each script run
    if os.path.exists(combined_output_csv_path):
        os.remove(combined_output_csv_path)

    log_path = os.path.join(LOG_DIR, f"{LOG_NAME}.xes")
    log = pm4py.read_xes(log_path, return_legacy_log_object=True)

    print(f"Loaded log: {LOG_NAME}")
    print(f"Traces: {len(log)}")
    print(f"Alignment-Variante: {ALIGN_VARIANT.name}")
    print(f"Alignment-Timeout: {ALIGN_TIMEOUT} s")

    available_cpus = os.cpu_count() or 1

    if MAX_PARALLEL_REPETITIONS is None:
        max_workers = min(REPETITIONS, available_cpus)
    else:
        max_workers = min(
            REPETITIONS,
            MAX_PARALLEL_REPETITIONS,
            available_cpus
        )

    print(f"Available CPUs: {available_cpus}")
    print(f"Parallel repetition workers: {max_workers}")

    with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(log_path,)
    ) as executor:

        for noise in NOISE_THRESHOLDS:
            noise_name = safe_noise_name(noise)

            print(f"\n==============================")
            print(f"Noise threshold: {noise}")
            print(f"==============================")

            net, im, fm = pm4py.discover_petri_net_inductive(
                log,
                noise_threshold=noise
            )

            model_path = os.path.join(
                MODEL_DIR,
                f"{LOG_NAME}_noise_{noise_name}.pnml"
            )

            pnml_exporter.apply(
                net,
                im,
                model_path,
                final_marking=fm
            )

            print(f"Saved model: {model_path}")

            tau_transitions = sum(
                1 for t in net.transitions if t.label is None
            )
            print(
                f"Modell: places={len(net.places)} "
                f"transitions={len(net.transitions)} "
                f"davon still={tau_transitions}"
            )

            shortest_path_alignment = algorithm.apply_trace(
                obj.Trace(),
                net,
                im,
                fm
            )

            shortest_path = len([
                tup for tup in shortest_path_alignment["alignment"]
                if tup[1] is not None
            ])

            print(f"Shortest path: {shortest_path}")

            opt_alignment_path = os.path.join(
                OPT_ALIGNMENTS_DIR,
                f"{LOG_NAME}_noise_{noise_name}_optimal_model_traces.txt"
            )

            total_start = time.time()

            # Each repetition is submitted as its own task.

            future_to_repetition = {}

            for repetition in range(REPETITIONS):
                future = executor.submit(
                    run_repetition,
                    repetition,
                    noise,
                    model_path,
                    shortest_path
                )

                future_to_repetition[future] = repetition

            # Collect finished repetitions.
            #
            # Writing happens exclusively in the main process,
            # and immediately as soon as a repetition is finished.

            opt_traces_written = False

            for future in as_completed(future_to_repetition):

                repetition = future_to_repetition[future]

                try:
                    values, optimal_model_traces = future.result()

                except Exception as exc:
                    print(
                        f"ERROR in repetition {repetition}, "
                        f"noise={noise}: {type(exc).__name__}: {exc}"
                    )
                    raise

                append_results_csv(
                    values,
                    combined_output_csv_path
                )

                print(
                    f"Results for repetition {repetition}, "
                    f"noise={noise} written to CSV."
                )

                # All repetitions produce identical model traces,
                # therefore the file is only written once.
                if not opt_traces_written:
                    write_optimal_model_traces(
                        optimal_model_traces,
                        opt_alignment_path
                    )
                    opt_traces_written = True

                    print(
                        f"Saved optimal model traces: "
                        f"{opt_alignment_path}"
                    )

            total_time = time.time() - total_start
            print(f"Total time for noise={noise}: {total_time}")
            print(f"Appended output CSV: {combined_output_csv_path}")

    # Collect unsolved variants, write the filtered CSV and the
    # filtered XES log. The log is still in memory here.
    finalize_outputs(log, combined_output_csv_path)

    print(f"\nFinished. Overall results: {combined_output_csv_path}")


if __name__ == "__main__":
    main()
