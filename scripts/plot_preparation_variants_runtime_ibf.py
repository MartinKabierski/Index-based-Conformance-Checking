import csv
import heapq
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import inf
from pathlib import Path

from pm4py import write_xes
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import Event, EventLog, Trace
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.importer import importer as pnml_importer

# local imports
from alignment_params_factory import generate_valid_combinations as params_factory
from alignment_txt_fitness_wrapper import alignment_txt_calculate_fitness
from enhanced_random_playout import basic_playout_unique_variants
from subprocess_wrapper import xes_to_txt

LOG_NAME = "BPI_Challenge_2012"
NOISE_THRESHOLDS = [0.8, 0.2]

REPETITIONS = 8

# None:
# automatically use as many workers as repetitions / available CPUs
MAX_PARALLEL_REPETITIONS = None

# playout params
# Target number of unique traces in the random playout
RANDOM_PLAYOUT_TRACES_TARGET = 10000
PLAYOUT_MODE = "random"
RANDOM_SEED = 42

# IBF parameters
#
# Exactly ONE parameter combination is used.
# None -> first valid combination from params_factory()
# dict -> this combination is used, e.g.
#         {"bucketing": ..., "bucket_limit": ..., "k": ..., ...}
IBF_PARAMS = None

# project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
OPT_ALIGNMENTS_DIR = BASE_DIR / "opt_alignments"
OUTPUT_DIR = BASE_DIR / "output"

# Intermediate files per repetition (deleted after merging).
# The directory must already exist.
TEMP_DIR = OUTPUT_DIR / "temp"

ORIGINAL_LOG_XES = LOGS_DIR / f"{LOG_NAME}.xes"
ORIGINAL_LOG_TXT = OUTPUT_DIR / f"{LOG_NAME}_original.txt"

# Final output: all rows of all alignment CSVs (every noise, every repetition)
OUTPUT_CSV = OUTPUT_DIR / f"IBF_alignment_{LOG_NAME}_all_noise_all_variants.csv"

ACTIVITY_KEY = "concept:name"

# helper functions
def noise_to_token(noise: float) -> str:
    # transforms e.g. 0.2 into 0p2
    return str(noise).replace(".", "p")


def parse_trace_line(line: str) -> tuple[str, ...]:
    """
    Transforms a TXT line from

        A - B - C

    into

        ("A", "B", "C")
    """
    return tuple(
        activity.strip()
        for activity in line.strip().split(" - ")
        if activity.strip()
    )


def count_traces_in_txt(path: Path) -> int:
    """Determines the number of traces actually present in the TXT log."""
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if parse_trace_line(line):
                count += 1
    return count


def load_optimal_alignment_traces(path: Path) -> list[tuple[str, ...]]:
    traces: list[tuple[str, ...]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            trace = parse_trace_line(line)
            if trace:
                traces.append(trace)
    return traces


def trace_to_tuple(trace: Trace, activity_key: str = ACTIVITY_KEY) -> tuple[str, ...]:
    """
    Converts a PM4Py Trace into a hashable tuple of activity names.
    Events without the specified activity key are ignored.
    """
    return tuple(
        event[activity_key]
        for event in trace
        if activity_key in event
    )


def tuple_to_trace(activities: tuple[str, ...]) -> Trace:
    """
    Converts a tuple of activity names back into a PM4Py Trace.
    Used to pass the shortest trace safely to worker processes.
    """
    trace = Trace()
    for activity in activities:
        trace.append(Event({ACTIVITY_KEY: activity}))
    return trace


def log_to_variant_set(log: EventLog) -> set[tuple[str, ...]]:
    return {trace_to_tuple(trace) for trace in log}


def is_silent(transition) -> bool:
    name = getattr(transition, "name", "") or ""
    label = getattr(transition, "label", None)

    return (
            label is None
            or label == ""
            or getattr(transition, "invisible", False)
            or name.startswith(("tau", "skip", "tauSplit", "tauJoin"))
    )


def marking_key(marking) -> tuple[tuple[int, int], ...]:
    """Generates a hashable marking key within a single program run."""
    return tuple(
        sorted(
            (id(place), tokens)
            for place, tokens in marking.items()
        )
    )


def shortest_visible_trace(
        net,
        initial_marking,
        final_marking,
        activity_key: str = ACTIVITY_KEY,
):
    """
    Determines a complete model path with the minimum
    number of visible transitions.

    Invisible transitions have a cost of 0.
    Visible transitions have a cost of 1.
    Purely silent paths are not accepted.

    Returns:
        trace:
            PM4Py trace containing all visible activities
            of the shortest complete path.

        cost:
            Number of visible transitions in this path.
    """

    start_key = marking_key(
        initial_marking
    )

    goal_key = marking_key(
        final_marking
    )

    priority_queue = []

    tie_breaker = 0

    heapq.heappush(
        priority_queue,
        (
            0,
            tie_breaker,
            initial_marking,
            False,
        ),
    )

    distances = {
        (
            start_key,
            False,
        ): 0
    }

    predecessors = {
        (
            start_key,
            False,
        ): (
            None,
            None,
            None,
        )
    }

    while priority_queue:

        (
            cost,
            _,
            marking,
            has_visible,
        ) = heapq.heappop(
            priority_queue
        )

        current_key = marking_key(
            marking
        )

        state = (
            current_key,
            has_visible,
        )

        if cost > distances.get(
                state,
                inf,
        ):
            continue

        # Final marking reached
        if (
                current_key == goal_key
                and has_visible
                and cost > 0
        ):

            path = []
            current_state = state

            while (
                    predecessors[
                        current_state
                    ][0]
                    is not None
            ):
                (
                    previous_key,
                    previous_visible,
                    transition,
                ) = predecessors[
                    current_state
                ]

                path.append(
                    transition
                )

                current_state = (
                    previous_key,
                    previous_visible,
                )

            path.reverse()

            trace = Trace()

            for transition in path:

                if not is_silent(
                        transition
                ):
                    trace.append(
                        Event(
                            {
                                activity_key:
                                    transition.label
                            }
                        )
                    )

            return (
                trace,
                cost,
            )

        for transition in (
                semantics.enabled_transitions(
                    net,
                    marking,
                )
        ):

            next_marking = (
                semantics.execute(
                    transition,
                    net,
                    marking,
                )
            )

            step_cost = (
                0
                if is_silent(
                    transition
                )
                else 1
            )

            next_cost = (
                    cost
                    + step_cost
            )

            next_has_visible = (
                    has_visible
                    or step_cost == 1
            )

            next_key = marking_key(
                next_marking
            )

            next_state = (
                next_key,
                next_has_visible,
            )

            if next_cost < distances.get(
                    next_state,
                    inf,
            ):
                distances[
                    next_state
                ] = next_cost

                predecessors[
                    next_state
                ] = (
                    current_key,
                    has_visible,
                    transition,
                )

                tie_breaker += 1

                heapq.heappush(
                    priority_queue,
                    (
                        next_cost,
                        tie_breaker,
                        next_marking,
                        next_has_visible,
                    ),
                )

    return (
        None,
        None,
    )


def optimal_traces_to_event_log(optimal_traces: list[tuple[str, ...]]) -> EventLog:
    log = EventLog()
    for optimal_trace in optimal_traces:
        log.append(tuple_to_trace(optimal_trace))
    return log


# ============================================================
# Mixed model log
# ============================================================

def create_mixed_model_log(
        playout_log: EventLog,
        optimal_traces: list[tuple[str, ...]],
        shortest_trace: Trace,
        seed: int,
) -> EventLog:
    """
    Generate the Mixed Log from:

        1. Random Playout
        2. Optimal alignment traces
        3. Shuffle
        4. Shortest visible trace as the LAST trace
    """
    mixed_log = EventLog()

    # 1. Playout traces
    for trace in playout_log:
        if len(trace) > 0:
            mixed_log.append(trace)

    # 2. Optimal alignment traces
    for trace in optimal_traces_to_event_log(optimal_traces):
        if len(trace) > 0:
            mixed_log.append(trace)

    # 3. Shuffle
    random.Random(seed).shuffle(mixed_log)

    # 4. Shortest visible trace at the end
    if shortest_trace is None or len(shortest_trace) == 0:
        raise RuntimeError(
            "The shortest visible trace is empty "
            "or could not be determined."
        )

    mixed_log.append(shortest_trace)

    return mixed_log


def validate_required_inputs() -> None:
    missing = []

    if not ORIGINAL_LOG_XES.exists():
        missing.append(ORIGINAL_LOG_XES)

    if not TEMP_DIR.is_dir():
        missing.append(TEMP_DIR)

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required inputs are missing:\n{formatted}")


def select_ibf_params() -> dict:
    """Returns exactly one IBF parameter combination."""
    if IBF_PARAMS is not None:
        return dict(IBF_PARAMS)

    for params in params_factory():
        return params

    raise RuntimeError(
        "params_factory() did not return "
        "any parameter combinations"
    )


# ============================================================
# Combined alignment CSV (all repetitions)
# ============================================================

ALIGNMENT_CSV_DELIMITER = ";"

ALIGNMENT_META_HEADER = [
    "noise",
    "repetition",
    "mixed_log_traces",
]


def alignment_csv_path(noise: float, repetition: int, random_playout_traces_target: int) -> Path:
    return TEMP_DIR / (
        f"alignment_{LOG_NAME}"
        f"_noise_{noise_to_token(noise)}"
        f"_rep_{repetition}"
        f"_target_{random_playout_traces_target}.csv"
    )


def read_alignment_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """
    Reads a CSV created by the alignment program.

    Rows with empty trailing fields (e.g. repeated trace variants
    without searched buckets / compared traces) are kept.
    """
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=ALIGNMENT_CSV_DELIMITER)
        header = next(reader, [])
        rows = [
            row
            for row in reader
            if any(field.strip() for field in row)
        ]
    return header, rows


def write_combined_alignment_csv(
        header: list[str],
        rows: list[list],
        write_header: bool,
) -> None:
    """
    Writes into the final output file.

    Only the main process calls this function, so multiple
    repetitions never write to the file concurrently.
    """
    with OUTPUT_CSV.open(
            "w" if write_header else "a",
            newline="",
            encoding="utf-8",
    ) as file:
        writer = csv.writer(file, delimiter=ALIGNMENT_CSV_DELIMITER)

        if write_header:
            writer.writerow(ALIGNMENT_META_HEADER + header)

        writer.writerows(rows)


def delete_temp_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"WARNING: could not delete {path}: {exc}")


# ============================================================
# One repetition
# ============================================================

def run_repetition(
        noise: float,
        repetition: int,
        random_playout_traces_target: int,
        model_path: Path,
        optimal_traces: list[tuple[str, ...]],
        shortest_trace_tuple: tuple[str, ...],
        shortest_path: int,
        max_trace_length: int,
        original_log_traces: int,
        params: dict,
) -> tuple[list[str], list[list], list[Path]]:
    """
    Executes one complete repetition in its own process.

    Returns:
        alignment_header,
        alignment_rows (content of the alignment CSV of this
        repetition, prefixed with metadata),
        temp_files (deleted by the main process after merging)
    """
    process_id = os.getpid()
    run_seed = RANDOM_SEED + repetition
    token = noise_to_token(noise)

    print(
        f"[PID {process_id}] "
        f"Starting repetition {repetition}, "
        f"target={random_playout_traces_target}, "
        f"seed={run_seed}"
    )

    # Load Petri net inside worker process
    net, initial_marking, final_marking = pnml_importer.apply(str(model_path))

    shortest_trace = tuple_to_trace(shortest_trace_tuple)

    # ========================================================
    # 1. Random Playout
    # ========================================================

    print(f"[PID {process_id}] Repetition {repetition}: creating Random Playout ...")

    random.seed(run_seed)

    playout_log = basic_playout_unique_variants(
        petri_net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
        max_trace_length=max_trace_length,
        target_unique=random_playout_traces_target,
        batch_size=max(1, random_playout_traces_target // 10),
        max_rounds=random_playout_traces_target,
    )

    actual_playout_variants = len(log_to_variant_set(playout_log))

    if actual_playout_variants < random_playout_traces_target:
        print(
            f"[PID {process_id}] "
            f"WARNING repetition {repetition}: "
            f"only {actual_playout_variants} "
            f"of {random_playout_traces_target} target variants generated."
        )

    # ========================================================
    # 2. Mixed Log
    #
    # Random Playout
    # + optimal alignment traces
    # + shuffle
    # + shortest visible trace (exactly once, at the end)
    # ========================================================

    mixed_log = create_mixed_model_log(
        playout_log=playout_log,
        optimal_traces=optimal_traces,
        shortest_trace=shortest_trace,
        seed=run_seed,
    )

    # These values already include the shortest visible trace.
    mixed_log_traces = len(mixed_log)
    mixed_log_variants = len(variants_filter.get_variants(mixed_log))

    if mixed_log_traces == 0:
        raise RuntimeError(f"Repetition {repetition}: generated Mixed Log is empty.")

    if trace_to_tuple(mixed_log[-1]) != shortest_trace_tuple:
        raise RuntimeError(
            f"Repetition {repetition}: "
            "shortest visible trace is not "
            "at the end of the Mixed Log."
        )

    print(
        f"[PID {process_id}] "
        f"Mixed Log: "
        f"{mixed_log_traces} traces | "
        f"{mixed_log_variants} variants"
    )

    mixed_log_name = (
        f"mixed_{LOG_NAME}"
        f"_noise_{token}"
        f"_rep_{repetition}"
        f"_target_{random_playout_traces_target}"
    )

    mixed_log_xes = TEMP_DIR / f"{mixed_log_name}.xes"
    mixed_log_txt = TEMP_DIR / f"{mixed_log_name}.txt"

    write_xes(mixed_log, str(mixed_log_xes))

    xes_to_txt(
        input=str(mixed_log_xes),
        output=str(mixed_log_txt),
    )

    mixed_log_txt_traces = count_traces_in_txt(mixed_log_txt)

    if mixed_log_txt_traces != mixed_log_traces:
        print(
            f"[PID {process_id}] "
            f"WARNING repetition {repetition}: "
            "Mixed Log XES and TXT contain different "
            "numbers of traces: "
            f"XES={mixed_log_traces}, "
            f"TXT={mixed_log_txt_traces}"
        )

    # ========================================================
    # 3. IBF: Original Log vs Mixed Log
    # ========================================================

    print(f"[PID {process_id}] Repetition {repetition}: starting IBF alignment ...")

    alignment_output_csv = alignment_csv_path(noise, repetition, random_playout_traces_target)

    # The return values of the wrapper are not needed here.
    # Only the CSV written by the alignment program is used.
    alignment_txt_calculate_fitness(
        params=params,
        log_txt=ORIGINAL_LOG_TXT,
        model_traces_txt=mixed_log_txt,
        output_csv=alignment_output_csv,
        shortest_path=shortest_path,
        log_length=original_log_traces,
    )

    # Content of the alignment CSV for the combined file
    alignment_header, raw_alignment_rows = read_alignment_csv(alignment_output_csv)

    meta = [noise, repetition, mixed_log_traces]

    alignment_rows = [meta + row for row in raw_alignment_rows]

    print(
        f"[PID {process_id}] "
        f"Finished repetition {repetition}, "
        f"target={random_playout_traces_target} | "
        f"{len(alignment_rows)} alignment rows"
    )

    temp_files = [
        mixed_log_xes,
        mixed_log_txt,
        alignment_output_csv,
    ]

    return alignment_header, alignment_rows, temp_files


# ============================================================
# Main
# ============================================================

def main() -> None:

    validate_required_inputs()

    # Convert original log to TXT only once,
    # before worker processes are started.
    print(f"Convert original log to TXT: {ORIGINAL_LOG_TXT}")

    original_xes_to_txt_time = xes_to_txt(
        input=str(ORIGINAL_LOG_XES),
        output=str(ORIGINAL_LOG_TXT),
    )

    original_log_traces = count_traces_in_txt(ORIGINAL_LOG_TXT)

    if original_log_traces <= 0:
        raise RuntimeError(
            "The converted original log TXT file "
            f"contains no valid traces: {ORIGINAL_LOG_TXT}"
        )

    print(
        "Original log: "
        f"{original_log_traces} traces | "
        f"TXT conversion={original_xes_to_txt_time} ms"
    )

    # ========================================================
    # IBF parameters (exactly one combination)
    # ========================================================

    params = select_ibf_params()

    print(f"IBF parameters: {params}")

    # ========================================================
    # Determine number of worker processes
    # ========================================================

    available_cpus = os.cpu_count() or 1

    if MAX_PARALLEL_REPETITIONS is None:
        max_workers = min(REPETITIONS, available_cpus)
    else:
        max_workers = min(REPETITIONS, MAX_PARALLEL_REPETITIONS, available_cpus)

    print(f"Available CPUs: {available_cpus}")
    print(f"Parallel repetition workers: {max_workers}")

    output_header_written = False

    # ========================================================
    # Noise levels
    # ========================================================

    for noise in NOISE_THRESHOLDS:

        token = noise_to_token(noise)

        model_path = MODELS_DIR / f"{LOG_NAME}_noise_{token}.pnml"

        optimal_path = OPT_ALIGNMENTS_DIR / (
            f"{LOG_NAME}_noise_{token}"
            f"_optimal_model_traces.txt"
        )

        if not model_path.exists():
            print(f"Skip {noise}: Missing model: {model_path}")
            continue

        if not optimal_path.exists():
            print(
                f"Skipping Noise {noise}: "
                f"Optimal alignment-traces are missing: {optimal_path}"
            )
            continue

        print(f"\n--- processing noise {noise} ---")

        # Model information is calculated once in main process
        net, initial_marking, final_marking = pnml_importer.apply(str(model_path))

        optimal_traces = load_optimal_alignment_traces(optimal_path)
        optimal_variants = set(optimal_traces)

        print(
            f"Optimal Alignments: "
            f"{len(optimal_traces)} traces | "
            f"{len(optimal_variants)} variants"
        )

        # Shortest visible trace
        max_optimal_length = max(
            (len(trace) for trace in optimal_variants),
            default=0,
        )

        shortest_trace, shortest_path = shortest_visible_trace(
            net,
            initial_marking,
            final_marking,
        )

        if shortest_trace is None or shortest_path is None:
            raise RuntimeError(
                f"For the model {model_path}, no complete, "
                "visible path to the final marking was found."
            )

        shortest_trace_tuple = trace_to_tuple(shortest_trace)

        print("Shortest visible trace: " + " - ".join(shortest_trace_tuple))

        max_trace_length = max_optimal_length + shortest_path

        if max_trace_length <= 0:
            raise RuntimeError(
                f"Invalid maximum trace length ({max_trace_length}) "
                f"for {model_path}."
            )

        print(
            f"Shortest visible path: {shortest_path} | "
            f"max trace length: {max_trace_length}"
        )

        # The main-process Petri net is no longer needed.
        # Workers load their own copy.
        del net, initial_marking, final_marking, shortest_trace

        # ====================================================
        # Repetitions
        # ====================================================

        print("\n======================================")
        print(f"Random playout traces target: {RANDOM_PLAYOUT_TRACES_TARGET}")
        print(
            f"Starting {REPETITIONS} repetitions "
            f"with up to {max_workers} parallel processes."
        )
        print("======================================")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            future_to_repetition = {
                executor.submit(
                    run_repetition,
                    noise,
                    repetition,
                    RANDOM_PLAYOUT_TRACES_TARGET,
                    model_path,
                    optimal_traces,
                    shortest_trace_tuple,
                    shortest_path,
                    max_trace_length,
                    original_log_traces,
                    params,
                ): repetition
                for repetition in range(REPETITIONS)
            }

            # Results are written by the main process only.
            for future in as_completed(future_to_repetition):

                repetition = future_to_repetition[future]

                try:
                    alignment_header, alignment_rows, temp_files = future.result()

                except Exception as exc:
                    print(
                        f"ERROR in repetition {repetition}, "
                        f"target={RANDOM_PLAYOUT_TRACES_TARGET}: {exc}"
                    )
                    raise

                write_combined_alignment_csv(
                    header=alignment_header,
                    rows=alignment_rows,
                    write_header=not output_header_written,
                )

                output_header_written = True

                # Content is now in the final CSV -> remove temp files
                delete_temp_files(temp_files)

                print(
                    f"Results for repetition {repetition}, "
                    f"target={RANDOM_PLAYOUT_TRACES_TARGET} written to CSV."
                )

    print(f"\nFinished. All alignment rows: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
