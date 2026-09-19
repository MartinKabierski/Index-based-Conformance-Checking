import csv
import heapq
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import inf
from pathlib import Path

import pm4py
from pm4py import write_xes
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import Event, EventLog, Trace
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.importer import importer as pnml_importer

# local imports
from alignment_params_factory import generate_valid_combinations as params_factory
from alignment_txt_fitness_wrapper import alignment_txt_calculate_fitness
from enhanced_random_playout import basic_playout_unique_variants
from subprocess_run_triebasedkristor import run_trie_conformance
from subprocess_wrapper import xes_to_txt

LOG_NAME = "BPI_Challenge_2012"
NOISE_THRESHOLDS = [0.8, 0.2]

REPETITIONS = 8

# None:
# automatically use as many workers as repetitions / available CPUs
MAX_PARALLEL_REPETITIONS = None

# playout params
TARGET_VARIANTS_LIST = [1000, 5000, 10000]
PLAYOUT_MODE = "random"
RANDOM_SEED = 42

# Kristor params
SKIP_TRIE_BASED = False

PATH_TO_TRIE_RUNNER = (
    "./trieBasedClasses/target/classes/"
    "ee/ut/cs/dsg/confcheck/Runner.class"
)

PATH_TO_JAVA = "/usr/bin/java"

# project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
OPT_ALIGNMENTS_DIR = BASE_DIR / "opt_alignments"
OUTPUT_DIR = BASE_DIR / "output"

ORIGINAL_LOG_XES = (
        LOGS_DIR
        / f"{LOG_NAME}.xes"
)

ORIGINAL_LOG_TXT = (
        OUTPUT_DIR
        / f"{LOG_NAME}_original.txt"
)

OUTPUT_CSV = (
        OUTPUT_DIR
        / f"Comparison_IBF_vs_TRIE_on_playout_with_opt_alignments_{LOG_NAME}.csv"
)

ACTIVITY_KEY = "concept:name"


# helper functions
def elapsed_ms(start_time: float) -> float:
    return (
            time.perf_counter()
            - start_time
    ) * 1000.0


def noise_to_token(noise: float) -> str:
    # transforms e.g. 0.2 into 0p2
    return str(noise).replace(".", "p")


def parse_trace_line(
        line: str,
) -> tuple[str, ...]:
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


def count_traces_in_txt(
        path: Path,
) -> int:
    """
    Determines the number of traces actually present
    in the TXT log.
    """

    count = 0

    with path.open(
            "r",
            encoding="utf-8",
    ) as file:

        for line in file:

            trace = parse_trace_line(
                line
            )

            if trace:
                count += 1

    return count


def load_optimal_alignment_traces(
        path: Path,
) -> list[tuple[str, ...]]:
    traces: list[tuple[str, ...]] = []

    with path.open(
            "r",
            encoding="utf-8",
    ) as file:

        for line in file:

            trace = parse_trace_line(
                line
            )

            if trace:
                traces.append(
                    trace
                )

    return traces


def trace_to_tuple(
        trace: Trace,
        activity_key: str = ACTIVITY_KEY,
) -> tuple[str, ...]:
    """
    Converts a PM4Py Trace into a hashable tuple of activity names.

    This representation is used to compare traces and to identify
    unique trace variants, for example when storing traces in a set.

    Events without the specified activity key are ignored.
    """

    return tuple(
        event[activity_key]
        for event in trace
        if activity_key in event
    )


def tuple_to_trace(
        activities: tuple[str, ...],
) -> Trace:
    """
    Converts a tuple of activity names back into a PM4Py Trace.

    This is useful when passing the shortest trace to worker
    processes because tuples can safely be serialized.
    """

    trace = Trace()

    for activity in activities:
        trace.append(
            Event(
                {
                    ACTIVITY_KEY:
                        activity
                }
            )
        )

    return trace


def log_to_variant_set(
        log: EventLog,
) -> set[tuple[str, ...]]:
    return {
        trace_to_tuple(trace)
        for trace in log
    }


def is_silent(
        transition,
) -> bool:
    name = (
            getattr(
                transition,
                "name",
                "",
            )
            or ""
    )

    label = getattr(
        transition,
        "label",
        None,
    )

    return (
            label is None
            or label == ""
            or getattr(
        transition,
        "invisible",
        False,
    )
            or name.startswith(
        (
            "tau",
            "skip",
            "tauSplit",
            "tauJoin",
        )
    )
    )


def marking_key(
        marking,
) -> tuple[tuple[int, int], ...]:
    """
    Generates a hashable marking key within a single program run.
    """

    return tuple(
        sorted(
            (
                id(place),
                tokens,
            )
            for place, tokens
            in marking.items()
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


def optimal_traces_to_event_log(
        optimal_traces: list[tuple[str, ...]],
) -> EventLog:
    log = EventLog()

    for optimal_trace in optimal_traces:

        trace = Trace()

        for activity in optimal_trace:
            trace.append(
                Event(
                    {
                        ACTIVITY_KEY:
                            activity
                    }
                )
            )

        log.append(
            trace
        )

    return log


# ============================================================
# Random model log
# ============================================================

def create_random_model_log(
        playout_log: EventLog,
        shortest_trace: Trace,
) -> EventLog:
    """
    Creates the Random Log from:

        1. Random Playout
        2. Shortest visible trace as the LAST trace

    The shortest visible trace is included exactly once.
    """

    random_log = EventLog()

    for trace in playout_log:

        if len(trace) > 0:
            random_log.append(
                trace
            )

    if (
            shortest_trace is not None
            and len(shortest_trace) > 0
    ):

        random_log.append(
            shortest_trace
        )

    else:

        raise RuntimeError(
            "The shortest visible trace is empty "
            "or could not be determined."
        )

    return random_log


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
            mixed_log.append(
                trace
            )

    # 2. Optimal alignment traces
    optimal_event_log = (
        optimal_traces_to_event_log(
            optimal_traces
        )
    )

    for trace in optimal_event_log:

        if len(trace) > 0:
            mixed_log.append(
                trace
            )

    # 3. Shuffle
    random.Random(
        seed
    ).shuffle(
        mixed_log
    )

    # 4. Shortest visible trace at the end
    if (
            shortest_trace is not None
            and len(shortest_trace) > 0
    ):

        mixed_log.append(
            shortest_trace
        )

    else:

        raise RuntimeError(
            "The shortest visible trace is empty "
            "or could not be determined."
        )

    return mixed_log


def validate_required_inputs() -> None:
    missing = []

    if not ORIGINAL_LOG_XES.exists():
        missing.append(
            ORIGINAL_LOG_XES
        )

    if (
            not Path(
                PATH_TO_JAVA
            ).exists()
            and not SKIP_TRIE_BASED
    ):
        missing.append(
            Path(
                PATH_TO_JAVA
            )
        )

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Required inputs are missing:\n"
            f"{formatted}"
        )


def get_csv_header() -> list[str]:
    return [
        "noise",
        "repetition",
        "playout_mode",
        "random_seed",

        "target_variants",

        "playout_traces",
        "playout_variants",

        "random_log_traces",
        "random_log_variants",

        "opt_alignment_traces",
        "opt_alignment_variants",

        "mixed_log_traces",
        "mixed_log_variants",

        "original_log_traces",
        "original_log_variants",

        # IBF parameters
        "bucketing",
        "bucket_limit",
        "k",
        "buckets_calc",
        "num_hashes",
        "omh_w",
        "omh_seed",
        "sorting",

        # IBF: Original Log vs Random Log
        "random_ibf_fitness",
        "random_ibf_total_cost",
        "random_ibf_total_cost_adjusted",
        "random_ibf_total_length_all_traces",
        "random_ibf_log_number_traces",

        # Variant-based evaluation
        "random_ibf_number_trace_variants",
        "random_ibf_total_variant_cost",
        "random_ibf_mean_variant_cost",
        "random_ibf_variant_cost_inconsistencies",

        "random_ibf_trace_calculation_time",
        "random_ibf_creation_time",
        "random_ibf_search_time",

        # IBF: Original Log vs Mixed Log
        "mixed_ibf_fitness",
        "mixed_ibf_total_cost",
        "mixed_ibf_total_cost_adjusted",
        "mixed_ibf_total_length_all_traces",
        "mixed_ibf_log_number_traces",

        # Variant-based evaluation
        "mixed_ibf_number_trace_variants",
        "mixed_ibf_total_variant_cost",
        "mixed_ibf_mean_variant_cost",
        "mixed_ibf_variant_cost_inconsistencies",

        "mixed_ibf_trace_calculation_time",
        "mixed_ibf_creation_time",
        "mixed_ibf_search_time",

        "shortest_path",

        "playout_time_ms",

        "random_log_xes_write_time_ms",
        "random_log_xes_to_txt_time_ms",

        "mixed_log_xes_write_time_ms",
        "mixed_log_xes_to_txt_time_ms",

        # Total time of both IBF calls for one parameter combination
        "ibf_parameter_run_time_ms",

        # Kristor: Original Log vs Random Log
        "trie_random_alignment_time_ms",
        "trie_random_fitness",

        # Kristor: Original Log vs Mixed Log
        "trie_mixed_alignment_time_ms",
        "trie_mixed_fitness",

        "total_experiment_time_ms",
    ]


def write_csv_header() -> None:
    with OUTPUT_CSV.open(
            "w",
            newline="",
            encoding="utf-8",
    ) as file:
        csv.writer(
            file
        ).writerow(
            get_csv_header()
        )


def append_csv_rows(
        rows: list[list],
) -> None:
    """
    Writes result rows from completed worker processes.

    Only the main process calls this function.
    Therefore multiple repetitions never write to the
    same result CSV concurrently.
    """

    with OUTPUT_CSV.open(
            "a",
            newline="",
            encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerows(
            rows
        )


# ============================================================
# One repetition
# ============================================================

def run_repetition(
        noise: float,
        repetition: int,
        target_variants: int,
        model_path: Path,
        optimal_traces: list[tuple[str, ...]],
        shortest_trace_tuple: tuple[str, ...],
        shortest_path: int,
        max_trace_length: int,
        original_log_traces: int,
        original_log_variants: int,
        parameter_combinations: list[dict],
) -> list[list]:
    """
    Executes one complete repetition.

    Each repetition runs in its own process.
    """

    experiment_start = (
        time.perf_counter()
    )

    process_id = os.getpid()

    run_seed = (
            RANDOM_SEED
            + repetition
    )

    token = noise_to_token(
        noise
    )

    print(
        f"[PID {process_id}] "
        f"Starting repetition {repetition}, "
        f"target={target_variants}, "
        f"seed={run_seed}"
    )

    # ========================================================
    # Load Petri net inside worker process
    # ========================================================

    (
        net,
        initial_marking,
        final_marking,
    ) = pnml_importer.apply(
        str(
            model_path
        )
    )

    shortest_trace = (
        tuple_to_trace(
            shortest_trace_tuple
        )
    )

    # ========================================================
    # 1. Random Playout
    # ========================================================

    print(
        f"[PID {process_id}] "
        f"Repetition {repetition}: "
        "creating Random Playout ..."
    )

    playout_start = (
        time.perf_counter()
    )

    random.seed(
        run_seed
    )

    playout_log = (
        basic_playout_unique_variants(
            petri_net=net,
            initial_marking=initial_marking,
            final_marking=final_marking,
            max_trace_length=max_trace_length,
            target_unique=target_variants,
            batch_size=max(
                1,
                target_variants // 10,
            ),
            max_rounds=target_variants,
        )
    )

    playout_time_ms = (
        elapsed_ms(
            playout_start
        )
    )

    # Raw playout statistics BEFORE shortest visible trace
    playout_variants = (
        log_to_variant_set(
            playout_log
        )
    )

    actual_playout_variants = len(
        playout_variants
    )

    if (
            actual_playout_variants
            < target_variants
    ):
        print(
            f"[PID {process_id}] "
            f"WARNING repetition {repetition}: "
            f"only {actual_playout_variants} "
            f"of {target_variants} target variants generated."
        )

    # ========================================================
    # Random Log
    #
    # Random Playout + shortest visible trace
    # ========================================================

    random_log = (
        create_random_model_log(
            playout_log=playout_log,
            shortest_trace=shortest_trace,
        )
    )

    # These values already include the shortest visible trace.
    random_log_traces = len(
        random_log
    )

    random_log_variants = len(
        variants_filter.get_variants(
            random_log
        )
    )

    if len(random_log) == 0:
        raise RuntimeError(
            f"Repetition {repetition}: "
            "generated Random Log is empty."
        )

    if (
            trace_to_tuple(
                random_log[-1]
            )
            != shortest_trace_tuple
    ):
        raise RuntimeError(
            f"Repetition {repetition}: "
            "shortest visible trace is not "
            "at the end of the Random Log."
        )

    print(
        f"[PID {process_id}] "
        f"Random Log: "
        f"{random_log_traces} traces | "
        f"{random_log_variants} variants"
    )

    # ========================================================
    # Save Random Log
    # ========================================================

    random_log_name = (
        f"random_{LOG_NAME}"
        f"_noise_{token}"
        f"_rep_{repetition}"
        f"_target_{target_variants}"
    )

    random_log_xes = (
            OUTPUT_DIR
            / f"{random_log_name}.xes"
    )

    random_log_txt = (
            OUTPUT_DIR
            / f"{random_log_name}.txt"
    )

    random_write_start = (
        time.perf_counter()
    )

    write_xes(
        random_log,
        str(
            random_log_xes
        ),
    )

    random_log_xes_write_time_ms = (
        elapsed_ms(
            random_write_start
        )
    )

    random_log_xes_to_txt_time = (
        xes_to_txt(
            input=str(
                random_log_xes
            ),
            output=str(
                random_log_txt
            ),
        )
    )

    random_log_txt_traces = (
        count_traces_in_txt(
            random_log_txt
        )
    )

    if (
            random_log_txt_traces
            != random_log_traces
    ):
        print(
            f"[PID {process_id}] "
            f"WARNING repetition {repetition}: "
            "Random Log XES and TXT contain different "
            "numbers of traces: "
            f"XES={random_log_traces}, "
            f"TXT={random_log_txt_traces}"
        )

    # ========================================================
    # 2. Mixed Log
    #
    # Raw Random Playout
    # + optimal alignment traces
    # + shuffle
    # + shortest visible trace
    #
    # The shortest trace is added exactly once.
    # ========================================================

    mixed_log = (
        create_mixed_model_log(
            playout_log=playout_log,
            optimal_traces=optimal_traces,
            shortest_trace=shortest_trace,
            seed=run_seed,
        )
    )

    # These values already include the shortest visible trace.
    mixed_log_traces = len(
        mixed_log
    )

    mixed_log_variants = len(
        variants_filter.get_variants(
            mixed_log
        )
    )

    if len(mixed_log) == 0:
        raise RuntimeError(
            f"Repetition {repetition}: "
            "generated Mixed Log is empty."
        )

    if (
            trace_to_tuple(
                mixed_log[-1]
            )
            != shortest_trace_tuple
    ):
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
        f"_target_{target_variants}"
    )

    mixed_log_xes = (
            OUTPUT_DIR
            / f"{mixed_log_name}.xes"
    )

    mixed_log_txt = (
            OUTPUT_DIR
            / f"{mixed_log_name}.txt"
    )

    mixed_write_start = (
        time.perf_counter()
    )

    write_xes(
        mixed_log,
        str(
            mixed_log_xes
        ),
    )

    mixed_log_xes_write_time_ms = (
        elapsed_ms(
            mixed_write_start
        )
    )

    mixed_log_xes_to_txt_time = (
        xes_to_txt(
            input=str(
                mixed_log_xes
            ),
            output=str(
                mixed_log_txt
            ),
        )
    )

    mixed_log_txt_traces = (
        count_traces_in_txt(
            mixed_log_txt
        )
    )

    if (
            mixed_log_txt_traces
            != mixed_log_traces
    ):
        print(
            f"[PID {process_id}] "
            f"WARNING repetition {repetition}: "
            "Mixed Log XES and TXT contain different "
            "numbers of traces: "
            f"XES={mixed_log_traces}, "
            f"TXT={mixed_log_txt_traces}"
        )

    # ========================================================
    # 3. KristoR
    #
    # Original Log vs Random Log
    # Original Log vs Mixed Log
    #
    # These checks are executed once per repetition/target,
    # before the IBF parameter loop.
    # ========================================================

    if SKIP_TRIE_BASED:

        trie_random_fitness = 0.0
        trie_random_runtime = 0.0

        trie_mixed_fitness = 0.0
        trie_mixed_runtime = 0.0

    else:

        # KristoR: Original vs Random

        print(
            f"[PID {process_id}] "
            f"Repetition {repetition}: "
            "starting Kristor "
            "Original vs Random ..."
        )

        (
            trie_random_fitness,
            trie_random_runtime,
        ) = run_trie_conformance(
            runner_class=(
                PATH_TO_TRIE_RUNNER
            ),
            proxy_log=str(
                random_log_xes
            ),
            sample_log=str(
                ORIGINAL_LOG_XES
            ),
            java_bin=(
                PATH_TO_JAVA
            ),
        )

        print(
            f"[PID {process_id}] "
            f"Kristor Original vs Random finished: "
            f"fitness={trie_random_fitness}, "
            f"internal search time="
            f"{trie_random_runtime} ms"
        )

        # ----------------------------------------------------
        # KristoR: Original vs Mixed

        print(
            f"[PID {process_id}] "
            f"Repetition {repetition}: "
            "starting Kristor "
            "Original vs Mixed ..."
        )

        (
            trie_mixed_fitness,
            trie_mixed_runtime,
        ) = run_trie_conformance(
            runner_class=(
                PATH_TO_TRIE_RUNNER
            ),
            proxy_log=str(
                mixed_log_xes
            ),
            sample_log=str(
                ORIGINAL_LOG_XES
            ),
            java_bin=(
                PATH_TO_JAVA
            ),
        )

        print(
            f"[PID {process_id}] "
            f"Kristor Original vs Mixed finished: "
            f"fitness={trie_mixed_fitness}, "
            f"internal search time="
            f"{trie_mixed_runtime} ms"
        )

    # 4. IBF parameter combinations
    result_rows = []

    for (
            parameter_index,
            params,
    ) in enumerate(
        parameter_combinations,
        1,
    ):
        print(
            f"[PID {process_id}] "
            f"Repetition {repetition} | "
            f"IBF comparison "
            f"({parameter_index}/"
            f"{len(parameter_combinations)})"
        )
        parameter_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Both IBF calls use the same helper CSV.
        #
        # Random writes first.
        # Mixed writes second and may overwrite it.
        # ----------------------------------------------------

        alignment_output_csv = (
                OUTPUT_DIR
                / (
                    f"alignment_{LOG_NAME}"
                    f"_noise_{token}"
                    f"_rep_{repetition}"
                    f"_target_{target_variants}.csv"
                )
        )

        # ====================================================
        # IBF: Original Log vs Random Log

        (
            random_ibf_fitness,
            random_ibf_trace_calculation_time,
            random_ibf_creation_time,
            random_ibf_search_time,
            random_ibf_total_cost,
            random_ibf_total_cost_adjusted,
            random_ibf_total_length_all_traces,
            random_ibf_number_traces,

            # Variant-based results
            random_ibf_total_variant_cost,
            random_ibf_mean_variant_cost,
            random_ibf_number_trace_variants,
            random_ibf_variant_cost_inconsistencies,

        ) = alignment_txt_calculate_fitness(
            params=params,
            log_txt=ORIGINAL_LOG_TXT,
            model_traces_txt=random_log_txt,
            output_csv=alignment_output_csv,
            shortest_path=shortest_path,
            log_length=original_log_traces,
        )

        # ====================================================
        # IBF: Original Log vs Mixed Log

        (
            mixed_ibf_fitness,
            mixed_ibf_trace_calculation_time,
            mixed_ibf_creation_time,
            mixed_ibf_search_time,
            mixed_ibf_total_cost,
            mixed_ibf_total_cost_adjusted,
            mixed_ibf_total_length_all_traces,
            mixed_ibf_number_traces,

            # Variant-based results
            mixed_ibf_total_variant_cost,
            mixed_ibf_mean_variant_cost,
            mixed_ibf_number_trace_variants,
            mixed_ibf_variant_cost_inconsistencies,

        ) = alignment_txt_calculate_fitness(
            params=params,
            log_txt=ORIGINAL_LOG_TXT,
            model_traces_txt=mixed_log_txt,
            output_csv=alignment_output_csv,
            shortest_path=shortest_path,
            log_length=original_log_traces,
        )

        # Total runtime for both IBF calls together.
        ibf_parameter_run_time_ms = (
            elapsed_ms(
                parameter_start
            )
        )

        total_experiment_time_ms = (
            elapsed_ms(
                experiment_start
            )
        )

        row = [
            noise,
            repetition,
            PLAYOUT_MODE,
            run_seed,

            target_variants,

            # Raw Playout
            len(
                playout_log
            ),
            actual_playout_variants,

            # Random Log including shortest visible trace
            random_log_traces,
            random_log_variants,

            # Optimal alignment traces
            len(
                optimal_traces
            ),
            len(
                set(
                    optimal_traces
                )
            ),

            # Mixed Log including shortest visible trace
            mixed_log_traces,
            mixed_log_variants,

            # Original Log
            original_log_traces,
            original_log_variants,

            # IBF parameters
            params.get(
                "bucketing"
            ),
            params.get(
                "bucket_limit"
            ),
            params.get(
                "k"
            ),
            params.get(
                "buckets_calc"
            ),
            params.get(
                "num_hashes"
            ),
            params.get(
                "omh_w"
            ),
            params.get(
                "omh_seed"
            ),
            params.get(
                "sorting"
            ),

            # IBF: Original vs Random
            random_ibf_fitness,
            random_ibf_total_cost,
            random_ibf_total_cost_adjusted,
            random_ibf_total_length_all_traces,
            random_ibf_number_traces,

            # Variant-based evaluation
            random_ibf_number_trace_variants,
            random_ibf_total_variant_cost,
            random_ibf_mean_variant_cost,
            random_ibf_variant_cost_inconsistencies,

            random_ibf_trace_calculation_time,
            random_ibf_creation_time,
            random_ibf_search_time,

            # IBF: Original vs Mixed
            mixed_ibf_fitness,
            mixed_ibf_total_cost,
            mixed_ibf_total_cost_adjusted,
            mixed_ibf_total_length_all_traces,
            mixed_ibf_number_traces,

            # Variant-based evaluation
            mixed_ibf_number_trace_variants,
            mixed_ibf_total_variant_cost,
            mixed_ibf_mean_variant_cost,
            mixed_ibf_variant_cost_inconsistencies,

            mixed_ibf_trace_calculation_time,
            mixed_ibf_creation_time,
            mixed_ibf_search_time,

            shortest_path,

            playout_time_ms,

            random_log_xes_write_time_ms,
            random_log_xes_to_txt_time,

            mixed_log_xes_write_time_ms,
            mixed_log_xes_to_txt_time,

            # Both IBF calls together
            ibf_parameter_run_time_ms,

            # KristoR: Original vs Random
            trie_random_runtime,
            trie_random_fitness,

            # KristoR: Original vs Mixed
            trie_mixed_runtime,
            trie_mixed_fitness,

            total_experiment_time_ms,
        ]

        result_rows.append(
            row
        )

        print(
            f"[PID {process_id}] "
            f"Finished parameter combination "
            f"{parameter_index} | "
            f"IBF Random fitness="
            f"{random_ibf_fitness} | "
            f"IBF Random mean variant cost="
            f"{random_ibf_mean_variant_cost} | "
            f"IBF Mixed fitness="
            f"{mixed_ibf_fitness} | "
            f"IBF Mixed mean variant cost="
            f"{mixed_ibf_mean_variant_cost} | "
            f"combined IBF time="
            f"{ibf_parameter_run_time_ms:.2f} ms"
        )

    print(
        f"[PID {process_id}] "
        f"Finished repetition {repetition}, "
        f"target={target_variants}"
    )

    return result_rows


# ============================================================
# Main
def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_required_inputs()

    write_csv_header()

    # Original Log
    print(
        f"Load Original Log: "
        f"{ORIGINAL_LOG_XES}"
    )

    original_log_raw = (
        pm4py.read_xes(
            str(
                ORIGINAL_LOG_XES
            )
        )
    )

    # convert to EventLog
    original_log = (
        pm4py.convert_to_event_log(
            original_log_raw
        )
    )

    # determines the original variants
    original_log_variants = len(
        variants_filter.get_variants(
            original_log
        )
    )

    # Convert original log to TXT only once,
    # before worker processes are started.
    print(
        f"Convert original log to TXT: "
        f"{ORIGINAL_LOG_TXT}"
    )

    original_xes_to_txt_time = (
        xes_to_txt(
            input=str(
                ORIGINAL_LOG_XES
            ),
            output=str(
                ORIGINAL_LOG_TXT
            ),
        )
    )

    original_log_traces = (
        count_traces_in_txt(
            ORIGINAL_LOG_TXT
        )
    )

    if original_log_traces <= 0:
        raise RuntimeError(
            "The converted original log TXT file "
            "contains no valid traces: "
            f"{ORIGINAL_LOG_TXT}"
        )

    print(
        "Original log: "
        f"{original_log_traces} traces | "
        f"{original_log_variants} variants | "
        f"TXT conversion="
        f"{original_xes_to_txt_time} ms"
    )

    # ========================================================
    # IBF parameter combinations
    # ========================================================

    parameter_combinations = list(
        params_factory()
    )

    if not parameter_combinations:
        raise RuntimeError(
            "params_factory() did not return "
            "any parameter combinations"
        )

    print(
        f"IBF parameter combinations: "
        f"{len(parameter_combinations)}"
    )

    # ========================================================
    # Determine number of worker processes
    # ========================================================

    available_cpus = (
            os.cpu_count()
            or 1
    )

    if MAX_PARALLEL_REPETITIONS is None:

        max_workers = min(
            REPETITIONS,
            available_cpus,
        )

    else:

        max_workers = min(
            REPETITIONS,
            MAX_PARALLEL_REPETITIONS,
            available_cpus,
        )

    print(
        f"Available CPUs: {available_cpus}"
    )

    print(
        f"Parallel repetition workers: "
        f"{max_workers}"
    )

    # ========================================================
    # Noise levels
    # ========================================================

    for noise in NOISE_THRESHOLDS:

        token = noise_to_token(
            noise
        )

        model_path = (
                MODELS_DIR
                / f"{LOG_NAME}_noise_{token}.pnml"
        )

        optimal_path = (
                OPT_ALIGNMENTS_DIR
                / (
                    f"{LOG_NAME}_noise_{token}"
                    f"_optimal_model_traces.txt"
                )
        )

        if not model_path.exists():
            print(
                f"Skip {noise}: "
                f"Missing model: {model_path}"
            )

            continue

        if not optimal_path.exists():
            print(
                f"Skipping Noise {noise}: "
                "Optimal alignment-traces are missing: "
                f"{optimal_path}"
            )

            continue

        print(
            f"\n--- processing noise {noise} ---"
        )

        # Model information is calculated once in main process

        # load Petri-Net
        (
            net,
            initial_marking,
            final_marking,
        ) = pnml_importer.apply(
            str(
                model_path
            )
        )

        # Load Optimal Alignment traces
        optimal_traces = (
            load_optimal_alignment_traces(
                optimal_path
            )
        )

        optimal_variants = set(
            optimal_traces
        )

        print(
            f"Optimal Alignments: "
            f"{len(optimal_traces)} traces | "
            f"{len(optimal_variants)} variants"
        )

        # ====================================================
        # Determines Shortest Visible Trace
        # ====================================================

        max_optimal_length = max(
            (
                len(trace)
                for trace
                in optimal_variants
            ),
            default=0,
        )

        (
            shortest_trace,
            shortest_path,
        ) = shortest_visible_trace(
            net,
            initial_marking,
            final_marking,
        )

        if (
                shortest_trace is None
                or shortest_path is None
        ):
            raise RuntimeError(
                f"For the model {model_path}, no complete, visible path to the final marking was found."
            )

        shortest_trace_tuple = (
            trace_to_tuple(
                shortest_trace
            )
        )

        print(
            "Shortest visible trace: "
            + " - ".join(
                shortest_trace_tuple
            )
        )

        max_trace_length = (
                max_optimal_length
                + shortest_path
        )

        if max_trace_length <= 0:
            raise RuntimeError(
                "Invalid maximum trace length "
                f"({max_trace_length}) "
                f"for {model_path}."
            )

        print(
            f"Shortest visible path: "
            f"{shortest_path} | "
            f"max trace length: "
            f"{max_trace_length}"
        )

        # The main-process Petri net is no longer needed.
        # Workers load their own copy.
        del net
        del initial_marking
        del final_marking
        del shortest_trace

        # ====================================================
        # Target variant sizes
        # ====================================================

        for target_variants in (
                TARGET_VARIANTS_LIST
        ):

            print(
                "\n======================================"
            )

            print(
                f"Target variants: "
                f"{target_variants}"
            )

            print(
                f"Starting {REPETITIONS} repetitions "
                f"with up to {max_workers} "
                f"parallel processes."
            )

            print(
                "======================================"
            )

            # ------------------------------------------------
            # Each repetition is submitted as an independent
            # process.
            # ------------------------------------------------

            with ProcessPoolExecutor(
                    max_workers=max_workers
            ) as executor:

                future_to_repetition = {}

                for repetition in range(
                        REPETITIONS
                ):
                    future = executor.submit(
                        run_repetition,
                        noise,
                        repetition,
                        target_variants,
                        model_path,
                        optimal_traces,
                        shortest_trace_tuple,
                        shortest_path,
                        max_trace_length,
                        original_log_traces,
                        original_log_variants,
                        parameter_combinations,
                    )

                    future_to_repetition[
                        future
                    ] = repetition

                # --------------------------------------------
                # Collect completed repetitions.
                #
                # Results are written by the main process only.
                # --------------------------------------------

                for future in as_completed(
                        future_to_repetition
                ):

                    repetition = (
                        future_to_repetition[
                            future
                        ]
                    )

                    try:

                        result_rows = (
                            future.result()
                        )

                    except Exception as exc:

                        print(
                            f"ERROR in repetition "
                            f"{repetition}, "
                            f"target={target_variants}: "
                            f"{exc}"
                        )

                        raise

                    append_csv_rows(
                        result_rows
                    )

                    print(
                        f"Results for repetition "
                        f"{repetition}, "
                        f"target={target_variants} "
                        f"written to CSV."
                    )

    print(
        f"\nFinished. Overall results: "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()