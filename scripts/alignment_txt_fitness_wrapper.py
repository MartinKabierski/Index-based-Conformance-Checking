import csv
from pathlib import Path

from subprocess_wrapper import run_alignment_ibf


def calculate_fitness(filepath, shortest_path):
    # Open file
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        header = next(reader)  # Skip first line

        number_traces = 0
        total_cost = 0
        total_cost_adjusted = 0
        total_length_all_traces = 0
        trace_calculation_time = 0.0

        # Store all costs for each trace variant
        trace_variants = {}

        for row in reader:
            if len(row) < 4:
                continue  # Skip empty or incomplete lines

            number_traces += 1

            # Analyse first column
            trace = row[0].strip()
            trace_events = [event.strip() for event in trace.split(' - ')]
            trace_length = len(trace_events)
            total_length_all_traces += trace_length

            try:
                levenshtein_distance = int(row[2])
                calculation_time = float(row[3])

                total_cost += levenshtein_distance
                total_cost_adjusted += min(
                    levenshtein_distance,
                    shortest_path + trace_length
                )
                trace_calculation_time += calculation_time

                # Variant-based calculation
                if trace not in trace_variants:
                    trace_variants[trace] = []

                trace_variants[trace].append(levenshtein_distance)

            except ValueError:
                pass  # If cost or calculation time is invalid

    # fitness calculation
    try:
        fitness_value = 1 - (
                total_cost_adjusted
                / (total_length_all_traces + (number_traces * shortest_path))
        )
    except ZeroDivisionError:
        fitness_value = 0

    # ---------------------------------------------------------
    # Variant-based evaluation
    # ---------------------------------------------------------

    number_trace_variants = len(trace_variants)
    total_variant_cost = 0
    variant_cost_inconsistencies = 0

    for trace, costs in trace_variants.items():
        unique_costs = set(costs)

        if len(unique_costs) > 1:
            variant_cost_inconsistencies += 1

            print("[WARNING] Different costs found for trace variant:")
            print("  Trace:", trace)
            print("  Costs:", costs)

        # The expected case is that there is exactly one cost
        # for every trace variant.
        total_variant_cost += costs[0]

    if number_trace_variants > 0:
        mean_variant_cost = total_variant_cost / number_trace_variants
    else:
        mean_variant_cost = 0

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print("Number of all traces:", number_traces)
    print("Number of trace variants:", number_trace_variants)

    print("Sum of all deviations:", total_cost)
    print("Adjusted sum of all deviations:", total_cost_adjusted)

    print("Total error over trace variants:", total_variant_cost)
    print("Mean error over trace variants:", mean_variant_cost)
    print("Variant cost inconsistencies:", variant_cost_inconsistencies)

    print("Total length of all traces:", total_length_all_traces)
    print("Calculation time over all traces:", trace_calculation_time)
    print("Fitness:", fitness_value)

    return (
        fitness_value,
        trace_calculation_time,
        total_cost,
        total_cost_adjusted,
        total_length_all_traces,
        number_traces,
        total_variant_cost,
        mean_variant_cost,
        number_trace_variants,
        variant_cost_inconsistencies,
    )


def alignment_txt_calculate_fitness(
        params=None,
        log_txt=Path('../output/BPI_Challenge_2018.txt'),
        model_traces_txt=Path(
            '../output/traces_BPI_Challenge_2018_noise0.2_max_loop1_trace_length2973.txt'
        ),
        output_csv=Path('../output/complete_alignment.csv'),
        shortest_path=0,
        log_length=0):

    if params is None:
        params = {}

    print("Starting alignment...")

    # 1. Execute florian marx' program
    total_ibf_creation_time, total_search_time = run_alignment_ibf(
        params=params,
        search=str(log_txt),
        traces=str(model_traces_txt),
        output=str(output_csv),
        shortest_path=shortest_path,
        log_length=log_length
    )

    print(f"[INFO] run completed. Final output in {output_csv}")
    print("[INFO] Calculating fitness...")

    (
        fitness_value,
        trace_calculation_time,
        total_cost,
        total_cost_adjusted,
        total_length_all_traces,
        number_traces,
        total_variant_cost,
        mean_variant_cost,
        number_trace_variants,
        variant_cost_inconsistencies,
    ) = calculate_fitness(output_csv, shortest_path)

    return (
        fitness_value,
        trace_calculation_time,
        total_ibf_creation_time,
        total_search_time,
        total_cost,
        total_cost_adjusted,
        total_length_all_traces,
        number_traces,
        total_variant_cost,
        mean_variant_cost,
        number_trace_variants,
        variant_cost_inconsistencies,
    )