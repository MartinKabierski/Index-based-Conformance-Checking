import csv
from collections import defaultdict

# ---------------- Configuration ----------------
# How should variants be treated for which no optimal
# alignment could be computed (status == "unsolved")?
#
#   "worst_case" : cost = trace_length.
#                  This is the upper bound of the optimal cost,
#                  because every trace can always be aligned in
#                  such a way that all events are log moves. Yields
#                  the lower bound of the fitness. Is guaranteed not
#                  to overestimate the fitness.
#
#   "best_case"  : cost = 0.
#                  Lower bound of the cost, upper bound of the
#                  fitness. Only useful as a counter-check.
#
#   "exclude"    : the row is ignored completely.
#                  CAUTION: the unsolved variants are the
#                  hardest ones, leaving them out raises the
#                  fitness artificially.
UNSOLVED_POLICY = "worst_case"


# ----------------------------------------------


def parse_optional_number(value):
    # Empty fields occur for unsolved variants.
    # In addition, pandas writes integers as "0.0" as soon as
    # there is an empty value anywhere in the column.
    if value is None:
        return None

    value = value.strip()

    if value == "" or value.lower() in ("nan", "none"):
        return None

    return float(value)


def aggregate(filepath, unsolved_policy):
    groups = defaultdict(lambda: {
        "number_traces": 0,
        "total_length": 0,
        "total_cost": 0,
        "trace_calculation_time": 0.0,
        "shortest_path": None,
        "fitness": 0.0
    })

    noise_totals = defaultdict(lambda: {
        "number_traces": 0,
        "total_length": 0,
        "total_cost": 0,
        "trace_calculation_time": 0.0,
        "shortest_path": None,
        "fitness": 0.0,
        "repetition_count": 0
    })

    stats = {
        "rows": 0,
        "solved": 0,
        "unsolved": 0,
        "unsolved_traces": 0,
        "unsolved_length": 0
    }

    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")

        for row in reader:
            repetition = row.get("repetition")
            noise = row.get("noise_threshold")

            if repetition is None or noise is None:
                continue

            stats["rows"] += 1

            variant_count = int(float(row["count"]))
            trace_length = int(float(row["trace_length"]))
            shortest_path = int(float(row["shortest_path"]))
            calculation_time_ms = float(row["time"]) * 1000

            cost = parse_optional_number(row["cost"])

            if cost is None:
                # Unsolved variant (timeout or error).
                stats["unsolved"] += 1
                stats["unsolved_traces"] += variant_count
                stats["unsolved_length"] += trace_length * variant_count

                if unsolved_policy == "exclude":
                    continue
                elif unsolved_policy == "worst_case":
                    cost = trace_length
                elif unsolved_policy == "best_case":
                    cost = 0
                else:
                    raise ValueError(
                        f"Unbekannte UNSOLVED_POLICY: {unsolved_policy}"
                    )
            else:
                stats["solved"] += 1

            cost = int(cost)

            key = (noise, repetition)

            for g in (groups[key], noise_totals[noise]):
                g["number_traces"] += variant_count
                g["total_length"] += trace_length * variant_count
                g["total_cost"] += cost * variant_count
                g["trace_calculation_time"] += calculation_time_ms

                if g["shortest_path"] is None:
                    g["shortest_path"] = shortest_path

    # Compute fitness per noise + repetition
    for g in groups.values():
        denominator = (
            g["total_length"] + (g["number_traces"] * g["shortest_path"])
        )
        g["fitness"] = (
            1 - (g["total_cost"] / denominator) if denominator else 0.0
        )

    # Determine the number of repetitions per noise
    repetitions_by_noise = defaultdict(set)

    for noise, repetition in groups.keys():
        repetitions_by_noise[noise].add(repetition)

    # Compute AVG rows per noise
    for noise, g in noise_totals.items():
        repetition_count = len(repetitions_by_noise[noise])
        g["repetition_count"] = repetition_count

        if repetition_count > 0:
            g["number_traces"] /= repetition_count
            g["total_length"] /= repetition_count
            g["total_cost"] /= repetition_count
            g["trace_calculation_time"] /= repetition_count

        denominator = (
            g["total_length"] + (g["number_traces"] * g["shortest_path"])
        )
        g["fitness"] = (
            1 - (g["total_cost"] / denominator) if denominator else 0.0
        )

    return groups, noise_totals, stats


def write_output(groups, noise_totals, output_csv):
    fieldnames = [
        "noise_threshold",
        "repetition",
        "number_traces",
        "total_length",
        "total_cost",
        "trace_calculation_time",
        "shortest_path",
        "fitness"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (noise, repetition), g in sorted(
                groups.items(),
                key=lambda x: (float(x[0][0]), int(x[0][1]))
        ):
            writer.writerow({
                "noise_threshold": noise,
                "repetition": repetition,
                "number_traces": g["number_traces"],
                "total_length": g["total_length"],
                "total_cost": g["total_cost"],
                "trace_calculation_time": g["trace_calculation_time"],
                "shortest_path": g["shortest_path"],
                "fitness": g["fitness"]
            })

        for noise, g in sorted(
                noise_totals.items(), key=lambda x: float(x[0])
        ):
            writer.writerow({
                "noise_threshold": noise,
                "repetition": "AVG",
                "number_traces": g["number_traces"],
                "total_length": g["total_length"],
                "total_cost": g["total_cost"],
                "trace_calculation_time": g["trace_calculation_time"],
                "shortest_path": g["shortest_path"],
                "fitness": g["fitness"]
            })


def calculate_fitness_by_noise_and_repetition(filepath, output_csv):
    groups, noise_totals, stats = aggregate(filepath, UNSOLVED_POLICY)
    write_output(groups, noise_totals, output_csv)

    return {
        "by_noise_and_repetition": dict(groups),
        "by_noise_avg": dict(noise_totals),
        "stats": stats
    }


if __name__ == "__main__":
    log_name = "BPI_Challenge_2019"

    fitness_filepath = (
        "../output/"
        f"{log_name}_all_noise_conformance_variant_runtimes.csv"
    )

    output_csv_filename = (
        "../output/"
        f"fitness_baseline_all_noise_{log_name}.csv"
    )

    result = calculate_fitness_by_noise_and_repetition(
        fitness_filepath,
        output_csv=output_csv_filename
    )

    stats = result["stats"]

    print("--- Abdeckung ---")
    print(f"Zeilen gesamt:      {stats['rows']}")
    print(f"Exakt geloest:      {stats['solved']}")
    print(f"Ungeloest:          {stats['unsolved']}")
    print(f"Betroffene Traces:  {stats['unsolved_traces']}")

    if stats["rows"]:
        anteil = stats["solved"] / stats["rows"]
        print(f"Variantenabdeckung: {anteil:.4%}")

    print(f"\nVerwendete Policy:  {UNSOLVED_POLICY}")

    print("\n--- Fitness je Policy (AVG-Zeilen) ---")
    print("worst_case = untere Schranke, best_case = obere Schranke.")

    for policy in ("worst_case", "best_case", "exclude"):
        _, totals, _ = aggregate(fitness_filepath, policy)

        for noise, g in sorted(totals.items(), key=lambda x: float(x[0])):
            print(
                f"  noise={noise:<6} {policy:<11} "
                f"fitness={g['fitness']:.6f}"
            )

    print(f"\nGeschrieben: {output_csv_filename}")
