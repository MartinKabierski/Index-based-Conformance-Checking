import csv
from collections import defaultdict


def calculate_fitness_by_noise_and_repetition(filepath, output_csv):
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

    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")

        for row in reader:
            repetition = row.get("repetition")
            noise = row.get("noise_threshold")

            if repetition is None or noise is None:
                continue

            variant_count = int(row["count"])
            trace_length = int(row["trace_length"])
            cost = int(row["cost"])
            shortest_path = int(row["shortest_path"])
            calculation_time_ms = float(row["time"]) * 1000

            key = (noise, repetition)

            for g in (groups[key], noise_totals[noise]):
                g["number_traces"] += variant_count
                g["total_length"] += trace_length * variant_count
                g["total_cost"] += cost * variant_count
                g["trace_calculation_time"] += calculation_time_ms

                if g["shortest_path"] is None:
                    g["shortest_path"] = shortest_path

    # calculates fitness per noise + repetition
    for g in groups.values():
        denominator = g["total_length"] + (g["number_traces"] * g["shortest_path"])
        g["fitness"] = 1 - (g["total_cost"] / denominator) if denominator else 0.0

    repetitions_by_noise = defaultdict(set)

    for noise, repetition in groups.keys():
        repetitions_by_noise[noise].add(repetition)

    for noise, g in noise_totals.items():
        repetition_count = len(repetitions_by_noise[noise])
        g["repetition_count"] = repetition_count

        if repetition_count > 0:
            g["number_traces"] /= repetition_count
            g["total_length"] /= repetition_count
            g["total_cost"] /= repetition_count
            g["trace_calculation_time"] /= repetition_count

        denominator = g["total_length"] + (g["number_traces"] * g["shortest_path"])
        g["fitness"] = 1 - (g["total_cost"] / denominator) if denominator else 0.0

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

        for noise, g in sorted(noise_totals.items(), key=lambda x: float(x[0])):
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

    return {
        "by_noise_and_repetition": dict(groups),
        "by_noise_avg": dict(noise_totals)
    }


if __name__ == "__main__":
    log_name = "BPI_Challenge_2012"

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

    print(result)