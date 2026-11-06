"""
Log-to-Log Alignment Approximator

Reference:
Fani Sani et al. (2023)
"Alignment Approximator: A ProM Plug-In to Approximate
Conformance Statistics"

https://vdaalst.com/publications/p1406.pdf

TXT format: One trace per line, events separated by " - ".

Costs: insertion 1, deletion 1, match 0, substitution 2.

Timeout: 60 seconds per trace variant by default.
On timeout, the best cost found so far is retained.
Timeouts are checked only before each proxy comparison.

All runtimes are reported in milliseconds.
"""

from __future__ import annotations

import csv

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns


# ============================================================
# 1. DATA TYPES
# ============================================================

Trace = tuple[str, ...]

LogInput = str | Path | Iterable[Iterable[str]]

DELIMITER = " - "


# ============================================================
# 2. RESULT CLASSES
# ============================================================

@dataclass(frozen=True)
class VariantResult:

    trace: Trace
    frequency: int
    cost: int
    weighted_cost: int
    runtime_ms: float
    closest_proxy: Trace
    timed_out: bool
    compared_proxies: int


@dataclass(frozen=True)
class ComparisonResult:

    total_cost: int

    runtime_ms: float
    preprocessing_ms: float
    comparison_ms: float
    csv_write_ms: float

    number_of_traces: int
    number_of_variants: int
    number_of_reference_variants: int

    variant_details: tuple[VariantResult, ...]

    number_of_timed_out_variants: int


# ============================================================
# 3. READ LOGS
# ============================================================

# Optimization 1: Group identical traces and weight costs by frequency.

def read_traces(source: LogInput) -> Counter[Trace]:

    variants: Counter[Trace] = Counter()

    if isinstance(source, (str, Path)):

        with Path(source).open(
            "r",
            encoding="utf-8-sig"
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                trace = tuple(
                    event.strip()
                    for event in line.split(DELIMITER)
                )

                if any(not event for event in trace):
                    raise ValueError(
                        f"Empty event in a row {line_number}: {source}"
                    )

                variants[trace] += 1

    else:

        for trace in source:

            if isinstance(trace, str):
                raise TypeError(
                    "A trace must be a sequence of events."
                )

            events = tuple(trace)

            if not all(
                isinstance(event, str) and event
                for event in events
            ):
                raise ValueError(
                    "Events must be non-empty strings."
                )

            variants[events] += 1

    return variants


# ============================================================
# 4. EDIT DISTANCE
# ============================================================

def edit_distance(
    log_trace: Trace,
    proxy_trace: Trace,
    threshold: int,
) -> int | None:
    """
    Compute the minimum insertion/deletion distance between two traces.

    threshold: Best distance found for the current log variant.
    Returns the edit distance, or None if the proxy cannot improve it.

    Costs: match 0, insertion 1, deletion 1, substitution 2.
    """

    # Optimization 4: Skip proxies whose length difference >= threshold.

    if abs(len(log_trace) - len(proxy_trace)) >= threshold:
        return None

    # Optimization 2: Store only two DP rows (O(m) space).

    previous = list(
        range(len(proxy_trace) + 1)
    )

    for i, log_event in enumerate(
        log_trace,
        start=1
    ):

        current = [i]

        for j, proxy_event in enumerate(
            proxy_trace,
            start=1
        ):

            # Deletion
            delete_cost = previous[j] + 1

            # Insertion
            insert_cost = current[j - 1] + 1

            if log_event == proxy_event:

                # Matching event
                match_cost = previous[j - 1]

                current.append(
                    min(
                        delete_cost,
                        insert_cost,
                        match_cost
                    )
                )

            else:

                # No direct substitution
                current.append(
                    min(
                        delete_cost,
                        insert_cost
                    )
                )

        # Optimization 5: Stop if the row minimum cannot beat the threshold.
        # Check only from i >= threshold; current[0] == i.

        if i >= threshold and min(current) >= threshold:
            return None

        previous = current

    return previous[-1]


# ============================================================
# 5. LOG-TO-LOG COMPARISON
# ============================================================

def approximate_alignment(
    event_log: LogInput,
    reference_log: LogInput,
    csv_output: str | Path | None = None,
    timeout_seconds: float | None = 60.0,
) -> ComparisonResult:
    """
    Find the minimum edit distance to the available proxies for each log variant.

    Total cost = sum(frequency * minimum_cost).

    Args:
        event_log: TXT file or iterable of traces.
        reference_log: TXT file or iterable of proxy traces.
        csv_output: Optional CSV export of variant results.
        timeout_seconds: Timeout per variant (default: 60 seconds).
            None disables the timeout.

    All runtimes are in milliseconds.
    """

    # --------------------------------------------------------
    # 5.1 SET UP TIMEOUT
    # --------------------------------------------------------

    if (
        timeout_seconds is not None
        and timeout_seconds < 0
    ):
        raise ValueError(
            "timeout_seconds must be >= 0 or None."
        )

    timeout_ns = (
        None
        if timeout_seconds is None
        else int(timeout_seconds * 1_000_000_000)
    )

    start_ns = perf_counter_ns()

    # --------------------------------------------------------
    # 5.2 READ LOGS
    # --------------------------------------------------------

    event_variants = read_traces(event_log)

    proxy_variants = read_traces(reference_log)

    if not event_variants:
        raise ValueError("The event log is empty.")

    if not proxy_variants:
        raise ValueError("The proxy log is empty.")

    # Use unique proxy traces only
    proxies = list(proxy_variants)

    # Shortest proxy provides a valid fallback cost
    shortest_proxy = min(
        proxies,
        key=len
    )

    preprocessing_ms = (
        perf_counter_ns() - start_ns
    ) / 1_000_000

    results: list[VariantResult] = []

    total_cost = 0

    comparison_start_ns = perf_counter_ns()

    # --------------------------------------------------------
    # 5.3 PROCESS LOG VARIANTS
    # --------------------------------------------------------

    for trace, frequency in event_variants.items():

        variant_start_ns = perf_counter_ns()

        # Per-variant timeout
        deadline_ns = (
            None
            if timeout_ns is None
            else variant_start_ns + timeout_ns
        )

        # Initial cost: delete all log events and insert the shortest proxy.

        minimum_cost = (
            len(trace) + len(shortest_proxy)
        )

        closest_proxy = shortest_proxy

        compared_proxies = 0

        timed_out = False

        # ----------------------------------------------------
        # 5.4 COMPARE PROXY TRACES
        # ----------------------------------------------------

        for proxy in proxies:

            # Check the timeout only before starting a proxy comparison.

            if (
                deadline_ns is not None
                and perf_counter_ns() >= deadline_ns
            ):

                timed_out = True
                break

            # Use the best cost as a threshold for early termination.

            cost = edit_distance(
                trace,
                proxy,
                threshold=minimum_cost,
            )

            # None means no improvement is possible, not a timeout.

            if cost is None:
                continue

            compared_proxies += 1

            # Update the best match

            if cost < minimum_cost:

                minimum_cost = cost

                closest_proxy = proxy

            # Optimization 3: Distance 0 is optimal; stop searching.

            if minimum_cost == 0:
                break

        # ----------------------------------------------------
        # 5.5 STORE VARIANT RESULT
        # ----------------------------------------------------

        variant_ms = (
            perf_counter_ns() - variant_start_ns
        ) / 1_000_000

        weighted_cost = (
            frequency * minimum_cost
        )

        total_cost += weighted_cost

        results.append(
            VariantResult(
                trace=trace,
                frequency=frequency,
                cost=minimum_cost,
                weighted_cost=weighted_cost,
                runtime_ms=variant_ms,
                closest_proxy=closest_proxy,
                timed_out=timed_out,
                compared_proxies=compared_proxies,
            )
        )

    # --------------------------------------------------------
    # 5.6 CALCULATE RUNTIMES
    # --------------------------------------------------------

    comparison_ms = (
        perf_counter_ns() - comparison_start_ns
    ) / 1_000_000

    runtime_ms = (
        perf_counter_ns() - start_ns
    ) / 1_000_000

    # --------------------------------------------------------
    # 6. CSV EXPORT
    # --------------------------------------------------------

    csv_write_ms = 0.0

    if csv_output is not None:

        csv_start_ns = perf_counter_ns()

        with Path(csv_output).open(
            "w",
            encoding="utf-8",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "variant_id",
                "trace",
                "trace_length",
                "frequency",
                "cost_per_trace",
                "weighted_cost",
                "runtime_ms",
                "closest_proxy",
                "timed_out",
                "compared_proxies",
                "total_proxies",
            ])

            for variant_id, result in enumerate(
                results,
                start=1
            ):

                writer.writerow([
                    variant_id,
                    DELIMITER.join(result.trace),
                    len(result.trace),
                    result.frequency,
                    result.cost,
                    result.weighted_cost,
                    result.runtime_ms,
                    DELIMITER.join(result.closest_proxy),
                    result.timed_out,
                    result.compared_proxies,
                    len(proxies),
                ])

        csv_write_ms = (
            perf_counter_ns() - csv_start_ns
        ) / 1_000_000

    # --------------------------------------------------------
    # 7. FINAL RESULT
    # --------------------------------------------------------

    # Report comparison time excluding preprocessing and CSV export.

    print(
        f"Fani Sani - pure comparison time: "
        f"{comparison_ms:.3f} ms"
    )

    return ComparisonResult(
        total_cost=total_cost,

        runtime_ms=runtime_ms,

        preprocessing_ms=preprocessing_ms,

        comparison_ms=comparison_ms,

        csv_write_ms=csv_write_ms,

        number_of_traces=sum(
            event_variants.values()
        ),

        number_of_variants=len(
            event_variants
        ),

        number_of_reference_variants=len(
            proxies
        ),

        variant_details=tuple(
            results
        ),

        number_of_timed_out_variants=sum(
            result.timed_out
            for result in results
        ),
    )
