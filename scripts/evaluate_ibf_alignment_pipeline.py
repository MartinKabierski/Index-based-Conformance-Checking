import csv
import heapq
import math
import time
from datetime import timedelta
from itertools import product
from math import inf
from pathlib import Path

import pm4py
from pm4py import write_xes
from pm4py.algo.filtering.log.cases.case_filter import filter_on_case_size
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.obj import Trace, Event
from pm4py.objects.petri_net import semantics
from pm4py.algo.conformance.alignments.petri_net import algorithm
from pm4py.algo.filtering.log.cases.case_filter import filter_on_case_size
from pm4py.objects.log import obj

from alignment_params_factory import generate_valid_combinations as params_factory
from alignment_txt_fitness_wrapper import alignment_txt_calculate_fitness
from subprocess_wrapper import xes_to_txt
from subprocess_run_triebasedkristor import run_trie_conformance

from enhanced_random_playout import beam_playout_unique_variants
from enhanced_random_playout import basic_playout_unique_variants
from diverse_beam import dbs_beam_playout_unique_variants

# Parameters for KristoRs Trie based approach
skip_trie_based = True
#path_to_trie_runner = "/home/jessy/Documents/GitHub/pm/ConformanceCheckingUsingTries/src/main/java/ee/ut/cs/dsg/confcheck/Runner.java"
path_to_trie_runner = "./trieBasedClasses/target/classes/ee/ut/cs/dsg/confcheck/Runner.class"
path_to_java = "/usr/bin/java"

# Main parameters:
#log_name = "Road_Traffic_Fines_Management_Process"
log_name = "BPI_Challenge_2012"
#log_name = "Sepsis_Cases_-_Event_Log"
process_model_noise_thresholds = [0.2]
repetitions = 1

#max_numbers_traces = [10, 100, 1000, 10000]
#max_numbers_traces = [1000, 2000, 3000, 4000, 5000]
max_numbers_traces = [10000,20000]
#playout_modes = ["gumbel", "random", "beam"]
playout_modes = ["dbs"]
#beam_widths = [16, 32, 64, 128]
beam_widths = [64, 128, 256]
numbers_of_groups= [8, 16]
weight_activity = 0.0
ngram_weights = {2: 1.0, 3: 1.0, 4: 1.0}
max_diversity_ngrams = [1,2,3,4]
max_traces_per_depth = 4
max_traces_per_round = 2000


# This basically affects how fitness is calculated
# if there is a difference between the shortest path and the
# shortest VISIBLE path through a model
use_shortest_visible_as_shortest = False

# The parameters for the ibf construction and search are in params_config.json
# to enable all bucketings, put this in the enabled section of the params file:  "i", "tracelen", "random", "lsh", "omh","set", "multiset"

log_path = "../logs/" + log_name + ".xes"

def main():
    start_time_evaluation = time.time()

    # Load the XES file
    log = pm4py.read_xes(log_path, return_legacy_log_object=True)

    # Length of log
    log_length = len(log)

    # Convert log to .txt for later comparison
    xes_to_txt(input=log_path, output="../output/" + log_name + ".txt")

    trace_simulation_length = 0

    # if a maximum trace length is provided, remove all longer traces from the log
    if trace_simulation_length != 0:
        cropped_log = filter_on_case_size(log, min_case_size=0, max_case_size=trace_simulation_length)
        write_xes(cropped_log,
                  "../output/" + "cropped_log_" + log_name + "_max_length_" + str(trace_simulation_length) + ".xes")
    else:
        cropped_log = log

    # Calculate length of each trace (number of events per trace) and determine max
    trace_lengths = [len(trace) for trace in cropped_log]
    max_trace_length = max(trace_lengths)
    print("Longest trace in Log:", max_trace_length)
    print("###############################################")

    # Define the header of the CSV file
    # If you want to add columns, do not forget to also edit the part where values are written to the csv
    csv_header = [
        "bucketing_algorithm",
        "max_checked_buckets",
        "kmer_size",
        "number_buckets_calc",
        "sort_by",
        "num_hashes",
        "omh_w",
        "omh_seed",
        "process_model_noise_threshold",
        "repetition",
        "playout_mode",
        "beam_width",
        "number_of_groups",
        "max_diversity_ngram",
        "playout_max_number_traces",
        "playout_trace_variants",
        "fitness_value",
        "total_cost",
        "total_cost_adjusted",
        "total_length_all_traces",
        "log_number_traces",
        "shortest_path",
        "process_model_discovery_time",
        "playout_time",
        "xes_to_txt_conversion_time",
        "trace_calculation_time",
        "total_ibf_creation_time",
        "total_ibf_search_time",
        "trie_based_alignment_time",
        "trie_based_fitness"
    ]

    with open("../output/evaluation_result_" + log_name + ".csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header
        writer.writerow(csv_header)

        # TODO: threads for repetitions

    for max_number_traces, process_model_noise_threshold, playout_mode, beam_width, number_of_groups, max_diversity_ngram, repetition in product(max_numbers_traces,
                                                                                process_model_noise_thresholds, playout_modes,
                                                                                beam_widths, numbers_of_groups, max_diversity_ngrams,
                                                                                range(repetitions)):
        start_time_total = time.time()

        print("max_number_traces:", max_number_traces,
              "\nprocess_model_noise_threshold:", process_model_noise_threshold,
              "\nrepetition:", repetition)

        # Discover Petri net (process_model) using the inductive miner
        start_time_process_model_discovery = time.time()
        process_model, initial_marking, final_marking = pm4py.discover_petri_net_inductive(cropped_log,
                                                                                           noise_threshold=process_model_noise_threshold)
        #export pnml for use in e.g. ProM
        # model_path = "../output/" + log_name + "_model_noise" + str(process_model_noise_threshold) + ".pnml"
        # pm4py.write_pnml(
        #     process_model,
        #     initial_marking,
        #     final_marking,
        #     model_path
        # )

        # If you want to view the model, uncomment this:
        # pm4py.view_petri_net(process_model, initial_marking, final_marking)

        process_model_discovery_time = timedelta(
            seconds=time.time() - start_time_process_model_discovery) / timedelta(milliseconds=1)

        if trace_simulation_length == 0 and repetition == 0:
            # calculates the shortest visible path from im to fm (no path containing only tau transitions)
            shortest_trace, shortest_visible_path = shortest_visible_trace(process_model, initial_marking, final_marking)
            print("Shortest non-silent path in process model:", shortest_visible_path)

            # calculates the shortest path from im to fm (including silent paths)
            shortest_path = algorithm.apply_trace(obj.Trace(), process_model, initial_marking, final_marking)
            shortest_path = len([tup for tup in shortest_path["alignment"] if tup[1] is not None])
            print("Shortest path in process model:", shortest_path)

            if use_shortest_visible_as_shortest:
                shortest_path = shortest_visible_path

            #there is usually no need to simulate a trace that is longer than this (for alignment)
            trace_simulation_length = max_trace_length + shortest_path

        print("Maximum considered simulated trace length:", trace_simulation_length)

        # simulate log from the model
        print("Simulating Log from process model")
        start_time_playout = time.time()
        playout_log_path = "../output/playout_logs/" + log_name +"_pruning_log_"+ playout_mode +"_playout_beam_width_" + str(beam_width) + "_number_of_groups_" + str(number_of_groups) + "_for_" + str(max_number_traces) + "_traces.csv"
        if playout_mode == "random":

            beam_width = 0

            # the provided parameters work well for larger trace counts, adjust them for e.g. 10
            simulated_log = basic_playout_unique_variants(petri_net=process_model, initial_marking=initial_marking,
                                                          final_marking=final_marking,
                                                          max_trace_length=trace_simulation_length,
                                                          target_unique=max_number_traces,
                                                          batch_size=int(max_number_traces / 10),
                                                          max_rounds=max_number_traces)

        elif playout_mode == "dbs":
            simulated_log = dbs_beam_playout_unique_variants(
                petri_net=process_model,
                initial_marking=initial_marking,
                final_marking=final_marking,
                max_trace_length=trace_simulation_length,
                target_unique=max_number_traces,
                beam_width=beam_width,
                max_rounds=max_number_traces,
                num_groups= number_of_groups,
                lambda_div= 0.5,
                max_traces_per_depth=max_traces_per_depth,
                max_traces_per_round=max_traces_per_round,
                weight_activity=weight_activity,
                ngram_weights=ngram_weights,
                max_diversity_ngram=max_diversity_ngram,
                csv_path=playout_log_path
            )
        else:
            # the provided beam_width parameter was arbitrarily chosen and should (depending on the log) be experimented with
            simulated_log = beam_playout_unique_variants(
                petri_net=process_model,
                initial_marking=initial_marking,
                final_marking=final_marking,
                max_trace_length=trace_simulation_length,
                target_unique=max_number_traces,
                beam_width=beam_width,
                selection_mode=playout_mode,
                max_rounds=max_number_traces,
                max_traces_per_depth=max_traces_per_depth,
                max_traces_per_round=max_traces_per_round,
                weight_activity=weight_activity,
                ngram_weights=ngram_weights,
                csv_path=playout_log_path
            )

        #initialize values to see what was simulated
        max_simulated_length = 0
        min_simulated_length = math.inf

        for i in range(len(simulated_log)):
            # prevents empty traces from appearing in the final simulated log
            while len(simulated_log[i]) == 0:
                tr = pm4py.algo.simulation.playout.petri_net.algorithm.basic_playout.apply(process_model,
                                                                                           initial_marking,
                                                                                           final_marking,
                                                                                           parameters={
                                                                                               "noTraces": 1,
                                                                                               "maxTraceLength": trace_simulation_length,
                                                                                               "add_only_if_fm_is_reached": True
                                                                                           })
                simulated_log[i] = tr[0]

            # track max and min length
            if len(simulated_log[i]) > max_simulated_length:
                max_simulated_length = len(simulated_log[i])
            if len(simulated_log[i]) < min_simulated_length:
                min_simulated_length = len(simulated_log[i])

        print("Longest trace length in playout:", max_simulated_length)
        print("Shortest trace length in playout:", min_simulated_length)
        print("Shortest visible trace length in process model:", shortest_path)
        # the shortest trace is added to the simulation so there is always an upper bound to the alignment cost
        simulated_log.append(shortest_trace)
        print("Shortest visible trace added to playout")

        playout_trace_variants = len(variants_filter.get_variants(simulated_log))

        playout_time = timedelta(seconds=time.time() - start_time_playout) / timedelta(milliseconds=1)

        print("Playout time:", playout_time)

        # this log name is later used to find the file for alignment
        simulated_log_name = "traces_" + playout_mode + "_" + log_name + "_noise" + str(
            process_model_noise_threshold) + "_max_number_traces" + str(max_number_traces) + "_trace_length" + str(
            trace_simulation_length)

        simulated_log_path = "../output/" + simulated_log_name + ".xes"
        simulated_log_path_txt = "../output/" + simulated_log_name + ".txt"
        write_xes(simulated_log, simulated_log_path)

        xes_to_txt_conversion_time = xes_to_txt(input=simulated_log_path,
                                                output=simulated_log_path_txt)

        print("----------------------------------------------")
        print("Anzahl Varianten:", playout_trace_variants)
        print("----------------------------------------------")
        #break
        # TODO: BREAK here, if you are only interested in experimenting with playout parameters


        print("IBF alignment of original log to simulated log")

        # the params are in params_config.json
        for i, params in enumerate(params_factory(), 1):
            print(f"{i}: {params}")
            bucketing_algorithm = params["bucketing"]
            bucket_limit = params["bucket_limit"]
            kmer_size = params["k"]
            number_buckets_calc = params["buckets_calc"]
            num_hashes = params["num_hashes"]
            omh_w = params["omh_w"]
            omh_seed = params["omh_seed"]
            sorting = params["sorting"]

            # this calls the subprocess with the alignment c++ program and calculates the fitness
            fitness_value, trace_calculation_time, total_ibf_creation_time, total_ibf_search_time, total_cost, total_cost_adjusted, total_length_all_traces, number_traces = alignment_txt_calculate_fitness(
                params=params,
                log_txt=Path("../output/" + log_name + ".txt"),
                model_traces_txt=Path(simulated_log_path_txt),
                output_csv=Path("../output/" + "alignment_" + simulated_log_name + ".csv"),
                shortest_path=shortest_path,
                log_length=log_length)

            print("For playout_trace_variants " + str(playout_trace_variants) + " playout_mode " + str(playout_mode) + " beam_width " + str(beam_width) + " number_of_groups " + str(number_of_groups) + " max_diversity_ngram " + str(max_diversity_ngram) )

            if (skip_trie_based):
                trie_fitness, trie_runtime = 0.0, 0.0
            else:
                print("_________________________________________________________________________________________________")
                print("Starting Trie based alignment of simulated log to original log...")
                trie_fitness, trie_runtime = run_trie_conformance(runner_class=path_to_trie_runner, proxy_log=simulated_log_path, sample_log=log_path, java_bin=path_to_java)
                print("Trie based alignment finished after {0}ms with a fitness of {1}".format(trie_runtime, trie_fitness))
                print("_________________________________________________________________________________________________")



            with open("../output/evaluation_result_" + log_name + ".csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # fill CSV-file
                # If you want to add columns, do not forget to also edit the header!
                writer.writerow([
                    bucketing_algorithm,
                    bucket_limit,
                    kmer_size,
                    number_buckets_calc,
                    sorting,
                    num_hashes,
                    omh_w,
                    omh_seed,
                    process_model_noise_threshold,
                    repetition,
                    playout_mode,
                    beam_width,
                    number_of_groups,
                    max_diversity_ngram,
                    max_number_traces,
                    playout_trace_variants,
                    fitness_value,
                    total_cost,
                    total_cost_adjusted,
                    total_length_all_traces,
                    number_traces,
                    shortest_path,
                    process_model_discovery_time,
                    playout_time,
                    xes_to_txt_conversion_time,
                    trace_calculation_time,
                    total_ibf_creation_time,
                    total_ibf_search_time,
                    trie_runtime,
                    trie_fitness
                ])

        total_time = timedelta(seconds=time.time() - start_time_total) / timedelta(milliseconds=1)

        print("Total time for max_number_traces " + str(
            max_number_traces) + " playout_mode " + str(playout_mode) + " process_model_noise_threshold " + str(process_model_noise_threshold)
            + " beam_width " + str(beam_width)
            + " number_of_groups " + str(number_of_groups)
            + " max_diversity_ngram " + str(max_diversity_ngram)
              + " repetition " + str(repetition) + " : " + str(total_time) + "ms")

        print("##################################################################################################")

    print("Finished evaluation after " + str(timedelta(seconds=time.time() - start_time_evaluation)))




# Helpers for shortest visible path/trace
def _mkey(marking):
    # hashable key for markings
    return tuple(sorted((id(p), v) for p, v in marking.items()))


def is_silent(t):
    name = (getattr(t, "name", "") or "")
    return (t.label is None) or (t.label == "") or getattr(t, "invisible", False) or \
        name.startswith(("tau", "skip", "tauSplit", "tauJoin"))


def shortest_visible_path_transitions(net, initial_marking, final_marking):
    """
    Dijkstra in augmented space:
      State = (Marking, seen_visible ∈ {0,1})
      Cost = Number of visible transitions
    Returns (Transition list incl. τ, visible_length) or (None, None).
    """
    start_k = _mkey(initial_marking)
    goal_k = _mkey(final_marking)

    # (cost, tie, marking, seen_visible)
    pq = []
    tie = 0
    heapq.heappush(pq, (0, tie, initial_marking, 0))

    # Distance per augmented state
    dist = {(start_k, 0): 0}
    # Predecessor for path reconstruction: (prev_key, prev_seen_visible, transition)
    prev = {(start_k, 0): (None, None, None)}

    while pq:
        cost, _, m, seen_vis = heapq.heappop(pq)
        mk = _mkey(m)
        if cost > dist.get((mk, seen_vis), inf):
            continue

        # Goal achieved AND at least one visible transition used
        if mk == goal_k and seen_vis == 1 and cost > 0:
            # Reconstruct path
            path = []
            cur = (mk, seen_vis)
            while prev[cur][0] is not None:
                pkey, pvis, t = prev[cur]
                path.append(t)
                cur = (pkey, pvis)
            path.reverse()
            return path, cost

        # expand
        for t in semantics.enabled_transitions(net, m):
            m2 = semantics.execute(t, net, m)
            step = 0 if is_silent(t) else 1
            new_cost = cost + step
            new_seen_vis = 1 if (seen_vis == 1 or step == 1) else 0
            key2 = (_mkey(m2), new_seen_vis)
            if new_cost < dist.get(key2, inf):
                dist[key2] = new_cost
                prev[key2] = (mk, seen_vis, t)
                tie += 1
                heapq.heappush(pq, (new_cost, tie, m2, new_seen_vis))

    return None, None  # no path with ≥1 visible steps


def shortest_visible_trace(net, initial_marking, final_marking, activity_key="concept:name"):
    # Returns (Trace, visible_length). Trace contains ONLY visible events.

    path, visible_len = shortest_visible_path_transitions(net, initial_marking, final_marking)
    if path is None:
        return None, None

    tr = Trace()
    for t in path:
        if not is_silent(t):
            tr.append(Event({activity_key: t.label}))
    return tr, visible_len


# this was important for extensive playouts and is not used at the moment
def filter_log_by_activity_repetition(log, threshold=3):
    """
    Filters an event log so that only the traces remain,
    in which no activity occurs more than 'threshold' times.

    :param log: EventLog, the imported event log
    :param threshold: Maximum permitted repetitions of an activity per trace
    :return: EventLog, the filtered log
    """
    filtered_log = EventLog()
    for trace in log:
        counts = {}
        valid = True
        for event in trace:
            activity = event["concept:name"]
            counts[activity] = counts.get(activity, 0) + 1
            if counts[activity] > threshold:
                valid = False
                break
        if valid:
            filtered_log.append(trace)
    return filtered_log

if __name__ == '__main__':
    main()
