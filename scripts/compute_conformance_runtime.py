import time

import pandas as pd
import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm
from pm4py.objects.log import obj
from tqdm import tqdm

REPETITIONS = 1

def get_alignment_cost(a):
    return sum([1 if (step[1] == ">>" or (step[0] == ">>" and step[1] is not None)) else 0 for step in a["alignment"]])

def main():
    log = pm4py.read_xes("./BPI_Challenge_2012.xes", return_legacy_log_object=True)
    net,im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)

    shortest_path = algorithm.apply_trace(obj.Trace(), net, im, fm)
    shortest_path = len([tup for tup in shortest_path["alignment"] if tup[1] is not None])

    values = []
    TIME = time.time()
    for i in range(0,REPETITIONS):
        variants = {}
        for idx, trace in enumerate(log):
            print(f"Repetition {i}: Trace {idx}/{len(log)}")
            trace_string = " - ".join([x["concept:name"] for x in trace])
            if trace_string in variants:
                variants[trace_string] = variants[trace_string] + 1
                print("Skipping")
                print()
            else:
                variants[trace_string] = 1
                t_start = time.time()
                conf = pm4py.conformance.conformance_diagnostics_alignments(log[idx:idx+1], net, im , fm)
                t =time.time() - t_start
                print(f"   Cost: {get_alignment_cost(conf[0])}, Fitness: {1- (get_alignment_cost(conf[0])/(len(trace)+shortest_path))}, Time: {t} ms")
                print()
#TEST
                values.append({"repetition": i,
                           "trace": trace_string,
                           "time":t,
                           "cost": get_alignment_cost(conf[0]),
                           "cost_pm4py": conf[0]["cost"],
                           "cost_pm4py_normalized": conf[0]["cost"] % 10000,
                           "trace_length": len(trace),
                           "shortest_path": shortest_path,
                           "fitness": 1- (get_alignment_cost(conf[0])/(len(trace)+shortest_path)),
                           "fitness_pm4py": 1- (conf[0]["cost"] % 10000/(len(trace)+shortest_path)),})
        for v in values:
            v["count"] = variants[v["trace"]]
    print(f"Total Time: {time.time()-TIME}")
    pd.DataFrame(values).to_csv("./BPI_Challenge_2012_conformance_variant_runtimes.csv")

if __name__ == "__main__":
    main()