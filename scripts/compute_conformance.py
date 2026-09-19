import pandas as pd
import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm
from pm4py.objects.log import obj
from tqdm import tqdm

### Computes conformance for all traces using alignments. Uses multithreading for efficiency

INPUT = "../logs/BPI_Challenge_2012"

def get_alignment_cost(a):
    #print([1 if (step[1] == ">>" or (step[0] == ">>" and step[1] is not None)) else 0 for step in a["alignment"]])
    return sum([1 if (step[1] == ">>" or (step[0] == ">>" and step[1] is not None)) else 0 for step in a["alignment"]])

def main():
    log = pm4py.read_xes("./"+ INPUT + ".xes", return_legacy_log_object=True)

    #net,im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.0)
    #pm4py.view_petri_net(net,im,fm)

    #conf = pm4py.conformance.conformance_diagnostics_token_based_replay(log, net,im,fm)
    #m = c = r = p = 0
    #for x in conf:
    #    m += x["missing_tokens"]
    #    c += x["consumed_tokens"]
    #    r += x["remaining_tokens"]
    #    p += x["produced_tokens"]

    #print(f"TBR Inductive Miner (n=0.0)")
    #print(f"{m},{c},{r},{p}")

    #conf = pm4py.conformance.conformance_diagnostics_alignments(log, net, im , fm, multi_processing=True)
    #fitness = pm4py.conformance.fitness_alignments(log, net, im, fm, multi_processing=True)

    #fitness_values = set()
    #costs = set()
    #manual_costs = set()
    #for x in conf:
    #    manual_costs.add(get_alignment_cost(x))
    #    fitness_values.add(x["fitness"])
    #    costs.add(x["cost"])

    #print(f"Alignments Inductive Miner (n=0.0)")
    #print(fitness_values)
    #print(costs)
    #print(manual_costs)
    #print(fitness["lof_fitness"])

    net,im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
    #pm4py.view_petri_net(net,im,fm)

    #variants = {}
    #for trace in log:
    #    # print(trace)
    #    events = " - ".join([x["concept:name"] for x in trace])
    #    if events in variants:
    #        variants[events] = (trace, variants[events][1] + 1)
    #    else:
    #        variants[events] = (trace, 1)

    #manual_costs = set()
    #total_costs = 0
    #for events_string, variant in tqdm(variants.items(), "Aligning Variants manually one by one"):
    #    alignment = pm4py.conformance.conformance_diagnostics_alignments(variant[0], net, im, fm)
    #    alignment_cost = get_alignment_cost(alignment)
    #    manual_costs.add(alignment_cost)
    #    total_costs += alignment_cost

    #print(f"Manual alignments Inductive Miner (n=0.2)")
    #print(manual_costs)
    #print(f"Total Cost: {total_costs}")
    #conf = pm4py.conformance.conformance_diagnostics_token_based_replay(log, net,im,fm)
    #m = c = r = p = 0
    #for x in conf:
    #    m += x["missing_tokens"]
    #    c += x["consumed_tokens"]
    #    r += x["remaining_tokens"]
    #    p += x["produced_tokens"]

    #print(f"TBR Inductive Miner (n=0.2)")
    #print(f"{m},{c},{r},{p}")
    shortest_path = algorithm.apply_trace(obj.Trace(), net, im, fm)
    shortest_path = len([tup for tup in shortest_path["alignment"] if tup[1] is not None])

    values = []

    conf = pm4py.conformance.conformance_diagnostics_alignments(log, net, im , fm, multi_processing=True)

    total_cost = 0
    total_trace_length = 0

    for trace, c in zip(log, conf):
        total_cost += get_alignment_cost(c)
        total_trace_length += len(trace)

        trace_string = " - ".join([x["concept:name"] for x in trace])
        values.append({"trace": trace_string,
                       "cost": get_alignment_cost(c),
                       "cost_pm4py": c["cost"],
                       "cost_pm4py_normalized": c["cost"] % 10000,
                       "trace_length": len(trace),
                       "shortest_path": shortest_path,
                       "fitness": (c["cost"] % 10000)/(len(trace)+shortest_path)})

    print(f"Fitness: {1 - (total_cost/(total_trace_length+ len(log)*shortest_path))}")

    for v in values:
        v["total_cost"] = total_cost
        v["total_trace_length"] = total_trace_length
        v["log_length"] = len(log)
        v["total_fitness"] = 1 - (total_cost/(total_trace_length+ len(log)*shortest_path))

    pd.DataFrame(values).to_csv("./"+ INPUT +"_conformance.csv")

if __name__ == "__main__":
    main()