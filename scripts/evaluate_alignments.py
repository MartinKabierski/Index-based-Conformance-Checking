import time

import pandas
import pm4py
from tqdm import tqdm

#LOG_NAMES = ["Sepsis_Cases_-_Event_Log.xes"]
LOG_NAMES = ["BPI_Challenge_2012.xes"]
#LOG_NAMES = ["Road_Traffic_Fines_Management_Process.xes"]
REPETITIONS = 5



def get_alignment_cost(a):
    #print([1 if (step[1] == ">>" or (step[0] == ">>" and step[1] is not None)) else 0 for step in a["alignment"]])
    return sum([1 if (step[1] == ">>" or (step[0] == ">>" and step[1] is not None)) else 0 for step in a["alignment"]])


stats = []

for name in LOG_NAMES:
    log = pm4py.read_xes("../logs/"+ name, return_legacy_log_object=True)

    net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)

    variants = {}
    for trace in log:
        #print(trace)
        events = " - ".join([x["concept:name"] for x in trace])
        if events in variants:
            variants[events] = (trace, variants[events][1] + 1)
        else:
            variants[events] = (trace, 1)


    #alignment = pm4py.conformance.fitness_alignments(log, net, im, fm)
    #print(alignment)


    for r in range (0,REPETITIONS):

        for events_string, variant in tqdm(variants.items(),"Aligning Variants. Repetition "+str(r)):
            t1 = time.time()
            alignment = pm4py.conformance.conformance_diagnostics_alignments(variant[0], net, im, fm)
            #print(alignment)
            t2 = time.time()
            length = len([x["concept:name"] for x in variant[0]])
            alignment_cost = get_alignment_cost(alignment)
            for i in range (0,variant[1]):
                stats.append({"repetition": r, "variant": events_string, "length":length, "alignment_time": t2-t1, "cost": alignment_cost, "fitness": alignment["fitness"]})

    df = pandas.DataFrame(stats)
    print(df)
    df.to_csv("../output/A_Star_alignments_"+name+".csv")
