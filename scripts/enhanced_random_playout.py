import math
import random
import csv
import os
from collections import deque
from typing import Dict, Optional

import pm4py
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.obj import Trace, Event
from pm4py.objects.petri_net.semantics import enabled_transitions, execute
from pm4py.util.xes_constants import DEFAULT_NAME_KEY as NAME


def _variant_key(labels):
    return tuple(labels)


def _trace_from_labels(labels):
    tr = Trace()
    for a in labels:
        tr.append(Event({NAME: a}))
    return tr


def _safe_execute(t, pn, m):
    try:
        return execute(t, pn, m)
    except TypeError:
        return execute(pn, m, t)


def _marking_key(marking):
    # create marking key so identical markings can be identified
    return tuple(sorted((p.name, c) for p, c in marking.items()))


def _new_ngrams_from_suffix(labels, new_label, n_values=(2, 3, 4)):
    # returns all new ngrams created by adding the new label
    seq = labels + [new_label]
    out = []
    for n in n_values:
        if len(seq) >= n:
            out.append((n, tuple(seq[-n:])))
    return out


def _compute_score(*,
                   lab,
                   new_marking,
                   labels_prefix,
                   seen_activities,
                   seen_ngrams,
                   final_marking,
                   weight_activity,
                   ngram_weights,
                   rnd_jitter=True,
                   rand_weight=0.05):

    added_act = 0
    added_ngram_score = 0.0

    if lab is not None:
        added_act = 1 if lab not in seen_activities else 0
        for n, gram in _new_ngrams_from_suffix(labels_prefix, lab, n_values=tuple(ngram_weights.keys())):
            if gram not in seen_ngrams:
                added_ngram_score += ngram_weights.get(n, 0.0)

    score = weight_activity * added_act + added_ngram_score

    if new_marking == final_marking:
        score += 0.0

    if rnd_jitter:
        score += rand_weight * random.random()

    return score


# tau closure that expands transitions until a labelled transition is found
def _labeled_successors(petri_net, start_marking, final_marking=None):
    q = deque([start_marking])
    visited = {_marking_key(start_marking)}
    labeled_succ = []
    final_in_tau = False

    while q:
        m = q.popleft()

        if final_marking is not None and m == final_marking:
            final_in_tau = True

        en = list(enabled_transitions(petri_net, m))

        # append labeled transitions to output list
        for t in en:
            if t.label is not None:
                new_m = _safe_execute(t, petri_net, m)
                labeled_succ.append((new_m, t, t.label))

        # expand τ-Transitions until a known marking or a labeled transition is reached
        for t in en:
            if t.label is None:
                m2 = _safe_execute(t, petri_net, m)
                k = _marking_key(m2)
                if k not in visited:
                    visited.add(k)
                    q.append(m2)

    return labeled_succ, final_in_tau


# Samples k candidates with gumbel noise
def _gumbel_top_k(cands, k, temperature=1.0):
    def sample_gumbel():
        # prevent u==0
        u = random.random()
        while u <= 0.0:
            u = random.random()
        return -math.log(-math.log(u))

    inv_temp = 1.0 / max(1e-12, temperature)
    perturbed = []
    for triplet in cands:
        s = triplet[2]
        y = s * inv_temp + sample_gumbel()
        perturbed.append((y, triplet))

    perturbed.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in perturbed[:k]]


# Beam search, either with classic top k pruning (beam) or with gumbel-top-k sampling
def beam_playout_unique_variants(petri_net,
                                 initial_marking,
                                 final_marking,
                                 target_unique: int,
                                 beam_width: int = 100,
                                 max_trace_length: int = 100,
                                 max_rounds: int = 50,
                                 rnd_seed: int = 13,
                                 *,
                                 selection_mode: str = "gumbel",  # "beam" or "gumbel"
                                 gumbel_temperature: float = 1.0,  # only for selection_mode="gumbel"
                                 max_traces_per_depth: int = 5,
                                 max_traces_per_round: int = 100,
                                 weight_activity: float = 1.0,
                                 ngram_weights: Optional[Dict[int, float]] = None,
                                 csv_path: Optional[str] = None
                                 ) -> EventLog:
    """
    Generates up to target_unique unique trace variants via beam-like search.
    Selection of candidates per depth:
      - selection_mode="beam": classic top-k by score
      - selection_mode="gumbel": Gumbel top-k (stochastic, exploratory)

    If csv_path is provided, writes one row per pruning step:
      pruning_step, round, depth, num_candidates, beam_width, num_pruned
    """

    # use random jitter for the beam score to break ties
    if selection_mode == "beam":
        score_rnd_jitter = True
    else:
        score_rnd_jitter = False

    mode_print_str = selection_mode.upper() + "-Playout"

    if ngram_weights is None:
        ngram_weights = {2: 0.5, 3: 1 / 3, 4: 0.25}

    if target_unique <= 0:
        print("[" + mode_print_str + "] Zielanzahl ist 0 – gebe leeren Log zurück.")
        return EventLog()

    random.seed(rnd_seed)
    beam_width = max(2, beam_width)

    # cap pro Depth für "continue"
    per_depth_cap = max_traces_per_depth

    simulated_log = EventLog()
    seen_variants = set()
    seen_activities = set()
    seen_ngrams = set()   # save tuples of Labels, e.g. ('A','B','C')

    rounds = 0

    # --- CSV logging setup ---
    pruning_step = 0
    csv_writer = None
    csv_fh = None
    if csv_path is not None:
        csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow([
            "pruning_step",
            "round",
            "depth",
            "num_candidates",
            "beam_width",
            "num_pruned",
            "total_completed",
            "num_seen_variants"
        ])

    # -------------------------

    try:
        while len(seen_variants) < target_unique and rounds < max_rounds:
            rounds += 1
            print(f"[{mode_print_str}] Runde {rounds}/{max_rounds} – bisher {len(seen_variants)}/{target_unique} Varianten.")
            beam = [(initial_marking, [], 0.0)]

            # completed nach Tiefe sammeln
            completed_by_depth = {}  # depth -> list[list[str]]

            for depth in range(1, max_trace_length + 1):
                # Wenn für diese Tiefe schon genug completed vorhanden sind -> nächste Tiefe
                if len(completed_by_depth.get(depth, [])) >= per_depth_cap:
                    continue

                candidates = []

                for marking, labels, bonus in beam:
                    # 1) τ-Closure to all labeled successors
                    succs, final_in_tau = _labeled_successors(
                        petri_net, marking, final_marking=final_marking
                    )

                    # fm in τ-Closure -> completed (unverändert), aber jetzt by depth (mit per_depth_cap)
                    if final_in_tau:
                        dkey = len(labels)
                        if len(completed_by_depth.get(dkey, [])) < per_depth_cap:
                            completed_by_depth.setdefault(dkey, []).append(labels)

                    # 2) add one new label per depth step
                    for new_marking, t, lab in succs:
                        new_labels = labels if lab is None else labels + [lab]

                        # Nachfolger ist final -> NICHT in candidates, sondern completed (cap beachten)
                        if new_marking == final_marking:
                            dkey = len(new_labels)
                            if len(completed_by_depth.get(dkey, [])) < per_depth_cap:
                                completed_by_depth.setdefault(dkey, []).append(new_labels)
                            continue

                        score = _compute_score(
                            lab=lab,
                            new_marking=new_marking,
                            labels_prefix=labels,
                            seen_activities=seen_activities,
                            seen_ngrams=seen_ngrams,
                            final_marking=final_marking,
                            weight_activity=weight_activity,
                            ngram_weights=ngram_weights,
                            rnd_jitter=score_rnd_jitter,
                            rand_weight=0.05
                        ) + bonus

                        candidates.append((new_marking, new_labels, score))

                # wenn gar nichts mehr geht: abbrechen
                total_completed = sum(len(v) for v in completed_by_depth.values())

                # Runde beenden, sobald insgesamt genug completed gesammelt sind
                if total_completed >= max_traces_per_round:
                    break
                if not candidates:
                    break

                # 3) Pruning
                if selection_mode.lower() == "gumbel":
                    beam = _gumbel_top_k(candidates, beam_width, temperature=gumbel_temperature)
                else:
                    candidates.sort(key=lambda x: x[2], reverse=True)
                    beam = candidates[:beam_width]

                # --- CSV logging per pruning step ---
                if csv_writer is not None:
                    pruning_step += 1
                    num_candidates = len(candidates)
                    num_kept = len(beam)
                    num_pruned = num_candidates - num_kept
                    total_completed = sum(len(v) for v in completed_by_depth.values())
                    csv_writer.writerow([
                        pruning_step,
                        rounds,
                        depth,
                        num_candidates,
                        beam_width,
                        num_pruned,
                        total_completed,
                        len(seen_variants)
                    ])
                # ------------------------------------

                # optional: wenn beam leer ist, bringt weitere Tiefe nichts
                if not beam:
                    break

            # process completed paths from depth loop: Round-Robin über Depths (kurz UND lang)
            added_now = 0

            # process ALL traces in completed_by_depth
            completed_ordered = []
            for d in sorted(completed_by_depth.keys()):
                bucket = completed_by_depth[d]
                random.shuffle(bucket)
                completed_ordered.extend(bucket)

            for labels in completed_ordered:
                if len(labels) > max_trace_length:
                    continue
                key = _variant_key(labels)
                if key in seen_variants:
                    continue

                # update seen activities and ngrams. This affects the score computation in the coming rounds
                for lab in labels:
                    seen_activities.add(lab)
                L = len(labels)
                for n in (2, 3, 4):
                    if L >= n:
                        for i in range(L - n + 1):
                            seen_ngrams.add(tuple(labels[i:i + n]))

                seen_variants.add(key)
                simulated_log.append(_trace_from_labels(labels))
                added_now += 1

                if len(seen_variants) >= target_unique:
                    break

            print(f"[{mode_print_str}] Runde {rounds} abgeschlossen: +{added_now} neue Varianten, gesamt {len(seen_variants)}.")

            if len(seen_variants) >= target_unique:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"[{mode_print_str}] Ziel erreicht: {target_unique} einzigartige Varianten nach {rounds} Runden.")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                break

        if len(seen_variants) < target_unique:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"[{mode_print_str}] Ziel NICHT erreicht: {len(seen_variants)}/{target_unique} Varianten nach {rounds} Runden.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # write unique variants to output
        uniq = {}
        for tr in simulated_log:
            k = tuple(ev[NAME] for ev in tr)
            if k not in uniq:
                uniq[k] = tr

        out_log = EventLog(list(uniq.values()))
        if len(out_log) > target_unique:
            out_log = EventLog(list(out_log)[:target_unique])

        return out_log

    finally:
        if csv_fh is not None:
            csv_fh.close()


def basic_playout_unique_variants(petri_net, initial_marking, final_marking, target_unique: int,
                                  batch_size: int = 5000,
                                  max_trace_length: int = 100,
                                  max_rounds: int = 200
                                  ) -> EventLog:
    """
    Repeatedly performs BASIC playout and filters with variants_filter.get_variants
    until exactly `target_unique` unique trace variants are reached.
    Returns an EventLog containing only these unique variants.
    """

    mode_print_str = "random".upper() + "-Playout"

    if target_unique <= 0:
        return EventLog()

    simulated_log = EventLog()
    rounds = 0

    while True:
        # 1) create new batch
        batch = pm4py.algo.simulation.playout.petri_net.algorithm.basic_playout.apply(
            petri_net,
            initial_marking,
            final_marking,
            parameters={
                "noTraces": batch_size,
                "maxTraceLength": max_trace_length,
                "add_only_if_fm_is_reached": True
            }
        )
        # 2) append to existing log
        for trace in batch:
            simulated_log.append(trace)

        # 3) filter for unique variants
        variants_dict = variants_filter.get_variants(simulated_log)
        unique_traces = [traces[0] for traces in variants_dict.values()]

        simulated_log = EventLog(unique_traces)

        # 4) check wether target is reached or max rounds exceeded
        if len(simulated_log) >= target_unique:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("[" + mode_print_str + "] Ziel von " + str(target_unique) + " einzigartigen Traces erreicht in Runde " + str(rounds))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # cut to desired length
            return EventLog(list(simulated_log)[:target_unique])

        rounds += 1
        if rounds >= max_rounds:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"{mode_print_str} Nur {len(simulated_log)} Varianten gefunden – Ziel {target_unique} nicht erreicht ")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return simulated_log
