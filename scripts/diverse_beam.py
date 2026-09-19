import random
import csv
import os
from collections import deque, Counter
from typing import Dict, Optional, List, Tuple

from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.obj import Trace, Event
from pm4py.objects.petri_net.semantics import enabled_transitions, execute
from pm4py.util.xes_constants import DEFAULT_NAME_KEY as NAME


def _variant_key(labels: List[str]) -> Tuple[str, ...]:
    return tuple(labels)


def _trace_from_labels(labels: List[str]) -> Trace:
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
    return tuple(sorted((p.name, c) for p, c in marking.items()))


def _new_ngrams_from_suffix(labels, new_label, n_values=(2, 3, 4)):
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

        for t in en:
            if t.label is not None:
                new_m = _safe_execute(t, petri_net, m)
                labeled_succ.append((new_m, t, t.label))

        for t in en:
            if t.label is None:
                m2 = _safe_execute(t, petri_net, m)
                k = _marking_key(m2)
                if k not in visited:
                    visited.add(k)
                    q.append(m2)

    return labeled_succ, final_in_tau


def _suffix_ngram(labels_prefix: List[str], new_label: str, n: int) -> Optional[Tuple[str, ...]]:
    seq = labels_prefix + [new_label]
    if len(seq) < n:
        return None
    return tuple(seq[-n:])


def _dbs_step_diversity_bonus_ngram(
    ngram: Optional[Tuple[str, ...]],
    prev_group_ngram_counts: Counter
) -> int:
    if ngram is None:
        return 0
    total_prev = sum(prev_group_ngram_counts.values())
    same = prev_group_ngram_counts.get(ngram, 0)
    return total_prev - same


def dbs_beam_playout_unique_variants(
        petri_net,
        initial_marking,
        final_marking,
        target_unique: int,
        beam_width: int = 100,
        max_trace_length: int = 100,
        max_rounds: int = 50,
        rnd_seed: int = 13,
        *,
        num_groups: int = 10,
        lambda_div: float = 0.5,
        max_traces_per_depth: int = 5,
        max_traces_per_round: int = 100,
        weight_activity: float = 1.0,
        ngram_weights: Optional[Dict[int, float]] = None,
        score_rnd_jitter: bool = True,
        max_diversity_ngram: int = 2,
        csv_path: Optional[str] = None
) -> EventLog:

    mode_print_str = "DBS-PLAYOUT"

    if ngram_weights is None:
        ngram_weights = {2: 0.5, 3: 1 / 3, 4: 0.25}

    max_diversity_ngram = max(1, max_diversity_ngram)

    if target_unique <= 0:
        print(f"[{mode_print_str}] Zielanzahl ist 0 – gebe leeren Log zurück.")
        return EventLog()

    random.seed(rnd_seed)
    beam_width = max(2, beam_width)

    num_groups = max(1, min(num_groups, beam_width))
    for g in range(num_groups, 0, -1):
        if beam_width % g == 0:
            num_groups = g
            break
    group_beam_width = beam_width // num_groups

    per_depth_cap = max(1, max_traces_per_depth)

    simulated_log = EventLog()
    seen_variants = set()
    seen_activities = set()
    seen_ngrams = set()

    rounds = 0

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
            "g",
            "num_candidates",
            "group_beam_width",
            "num_pruned",
            "total_completed",
            "num_seen_variants"
        ])

    try:
        while len(seen_variants) < target_unique and rounds < max_rounds:
            rounds += 1
            print(f"[{mode_print_str}] Runde {rounds}/{max_rounds} – bisher {len(seen_variants)}/{target_unique} Varianten.")

            beams = [[(initial_marking, [], 0.0)] for _ in range(num_groups)]
            completed_by_depth: Dict[int, List[List[str]]] = {}

            for depth in range(1, max_trace_length + 1):
                if len(completed_by_depth.get(depth, [])) >= per_depth_cap:
                    continue

                prev_group_ngram_counts = {
                    n: Counter() for n in range(1, max_diversity_ngram + 1)
                }

                for g in range(num_groups):
                    candidates_g = []

                    for marking, labels, bonus in beams[g]:
                        succs, final_in_tau = _labeled_successors(
                            petri_net, marking, final_marking
                        )

                        if final_in_tau:
                            dkey = len(labels)
                            if len(completed_by_depth.get(dkey, [])) < per_depth_cap:
                                completed_by_depth.setdefault(dkey, []).append(labels)

                        for new_marking, _, lab in succs:
                            new_labels = labels + [lab]

                            if new_marking == final_marking:
                                dkey = len(new_labels)
                                if len(completed_by_depth.get(dkey, [])) < per_depth_cap:
                                    completed_by_depth.setdefault(dkey, []).append(new_labels)
                                continue

                            base_score = _compute_score(
                                lab=lab,
                                new_marking=new_marking,
                                labels_prefix=labels,
                                seen_activities=seen_activities,
                                seen_ngrams=seen_ngrams,
                                final_marking=final_marking,
                                weight_activity=weight_activity,
                                ngram_weights=ngram_weights,
                                rnd_jitter=score_rnd_jitter
                            )

                            div_bonus = 0.0
                            if g > 0 and lambda_div > 0:
                                base_score = 0.0
                                total_ngram_div_bonus = 0.0

                                for n in range(1, max_diversity_ngram + 1):
                                    ngram = _suffix_ngram(labels, lab, n)
                                    total_ngram_div_bonus += _dbs_step_diversity_bonus_ngram(
                                        ngram,
                                        prev_group_ngram_counts[n]
                                    )

                                div_bonus = lambda_div * total_ngram_div_bonus

                            candidates_g.append((new_marking, new_labels, bonus + base_score + div_bonus))

                    if not candidates_g:
                        beams[g] = []
                        continue

                    candidates_g.sort(key=lambda x: x[2], reverse=True)
                    beams[g] = candidates_g[:group_beam_width]

                    if csv_writer is not None:
                        pruning_step += 1
                        total_completed = sum(len(v) for v in completed_by_depth.values())
                        csv_writer.writerow([
                            pruning_step,
                            rounds,
                            depth,
                            g,
                            len(candidates_g),
                            group_beam_width,
                            len(candidates_g) - len(beams[g]),
                            total_completed,
                            len(seen_variants)
                        ])

                    for _, lab_seq, _ in beams[g]:
                        for n in range(1, max_diversity_ngram + 1):
                            if len(lab_seq) >= n:
                                prev_group_ngram_counts[n][tuple(lab_seq[-n:])] += 1

                total_completed = sum(len(v) for v in completed_by_depth.values())
                if total_completed >= max_traces_per_round:
                    break
                if all(len(bg) == 0 for bg in beams):
                    break

            added_now = 0
            for labels in sum(completed_by_depth.values(), []):
                key = _variant_key(labels)
                if key in seen_variants:
                    continue

                for lab in labels:
                    seen_activities.add(lab)

                for n in (2, 3, 4):
                    for i in range(len(labels) - n + 1):
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

        return EventLog(list({tuple(ev[NAME] for ev in tr): tr for tr in simulated_log}.values()))

    finally:
        if csv_fh is not None:
            csv_fh.close()