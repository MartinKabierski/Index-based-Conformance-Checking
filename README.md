# IBF Alignment Evaluation Pipeline

## Overview


This repository provides the implementation and evaluation scripts needed to reproduce 
the results reported in the paper "Efficient Approximation of Alignments based on Indexed Proxy-Behavior"

The project investigates how well alignments can be approximated with an **IBF-based method** (Interleaved Bloom Filter). Instead of aligning an event log directly against a Petri net, a **playout** is generated from the model (the model language), and the original log is compared against these model traces.

The following serve as comparison:

- **optimal A\* alignments** from PM4Py (exact baseline)
- a **trie-based method** for approximate alignment computation

First, the models and the baseline are generated; afterwards, the actual experiments and plots run on top of them (see [Typical Workflow](#typical-workflow)).

---

## Requirements

### Required

- Python 3.9+ with `pm4py`, `pandas`, `numpy`, `tqdm`, `matplotlib`...([requirements.txt](requirements.txt)) 
- The compiled C++ programs `alignment_txt` and `xes_to_txt` are located in the `scripts` folder.
  Read [here](c++/README.md#requirements) how to build them.
- The event log to be analyzed is located as `.xes` in the `logs` folder
- `scripts/params_config.json` is located in the `scripts/` folder
- The scripts are started **from within `scripts`** (binaries and `scripts/params_config.json` are looked up relative to the working directory)


### Optional: Trie-Based Approach

Only needed for `scripts/ibf_vs_trie_based_comparison.py`. With `SKIP_TRIE_BASED = True` it is skipped.

If it is to be used, make sure that:

- Java 8 (openjdk-8-jre-headless) is installed
- the compiled classes are located under `scripts/trieBasedClasses`
- `PATH_TO_JAVA` and `PATH_TO_TRIE_RUNNER` at the top of `scripts/ibf_vs_trie_based_comparison.py` are correct

The wrapper expects the class `Runner` to accept the two log paths as arguments and to output lines of the form `Time taken ... <n> milliseconds` and `Overall fitness = <x>`.
You can find the original trie-based repository [here](https://github.com/DataSystemsGroupUT/ConformanceCheckingUsingTries/). In order to be able to call the trie-based approach as a subprocess, some changes to the Runner class were necessary; you can view them [here](scripts/trieBasedClasses/Runner.java).

---

## Directory Structure

```text
project/
├── c++/                      # c++ code for the ibf 
├── logs/                     # original logs (.xes)
├── models/                   # discovered Petri nets (.pnml)
├── opt_alignments/           # optimal model traces from A*
├── output/                   # result CSVs
│   ├── temp/                 # intermediate files (kept in repo via .gitkeep)
│   └── plots/
│
└── scripts/                  # all scripts, working directory
    ├── params_config.json
    ├── alignment_txt
    ├── xes_to_txt
    ├── ...
    └── trieBasedClasses/     # optional


```

---

## Configuration

All settings are defined as constants **at the top of the respective script**. To be adjusted in each script:

- the log name (`LOG_NAME` or `log_name`)
- the noise thresholds for model discovery (`NOISE_THRESHOLDS`, default: `[0.8, 0.2]`)
- the number of repetitions (`REPETITIONS`)
- the number of parallel workers (`MAX_PARALLEL_REPETITIONS`, `None` = automatic)


Script-specific settings (playout modes, target sizes, timeouts, etc.) are described in the individual [program flows](ARCHITECTURE.md#program-flows).

---

## Alignment Parameters

The parameters for the IBF search with `alignment_txt` are defined in `scripts/params_config.json`. They control:

- how many buckets are created
- how traces of the model language are distributed across buckets
- how traces are processed during the IBF search

Structure:

- `defaults`: value lists for all parameters
- `bucketing`: overrides of the defaults per bucketing method
- `enabled_bucketings` (optional): which methods are used; if the entry is missing, all are used

`scripts/alignment_params_factory.py` merges defaults and overrides for each active method and forms the Cartesian product.



Which scripts use which combinations:

| Script | combinations used |
|---|---|
| `scripts/ibf_vs_trie_based_comparison.py` | **all** |
| `scripts/evaluate_playout_quality.py` | one (`IBF_PARAMS`, otherwise the first) |
| `scripts/plot_preparation_variants_runtime_ibf.py` | one (`IBF_PARAMS`, otherwise the first) |

### Parameter Values

| Parameter | Value | Meaning |
|---|---|---|
| `bucketing` | `i` | each model trace gets its own bucket |
| | `random` | model traces are randomly distributed across the buckets |
| | `omh` | model traces are distributed across the buckets via Order-Min-Hash |
| `buckets_calc` | `n` | number of buckets = number of model traces |
| | `sqrt` | number of buckets = √(number of model traces) |
| `sorting` | `hitcount` | buckets are searched sorted by their hit count |
| `k` | number | length of the k-mers |
| `num_hashes` | number | number of hash functions for lsh bucketing (minhash) |
| `omh_w` | number | parameter *w* of the Order-Min-Hash |
| `omh_seed` | number | seed of the Order-Min-Hash |
| `bucket_limit` | number | maximum number of searched buckets, `0` = no limit |

`omh_w` and `omh_seed` only take effect with `bucketing = omh` and are ignored otherwise. In the result files they are then set to `0`.

### Important Note on `bucket_limit`

- **`bucket_limit = 0`**
  - The computed log fitness does not depend on the alignment parameters
  - The fitness depends solely on the quality of the generated model language

- **`bucket_limit > 0`**
  - The runtime decreases because fewer buckets are considered
  - In return, the determined fitness may decrease

---

## Runtime Notes

⚠️ **Parameter combinations grow quickly.**

All options in `scripts/params_config.json` are combined as a full Cartesian product. In `ibf_vs_trie_based_comparison.py`, this product is additionally multiplied by noise values, target sizes and repetitions. This can lead to very long runtimes.

The same applies to `scripts/evaluate_playout_quality.py`: the number of playout configurations results from target sizes × beam widths × groups × n-gram sizes.

### Recommendation

- Start with small values, e.g.:
  ```
  REPETITIONS = 1
  TARGET_VARIANTS_LIST = [1000]
  ```
- Only increase them once correctness and runtime have been verified on small runs.
- The A\* alignment in `scripts/discover_models_and_opt_alignments.py` can require a lot of RAM. In that case, set `ALIGN_VARIANT` to `VERSION_DIJKSTRA_LESS_MEMORY` or lower `MAX_PARALLEL_REPETITIONS`.

---

## Typical Workflow

1. Place the event log in `logs`
2. Provide/adjust `alignment_txt`, `xes_to_txt` and `params_config.json` in `scripts/`
3. Set `LOG_NAME` in all scripts
4. Generate models and A\* baseline:
   `scripts/discover_models_and_opt_alignments.py`
5. Compute baseline fitness:
   `scripts/calculate_baseline_fitness.py` or, if alignments ran into the timeout,
   `calculate_baseline_fitness_for_partially_unsolved_alignments.py`
6. Run experiments (independent of each other):
   - `scripts/evaluate_playout_quality.py` – comparison of the playout methods
   - `scripts/ibf_vs_trie_based_comparison.py` – IBF vs. Trie
   - `scripts/plot_preparation_variants_runtime_ibf.py` – IBF runtimes per variant
7. Generate runtime plots:
   `scripts/plot_variant_runtimes_ibf_vs_astar.py`

```text
discover_models_and_opt_alignments.py
│   -> models/, opt_alignments/, <LOG>_all_noise_conformance_variant_runtimes.csv
│
├── calculate_baseline_fitness*.py
│      -> fitness_baseline_all_noise_<LOG>.csv
│      │
│      └── evaluate_playout_quality.py      (uses the baseline for the MAE)
│
├── ibf_vs_trie_based_comparison.py
│
└── plot_preparation_variants_runtime_ibf.py
       -> IBF_alignment_<LOG>_all_noise_all_variants.csv
       │
       └── plot_variant_runtimes_ibf_vs_astar.py   (+ A* CSV from discover_models_and_opt_alignments.py)
```

---


## Output Columns

### IBF/Trie Comparison (`Comparison_IBF_vs_TRIE_on_playout_with_opt_alignments_<LOG>.csv`)

Each row corresponds to **one IBF parameter combination** within a repetition (noise value × target size × repetition × parameter combination). Values that are computed only once per repetition (playout, log sizes, trie results) are therefore identical in all rows of that repetition.

All times are given in **milliseconds**.

#### Experiment

| Column | Meaning |
|---|---|
| `noise` | noise threshold of the Inductive Miner with which the model was discovered |
| `repetition` | number of the repetition (starting at 0) |
| `playout_mode` | playout method used (always `random` here) |
| `random_seed` | seed of this repetition (`RANDOM_SEED + repetition`) |
| `target_variants` | targeted number of unique variants in the playout |
| `shortest_path` | number of visible transitions in the shortest complete model path |

#### Log Sizes

"Traces" counts all traces, "variants" only unique activity sequences.

| Column | Meaning |
|---|---|
| `playout_traces` | traces in the raw Random Playout |
| `playout_variants` | variants in the raw Random Playout. Can be smaller than `target_variants` if the model does not allow more variants within the maximum trace length or `max_rounds` was reached |
| `random_log_traces` | traces in the Random Log (playout + shortest visible trace) |
| `random_log_variants` | variants in the Random Log. Equals `playout_variants` if the shortest trace was already contained in the playout |
| `opt_alignment_traces` | number of optimal model traces from A\* (one per variant of the original log) |
| `opt_alignment_variants` | of these, the unique model traces (several log variants can be aligned to the same model trace) |
| `mixed_log_traces` | traces in the Mixed Log (`playout_traces` + `opt_alignment_traces` + 1) |
| `mixed_log_variants` | variants in the Mixed Log. Equals `random_log_variants` if all optimal traces were already contained in the playout |
| `original_log_traces` | traces in the original log |
| `original_log_variants` | variants in the original log |

#### IBF Parameters

The values are passed unchanged as options to `alignment_txt`. The meaning of the individual values is described under [Parameter Values](#parameter-values).


| Column | Meaning |
|---|---|
| `bucketing` | method by which model traces are distributed across buckets (`i`, `random`, `omh`) |
| `bucket_limit` | maximum number of searched buckets; `0` = no limit (see [Note on `bucket_limit`](#important-note-on-bucket_limit)) |
| `k` | length of the k-mers |
| `buckets_calc` | how the number of buckets is calculated (`n`, `sqrt`) |
| `num_hashes` | number of hash functions (only for `lsh`, otherwise `0`) |
| `omh_w` | parameter *w* of the Order-Min-Hash (only for `omh`, otherwise `0`) |
| `omh_seed` | seed of the Order-Min-Hash (only for `omh`, otherwise `0`) |
| `sorting` | order in which buckets are searched (`hitcount`: sorted by hit count) |


#### IBF Results

Each metric appears twice: with prefix `random_ibf_` for the comparison *Original Log <-> Random Log* and with `mixed_ibf_` for *Original Log <-> Mixed Log*.

| Column (without prefix) | Meaning |
|---|---|
| `fitness` | log fitness: `1 − total_cost_adjusted / (total_length_all_traces + log_number_traces · shortest_path)` |
| `total_cost` | sum of the alignment costs (Levenshtein distance with insertion/deletion) over all traces of the original log |
| `total_cost_adjusted` | like `total_cost`, but capped per trace at `shortest_path + trace length` (cost of an alignment consisting only of log and model moves) |
| `total_length_all_traces` | sum of the lengths of all traces of the original log |
| `log_number_traces` | number of evaluated traces of the original log |
| `number_trace_variants` | number of variants of the original log in the evaluation |
| `total_variant_cost` | sum of the costs, each variant counted only once |
| `mean_variant_cost` | `total_variant_cost / number_trace_variants` |
| `variant_cost_inconsistencies` | number of variants that received different costs for different occurrences (expected: `0`) |
| `trace_calculation_time` | sum of the per-trace computation times from the alignment CSV |
| `creation_time` | time to build the IBF from the model traces (reported by `alignment_txt`) |
| `search_time` | total time of the search in the IBF (reported by `alignment_txt`). Greater than `trace_calculation_time` because some traces are searched multiple times |

#### Trie Results (KristoR)

| Column | Meaning |
|---|---|
| `trie_random_fitness` | fitness reported by the trie method, *Original Log <-> Random Log* |
| `trie_random_alignment_time_ms` | search time reported by the trie method, *Original Log <-> Random Log* |
| `trie_mixed_fitness` | as above, *Original Log <-> Mixed Log (Random Log + optimal alignments)* |
| `trie_mixed_alignment_time_ms` | as above, *Original Log <-> Mixed Log (Random Log + optimal alignments)* |

With `SKIP_TRIE_BASED = True`, all four columns are `0.0`.

#### Runtimes

| Column | Meaning |
|---|---|
| `playout_time_ms` | duration of the Random Playout |
| `random_log_xes_write_time_ms` | writing the Random Log as XES |
| `random_log_xes_to_txt_time_ms` | conversion of the Random Log to TXT (reported by `xes_to_txt`) |
| `mixed_log_xes_write_time_ms` | writing the Mixed Log as XES |
| `mixed_log_xes_to_txt_time_ms` | conversion of the Mixed Log to TXT |
| `ibf_parameter_run_time_ms` | wall-clock time for **both** (random, mixed) IBF runs of this parameter combination, incl. process startup, reading input and fitness computation |
| `total_experiment_time_ms` | time from the start of the repetition to the end of **this** row |

⚠️ `total_experiment_time_ms` is cumulative: the time includes all parameter combinations processed earlier in this repetition and therefore increases from row to row.

---
