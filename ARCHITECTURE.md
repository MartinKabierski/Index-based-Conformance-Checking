## Program Flows

### Model Discovery and A\* Alignment (Baseline)([🔗discover_models_and_opt_alignments.py](scripts/discover_models_and_opt_alignments.py))

Key settings: `ALIGN_VARIANT` (A\* with state equation or memory-saving Dijkstra) and `ALIGN_TIMEOUT` (seconds per variant).

```text
main()
│
├── Preparation
│   ├── Create directories
│   ├── Delete old combined result CSV
│   ├── Read original log
│   └── Determine number of workers
│        └── min(REPETITIONS, cpu_count, MAX_PARALLEL_REPETITIONS)
│
├── Start ProcessPoolExecutor  (spawn, once for the entire run)
│   └── _init_worker(log_path)
│        └── Read XES log once per worker process
│
├── For each noise value
│   │
│   ├── Discover Petri net (Inductive Miner)
│   ├── Export model as PNML
│   ├── Determine shortest_path (align empty trace)
│   │
│   ├── Start repetitions in parallel
│   │   │
│   │   └── run_repetition(rep=0 … n)
│   │        │
│   │        ├── log = _WORKER_LOG          (already loaded)
│   │        ├── _get_model(model_path)     (cached per process)
│   │        │
│   │        └── for each trace in the log
│   │             │
│   │             ├── Known variant?
│   │             │    └── count += 1, skip
│   │             │
│   │             └── New variant
│   │                  ├── algorithm.apply_trace()  (with timeout)
│   │                  ├── Measure runtime
│   │                  │
│   │                  ├── Timeout / error
│   │                  │    └── Row with status="unsolved", empty costs
│   │                  │
│   │                  └── Successful
│   │                       ├── get_alignment_cost()
│   │                       ├── extract_optimal_model_trace()
│   │                       ├── Compute fitness (own + pm4py)
│   │                       └── Row with status="ok"
│   │        │
│   │        ├── Append variant frequency to all rows
│   │        └── return (values, optimal_model_traces)
│   │
│   └── as_completed(...)
│        │
│        └── Main process
│             ├── append_results_csv(values) to the result CSV
│             └── Write optimal_model_traces.txt (only first completed repetition)
│
└── finalize_outputs()
     ├── Collect unsolved variants (unsolved in any row)
     ├── Write <LOG>_unsolved_variants.csv
     ├── Write result CSV without these variants  (..._without_unsolved.csv)
     └── Write filtered log                        (logs/<LOG>_without_unsolved.xes)
```

Cost = number of log moves plus visible model moves.

---

### Baseline Fitness ([🔗 calculate_baseline_fitness.py](scripts/calculate_baseline_fitness.py), [🔗 calculate_baseline_fitness_for_partially_unsolved_alignments.py](scripts/calculate_baseline_fitness_for_partially_unsolved_alignments.py))

Both scripts aggregate the A\* CSV into fitness per noise value and repetition and append an `AVG` row per noise value:

```text
fitness = 1 − Σ(cost · count) / (Σ(trace_length · count) + number_traces · shortest_path)
```

[`calculate_baseline_fitness.py`](scripts/calculate_baseline_fitness.py) assumes that all variants were solved. The second script handles unsolved variants according to `UNSOLVED_POLICY`:


| Policy | Cost for unsolved variants | Effect on fitness |
|---|---|---|
| `worst_case` (default) | `trace_length` (log moves only, required model moves ignored) | estimate; despite its name **not** a guaranteed lower bound (see note) |
| `best_case` | `0` | guaranteed upper bound, for sanity checking only |
| `exclude` | row is left out entirely, i.e. this variant in this repetition only (numerator and denominator) | ⚠️ fitness only covers the solved variants; usually biased upward, since timeouts tend to hit long, strongly deviating traces |

**Note on `worst_case`:** Log moves alone are not a complete alignment, because the model run still has to reach its final marking. The optimal cost of an unsolved variant lies anywhere between `0` and `trace_length + shortest_path`, so the assumed `trace_length` can be too low as well as too high (e.g. model A→B→C, trace ⟨A⟩: optimal cost 2, assumed cost 1). The true fitness is guaranteed to lie between `worst_case − Δ` and `best_case`, with

`Δ = unsolved_traces · shortest_path / (Σ(trace_length · count) + number_traces · shortest_path)`

where `unsolved_traces` is the number of traces (not variants) of unsolved variants for the same noise value and repetition (for `AVG` rows: averaged over the repetitions). The `Affected traces` value printed by the script is summed over all noise values and repetitions; plugging it in still yields a valid but looser bound.


Coverage and fitness under all three policies are printed to the console.

---

### Comparison of Playout Methods ([🔗 evaluate_playout_quality.py](scripts/evaluate_playout_quality.py))

Key settings: `playout_modes` (`random`, `beam`, `gumbel`, `dbs`), `target_variants_list`, `beam_widths`, `numbers_of_groups`, `max_diversity_ngrams`, `IBF_PARAMS`.

```text
main()
│
├── Preparation
│   ├── Create result CSV
│   ├── Convert original log to TXT once (if not present)
│   ├── Determine IBF parameters (exactly one combination)
│   ├── Load A* total_cost + total_variant_cost from the baseline CSV
│   └── Determine number of workers
│
├── For each noise value
│   ├── Load model + optimal alignment traces
│   ├── Determine shortest_visible_trace
│   ├── max_trace_length = longest optimal trace + shortest_path
│   └── Start ProcessPoolExecutor
│
└── Run repetitions in parallel
    │
    └── run_repetition(rep=0 … n)
         │
         ├── Seed = BASE_SEED + repetition
         ├── Load model for this repetition
         │
         └── for config in iter_playout_configs()
              │
              └── run_parameter_combination(...)
                   │
                   ├── Set seed for this configuration
                   ├── run_playout(config, ...)
                   │    ├── random      -> basic_playout_unique_variants()
                   │    ├── dbs         -> dbs_beam_playout_unique_variants()
                   │    └── beam/gumbel -> beam_playout_unique_variants()
                   │
                   ├── Measure runtime
                   ├── Count hits with optimal alignment traces
                   ├── IBF: Original Log <-> Playout
                   ├── MAE against the A* baseline
                   │    ├── mae_by_variant_count  (per number of trace variants in the log)
                   │    └── mae_by_trace_count    (per number of traces in the log)
                   │
                   └── Return one CSV row
         │
         └── return rows
              │
              V
         Main process
              └── writerows(rows) to playout_vs_opt_alignments_<LOG>.csv
```

Both MAE values (mean alignment error against the A\* baseline) are computed from columns of the result CSV:

- `mae_by_variant_count = (ibf_total_variant_cost - astar_total_variant_cost) / variant_count`
- `mae_by_trace_count = (ibf_total_cost - astar_total_cost) / ibf_number_traces`

`total_variant_cost` counts the alignment cost once per trace variant, while `total_cost` weights it by the variant frequency. The denominators are the number of trace variants and the number of traces in the original log. The A\* values come from the AVG rows of `fitness_baseline_all_noise_<LOG>.csv`, written by `calculate_baseline_fitness.py` or `calculate_baseline_fitness_for_partially_unsolved_alignments.py`. The two MAEs are computed independently: if an A\* value is missing for a noise level, only the corresponding MAE stays empty and a warning is printed.

---

### IBF/Trie Comparison ([🔗 ibf_vs_trie_based_comparison.py](scripts/ibf_vs_trie_based_comparison.py))

Key settings: `TARGET_VARIANTS_LIST`, `SKIP_TRIE_BASED`, `PATH_TO_JAVA`, `PATH_TO_TRIE_RUNNER`.

```text
Repetition starts  (per noise value and target size)
│
├─ 1. Generate Random Playout
│     ├─ Random playout traces
│     ├─ Append shortest visible trace as LAST trace
│     ├─ Save Random Log as XES
│     └─ Save Random Log as TXT
│
├─ 2. Generate Mixed Playout
│     ├─ Random playout traces
│     ├─ Add optimal alignment traces
│     ├─ Shuffle
│     ├─ Append shortest visible trace as LAST trace
│     ├─ Save Mixed Log as XES
│     └─ Save Mixed Log as TXT
│
├─ 3. KristoR (Trie)
│     ├─ Original Log <-> Random Playout
│     │     -> Fitness
│     │     -> Internal search runtime
│     │
│     └─ Original Log <-> Mixed Playout
│           -> Fitness
│           -> Internal search runtime
│
└─ 4. For each IBF parameter combination
      │
      ├─ IBF: Original Log <-> Random Playout
      │     -> Fitness, cost, variant-based cost
      │     -> Creation and search times
      │
      ├─ IBF: Original Log <-> Mixed Playout
      │     -> Fitness, cost, variant-based cost
      │     -> Creation and search times
      │
      └─ Exactly ONE shared result row
            for this parameter combination
```

The Mixed Log is guaranteed to contain the optimal traces. It shows how close both methods get to the A\* baseline when the model language is "good enough".

---

### Preparation of the Runtime Plots ([🔗 plot_preparation_variants_runtime_ibf.py](scripts/plot_preparation_variants_runtime_ibf.py))

Generates the IBF runtimes **per trace variant** so that they can be compared with the A\* runtimes. Only the Mixed Log and exactly one IBF parameter combination are used.

Key settings: `RANDOM_PLAYOUT_TRACES_TARGET`, `IBF_PARAMS`.

```text
main()
│
├── Preparation
│   ├── Check inputs (original log, output/temp/ present)
│   ├── Convert original log to TXT once
│   ├── Determine IBF parameters (exactly one combination)
│   └── Determine number of workers
│
├── For each noise value
│   ├── Load model + optimal alignment traces
│   ├── Determine shortest_visible_trace
│   └── Start ProcessPoolExecutor
│
└── Run repetitions in parallel
    │
    └── run_repetition(rep=0 … n)
         │
         ├── 1. Generate Random Playout
         ├── 2. Build Mixed Log (playout + optimal traces + shuffle
         │                       + shortest trace at the end)
         ├── Save Mixed Log as XES/TXT in output/temp/
         ├── 3. IBF: Original Log <-> Mixed Log
         ├── Read alignment CSV from alignment_txt
         ├── Prepend noise, repetition, mixed_log_traces to each row
         └── return (header, rows, temp_files)
              │
              V
         Main process
              ├── Append rows to IBF_alignment_<LOG>_all_noise_all_variants.csv
              └── Delete temp files
```

The result CSV is semicolon-separated and contains all rows of the alignment output, i.e. also repeated variants (time `0`, empty field `searched buckets`).

---

### Runtime Plots A\* vs. IBF ([🔗 plot_variant_runtimes_ibf_vs_astar.py](scripts/plot_variant_runtimes_ibf_vs_astar.py))

Generates a scatter plot per noise value: x = trace length, y = runtime per variant (in seconds, averaged over all repetitions). Two plots are created per noise value, one with a linear and one with a logarithmic y-axis.

```text
main()
│
├── find_files()
│    └── Search for A* and IBF CSV (if missing: list available CSVs)
│
├── load() per method
│    ├── Read only required columns
│    ├── IBF: only actually computed rows  (searched buckets filled)
│    ├── Convert time to seconds
│    ├── Take trace length or count it from the variant
│    └── Mean per variant over the repetitions
│
└── For each noise value
     ├── Print statistics (count, median, mean, max)
     └── plot_noise() for linear and log
          ├── log: remove runtimes = 0 (with notice)
          ├── Draw points, optional median line
          ├── x-axis starting at 0, tick at the longest variant
          ├── fit_ylabel(): shrink y-label if needed
          └── output/plots/<LOG>_noise_<n>_<linear|log>.<format>
```

Both input files are expected in `DATA_DIR` (default: `output`): the A\* CSV from [`discover_models_and_opt_alignments.py`](scripts/discover_models_and_opt_alignments.py) and `IBF_alignment_<LOG>_all_noise_all_variants.csv` from [`plot_preparation_variants_runtime_ibf.py`](scripts/plot_preparation_variants_runtime_ibf.py).

---

### Helper Modules

- **[`alignment_params_factory.py`](scripts/alignment_params_factory.py)** – generates all IBF parameter combinations from [`params_config.json`](scripts/params_config.json) (see [Alignment Parameters](README.md#alignment-parameters)).
- **[`subprocess_wrapper.py`](scripts/subprocess_wrapper.py)** – calls `xes_to_txt` and `alignment_txt` as subprocesses and reads the runtimes (Bloom filter creation, search) from the output.
- **[`alignment_txt_fitness_wrapper.py`](scripts/alignment_txt_fitness_wrapper.py)** – starts an IBF run and computes fitness, cost and variant-based metrics from the alignment CSV. Costs are capped per trace at `shortest_path + trace_length`. Variants with differing costs are reported as a warning.
- **[`subprocess_run_triebasedkristor.py`](scripts/subprocess_run_triebasedkristor.py)** – starts the Java `Runner`, builds the classpath itself and reads fitness and runtime from the output.
- **[`enhanced_random_playout.py`](scripts/enhanced_random_playout.py)** – random playout with unique variants as well as beam search playout with top-k or Gumbel-top-k selection. New activities and new n-grams are rewarded in the scoring.
- **[`diverse_beam.py`](scripts/diverse_beam.py)** – Diverse Beam Search: the beam is split into groups. All groups except the first are scored with a diversity bonus relative to the previous groups (`lambda_div`, `max_diversity_ngram`).

---

### IBF Alignment Search ([🔗 alignment_txt.cpp](c++/index-basiertes-conformance-checking/source/alignment_txt.cpp))

Inputs/outputs: `--traces` model traces TXT, `--search` log TXT, `--output` result CSV. Both TXT files contain one trace per line with events separated by `" - "`.

Key settings: `--kmer`, `--bucketing` (`omh`, `lsh`, `random`, `tracelen`, `set`, `multiset`, `i`), `--number_buckets_calc` (`n`, `sqrt`), `--sorting` (`score`, `score2`, `hitcount`), `--bucket_limit`, `--shortest_path`, `--num_hashes` (lsh), `--omh_w` and `--omh_seed` (omh), `--seed` (MurmurHash3).

```text
main()
│
├── Parse CLI arguments
├── Select execution mode
│   └── NORMAL: --traces + --search + --output
│       (--dump / --load for IBF serialization are not implemented yet)
│
└── search_alignments(...)
    │
    ├── Preparation
    │   ├── Count model traces n (lines in --traces)
    │   ├── number_buckets = n or sqrt(n)
    │   └── Create IBF: number_buckets bins, 2048 bits per bin, 2 hash functions
    │
    ├── build_interleaved_bloom_filter(...)
    │   │
    │   └── for each model trace
    │        ├── Split into events
    │        ├── Build unique k-mers (traces shorter than k are padded with "__")
    │        ├── Assign bucket
    │        │    ├── omh      -> omh_bucket_index()
    │        │    ├── lsh      -> get_bucket_index_from_kmers()
    │        │    ├── random   -> random bucket
    │        │    ├── tracelen -> bucket = trace length (≥ number_buckets -> bucket 0)
    │        │    ├── set      -> one bucket per distinct k-mer set
    │        │    ├── multiset -> one bucket per distinct k-mer multiset
    │        │    └── i        -> one bucket per trace
    │        │
    │        └── Insert the MurmurHash3 hash of every k-mer into the bucket's bin
    │
    ├── Print IBF generation time
    ├── export_bucketlist_to_csv()
    │   └── ../output/temp/bucketlist_<bucketing>_<num_hashes>_<buckets>_k_<k>.csv
    │
    ├── search_in_interleaved_bloom_filter(...)
    │   │
    │   └── for each log trace
    │        │
    │        ├── Duplicate trace? -> write cached result, next trace
    │        ├── Split into events, build unique k-mers, hash them
    │        ├── IBF bulk_count -> k-mer hits per bucket
    │        ├── Sort buckets by --sorting
    │        │    ├── score    -> hits / traces in bucket
    │        │    ├── score2   -> hits / k-mers of the log trace
    │        │    └── hitcount -> hits
    │        │
    │        ├── Initial best match = last model trace (shortest visible trace)
    │        │
    │        ├── for bucket in sorted buckets
    │        │    ├── First non-empty bucket is always checked
    │        │    ├── Skip if hits < #k-mers − k · best cost (hitcount sorting: stop)
    │        │    ├── Levenshtein against every trace in the bucket, abort at best cost
    │        │    ├── Cost 0 -> exact match, stop
    │        │    └── Stop after --bucket_limit checked buckets (0 = no limit)
    │        │
    │        └── Write CSV row
    │
    └── Print IBF search time
```

The Levenshtein distance uses cost 1 for insertions and deletions and cost 2 for substitutions. This equals the alignment cost against the found model trace when log moves and model moves cost 1. Only the model traces in the file are considered, so the result is an upper bound of the optimal alignment cost. A model trace with cost c shares at least #k-mers − k · c k-mers with the log trace, so the hit threshold never skips a bucket that could contain a better match. With `--bucket_limit 0`, bucketing and sorting therefore only affect the runtime, not the costs.

The model trace file must end with the shortest visible model trace, and `--shortest_path` must be its length. This trace is the initial best match for every log trace, with trace length + shortest_path as the cost bound. [`evaluate_playout_quality.py`](scripts/evaluate_playout_quality.py) appends it automatically (`APPEND_SHORTEST_TRACE`).

The result CSV is `;`-separated with the columns `given trace`, `found trace`, `levenshtein distance`, `calculation time [ms]`, `searched buckets` and `compared traces`.

---


### Conventions

- **TXT format:** one trace per line, activities separated by ` - `.
- **Shortest trace:** The IBF tool expects the shortest visible model trace as the **last** trace of the model trace file. The scripts append it and verify this.
- **File names:** Noise values are written with `p` instead of a dot, e.g. `0p2`.
- **Parallelization:** Parallelization is done over the repetitions. Only the main process writes results.

---
