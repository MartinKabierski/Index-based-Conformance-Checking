// SeqAn3 includes
#include <seqan3/argument_parser/all.hpp>
#include <seqan3/core/debug_stream.hpp>
#include <seqan3/search/dream_index/interleaved_bloom_filter.hpp>
#include <seqan3/alphabet/nucleotide/dna4.hpp>
#include <seqan3/search/views/kmer_hash.hpp>

// includes for serialization / deserialization seqan3
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

// includes for reading traces from log file
#include "pugixml.hpp"
#include <iostream>

//include for random number genration
#include <random>

//include for levenshtein abort
#include <unordered_set>

// include for time measurement
#include <chrono>

// reading from a text file
#include <fstream>

// include MurMurHash3 hashing algorithm
#include <algorithm>
#include <map>

#include "MurmurHash3.h"

#include <string_view>
#include <vector>
#include <unordered_map>
#include <cstring>   // std::memcpy
#include <cassert>



// buckets for trace indices
using Bucket = std::vector<size_t>;
using BucketList = std::vector<Bucket>;


// enum for differentiate between different program executions
enum class Execution_Types { SERIALIZE, DESERIALIZE, NORMAL };


/**
 * @brief Serialize ibf into a file
 *
 * @param os Output stream of file
 * @param ibf Interleaved Bloom Filter object
 */
// at the moment not implemented. To really facilitate a serialization, the supporting data structures (bucket lists) also need to be included
void serialize(std::ofstream & os, seqan3::interleaved_bloom_filter<> & ibf)
{
    cereal::BinaryOutputArchive archive(os);
    archive(ibf);
}

/**
 * @brief Deserialize ibf from file
 *
 * @param is Input stream of file
 * @param ibf Interleaved Bloom Filter object
 */
//this is currently not in use, as the serialization is not implemented
void deserialize(std::ifstream & is, seqan3::interleaved_bloom_filter<> & ibf)
{
    cereal::BinaryInputArchive archive(is);
    archive(ibf);
}

// Builds all unique contiguous k-mers from a sequence of events.
//
// Input:
//   - events: ordered list of event tokens as std::string_view
//   - k: length of each k-mer
//   - delimiter: string inserted between adjacent events in the output
//   - pad_token: token used when the trace is shorter than k
//
// Behavior:
//   - If events is empty, returns an empty result.
//   - If events.size() < k, the sequence is padded with pad_token until length k,
//     so exactly one k-mer can still be produced.
//   - If events.size() >= k, all sliding windows of size k are generated.
//   - Duplicate k-mers are removed while preserving first occurrence order.
inline std::vector<std::string>
build_kmers_fast_unique(const std::vector<std::string_view>& events,
                        std::size_t                          k,
                        std::string_view                     delimiter = " - ",
                        std::string_view                     pad_token = "__")
{
    std::vector<std::string> kmers;
    if (events.empty()) return kmers;

    std::vector<std::string_view> ev;
    if (events.size() < k) {
        ev = events;
        ev.reserve(k);
        while (ev.size() < k) ev.emplace_back(pad_token);
    } else {
        ev = events;
    }

    const std::size_t n         = ev.size();
    // Number of sliding windows:
    //   - exactly 1 when the original sequence was shorter than k and has been padded
    //   - otherwise n - k + 1 windows
    const std::size_t kmer_cnt  = (n < k) ? 1 : n - k + 1;
    const std::size_t dlen      = delimiter.size();

    // Reserve memory up front to reduce reallocations
    kmers.reserve(kmer_cnt);
    std::unordered_set<std::string> seen;
    seen.reserve(kmer_cnt * 2);

    /* Prefix sum of token lengths (O(n)) ----------------------
       prefix[i] stores the total number of characters in ev[0..i-1].
       This allows constant-time computation of the total payload length of any
       window ev[i..i+k-1]:
           prefix[i + k] - prefix[i]
       That makes it possible to allocate each output string with exact size
       before copying data into it.
    ------------------------------------------------------------------------ */
    std::vector<std::size_t> prefix(n + 1);
    prefix[0] = 0;
    for (std::size_t i = 0; i < n; ++i)
        prefix[i + 1] = prefix[i] + ev[i].size();

    /* Build k-mer strings and deduplicate ----------------------
       For each sliding window:
         1. Compute exact output size
         2. Allocate one std::string of that size
         3. Copy tokens and delimiters directly into the buffer
         4. Insert into hash set and keep only first occurrence in result
    ------------------------------------------------------------------------ */
    for (std::size_t i = 0; i < kmer_cnt; ++i) {
        // Total output size for this k-mer:
        //   sum of token lengths in the current window
        //   + delimiter length repeated between the k tokens
        std::size_t bytes = prefix[i + k] - prefix[i] + dlen * (k - 1);

        // Allocate the full output buffer once.
        // The string is pre-sized so we can write directly into its storage.
        std::string out(bytes, '\0');
        char* p = out.data();

        for (std::size_t j = 0; j < k; ++j) {
            auto sv = ev[i + j];

            // Copy the current token into the output buffer.
            std::memcpy(p, sv.data(), sv.size());
            p += sv.size();

            // Insert delimiter between tokens, but not after the last token.
            if (j + 1 < k) {
                std::memcpy(p, delimiter.data(), dlen);
                p += dlen;
            }
        }

        // Insert into the deduplication set.
        // Only k-mers that were not seen before are appended to the result vector.
        // This preserves the order of first appearance.
        auto [it, inserted] = seen.insert(out);
        if (inserted) kmers.emplace_back(std::move(out));
    }

    return kmers;
}


// Wrapper converting std::string inputs to std::string_view to reuse the core implementation without copying.
inline std::vector<std::string>
build_kmers_fast_unique(const std::vector<std::string>& events,
                        std::size_t                     k,
                        const std::string&              delimiter = " - ",
                        const std::string&              pad_token = "__")
{
    std::vector<std::string_view> views;
    views.reserve(events.size());
    for (auto& s : events) views.emplace_back(s);
    return build_kmers_fast_unique(views, k, delimiter, pad_token);
}


// Splits a delimited input line into individual event tokens.
//
// Input:
// - line: the full input string containing events separated by `delim`
// - delim: separator between events (default: " - ")
//
// Behavior:
// - Returns a vector of std::string_view, each referencing a token in `line`
// - No string copies are made; all views point into the original input
// - The lifetime of `line` must outlive the returned views
// - If no delimiter is found, the entire line is returned as a single element
inline std::vector<std::string_view>
split_events_fast(std::string_view line,
                  std::string_view delim = " - ")
{
    // Preallocate a small default capacity to reduce reallocations for typical inputs
    constexpr std::size_t RESERVE_HINT = 16;
    std::vector<std::string_view> out;
    out.reserve(RESERVE_HINT);

    std::size_t start = 0;
    while (true) {
        // Find next occurrence of the delimiter starting from `start`
        std::size_t pos = line.find(delim, start);

        // If no more delimiters are found, append the remaining substring (last token)
        if (pos == std::string_view::npos) {
            out.emplace_back(line.substr(start));
            break;
        }

        // Append substring between current position and delimiter
        out.emplace_back(line.substr(start, pos - start));

        // Move start index to the first character after the delimiter
        start = pos + delim.size();
    }

    // Return result; typically optimized via NRVO (no copy)
    return out;
}


// Takes the string by value (i.e. a copy), trims it and returns the result.
/* Was required to prevent errors in Levenshtein distance calculations
    caused by inconsistent line separators (from different operating systems)
    altering the last event of a trace when files were opened manually. */
std::string remove_line_separator(std::string s) {
    // Remove all \n and \r at the end
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}


// sorts a kmer vector and removes duplicates
std::vector<std::string> make_kmer_set(std::vector<std::string> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}


// Builds a multiset of contiguous k-mers from an ordered event sequence.
//
// Input:
// - events: ordered event tokens as std::string_view
// - k: number of events per k-mer, must be > 0
// - delimiter: inserted between events when building the k-mer string
// - pad_token: used to pad short traces so one full k-mer can still be created
//
// Behavior:
// - Returns (kmer, count) pairs instead of unique k-mers only
// - If events.size() < k, the sequence is padded to length k
// - Counts how often each generated k-mer occurs
// - Returns the result sorted lexicographically by k-mer
//
// Purpose:
// - Used where k-mer frequency matters, e.g. for multiset-based bucketing or comparison
inline std::vector<std::pair<std::string, int>>
make_kmer_multiset(const std::vector<std::string_view>& events,
                   std::size_t                          k,
                   std::string_view                     delimiter = " - ",
                   std::string_view                     pad_token = "__")
{
    std::vector<std::pair<std::string, int>> result;
    if (events.empty()) return result;
    assert(k > 0 && "k must be > 0");

    // Pad short input so at least one complete k-mer can be generated.
    std::vector<std::string_view> ev = events;
    if (ev.size() < k) {
        ev.reserve(k);
        while (ev.size() < k) ev.emplace_back(pad_token);
    }

    // guaranteed: n >= k after padding
    const std::size_t n        = ev.size();
    const std::size_t dlen     = delimiter.size();

    // number of contiguous sliding windows
    const std::size_t kmer_cnt = n - k + 1;

    // Prefix sums allow exact output-size calculation for each k-mer in O(1).
    std::vector<std::size_t> prefix(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i)
        prefix[i + 1] = prefix[i] + ev[i].size();

    // Count occurrences of each generated k-mer.
    std::unordered_map<std::string, int> counts;
    counts.reserve(kmer_cnt);

    for (std::size_t i = 0; i < kmer_cnt; ++i) {
        const std::size_t bytes = (prefix[i + k] - prefix[i]) + dlen * (k - 1);

        // Allocate the final string once with exact size and fill it directly.
        std::string out(bytes, '\0');
        char* p = out.data();

        for (std::size_t j = 0; j < k; ++j) {
            auto sv = ev[i + j];
            std::memcpy(p, sv.data(), sv.size());
            p += sv.size();
            if (j + 1 < k) {
                std::memcpy(p, delimiter.data(), dlen);
                p += dlen;
            }
        }

        // Insert new k-mer with count 0, or reuse the existing entry, then increment.
        auto [it, inserted] = counts.try_emplace(std::move(out), 0);
        ++(it->second);
    }

    // Convert hash map output into a sorted vector
    result.reserve(counts.size());
    for (auto& kv : counts)
        result.emplace_back(std::move(kv.first), kv.second);

    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });

    return result;
}


// Wrapper converting std::string inputs to std::string_view to reuse the core implementation without copying.
inline std::vector<std::pair<std::string, int>>
make_kmer_multiset(const std::vector<std::string>& events,
                   std::size_t                     k,
                   const std::string&              delimiter = " - ",
                   const std::string&              pad_token = "__")
{
    std::vector<std::string_view> views;
    views.reserve(events.size());
    for (const auto& s : events) views.emplace_back(s);
    return make_kmer_multiset(views, k, delimiter, pad_token);
}



class KmerSetBucketFinder {
public:
    // Assigns a integer bucket ID to a k-mer set.
    //
    // Input:
    // - s: sorted vector of unique k-mers (typically from make_kmer_set)
    //
    // Behavior:
    // - If this exact set was seen before, returns the existing bucket ID
    // - Otherwise assigns a new ID and stores the mapping
    //
    // Purpose:
    // - Groups identical k-mer sets into the same bucket for fast comparison or indexing
    int bucket_for_set(const std::vector<std::string>& s) {
        auto [it, inserted] = set2bucket.emplace(s, next_id);
        if (inserted) return next_id++;
        return it->second;
    }

    // Assigns a integer bucket ID to a k-mer multiset (with counts).
    //
    // Input:
    // - ms: sorted vector of (kmer, count) pairs (from make_kmer_multiset)
    //
    // Behavior:
    // - If this exact multiset was seen before, returns the existing bucket ID
    // - Otherwise assigns a new ID and stores the mapping
    //
    // Purpose:
    // - Enables grouping of sequences based on frequency-aware k-mer representation
    int bucket_for_multiset(const std::vector<std::pair<std::string,int>>& ms) {
        auto [it, inserted] = mset2bucket.emplace(ms, next_id);
        if (inserted) return next_id++;
        return it->second;
    }

private:
    // Monotonically increasing ID assigned to new unique inputs
    int next_id = 0;

    // Maps a k-mer set to its assigned bucket ID
    // std::map is used to allow vector comparison as key (lexicographical order)
    std::map<std::vector<std::string>, int> set2bucket;

    // Maps a k-mer multiset (with counts) to its assigned bucket ID
    std::map<std::vector<std::pair<std::string,int>>, int> mset2bucket;
};


/**
 * Calculates the Levenshtein distance between two event traces with early termination.
 *
 * Input:
 * - trace1, trace2: sequences of events (already tokenized, not raw strings)
 * - cost_threshold: maximum allowed distance before computation is aborted
 *
 * Behavior:
 * - Computes edit distance with insertion/deletion cost = 1 and substitution cost = 2
 * - Terminates early if it is guaranteed that the final distance will be >= cost_threshold
 * - Returns the exact distance if below threshold, otherwise returns cost_threshold
 *
 * Purpose:
 * - Optimized for performance in scenarios where only small distances are relevant
 * - Avoids unnecessary computation for clearly dissimilar traces
 */
unsigned int levenshtein_distance_with_threshold(const std::vector<std::string>& trace1, const std::vector<std::string>& trace2, unsigned int cost_threshold)
{

    // Lengths of both traces
    const std::size_t len1 = trace1.size(), len2 = trace2.size();

    // Early exit: if length difference alone exceeds threshold, distance cannot be smaller
    if (std::abs(static_cast<int>(len1) - static_cast<int>(len2)) >= static_cast<int>(cost_threshold))
        return cost_threshold;

    // Early exit: count how many unique symbols differ between traces
    // If too many completely different events exist, distance must exceed threshold
    std::unordered_set<std::string> set1(trace1.begin(), trace1.end());
    std::unordered_set<std::string> set2(trace2.begin(), trace2.end());

    unsigned int symbol_diff_count = 0;

    for (const auto& e : set1){
        if (set2.find(e) == set2.end())
            ++symbol_diff_count;
    }

    for (const auto& e : set2){
        if (set1.find(e) == set1.end())
            ++symbol_diff_count;
    }

    if (symbol_diff_count >= cost_threshold)
        return cost_threshold;

    // d[i][j] = edit distance between first i elements of trace1 and first j elements of trace2
    std::vector<std::vector<unsigned int>> d(len1 + 1, std::vector<unsigned int>(len2 + 1));

    // Initialize base cases (transforming from/to empty sequence)
    d[0][0] = 0;

    for(unsigned int i = 1; i <= len1; ++i)
        d[i][0] = i;

    for(unsigned int i = 1; i <= len2; ++i)
        d[0][i] = i;

    // Fill matrix row by row
    for(unsigned int i = 1; i <= len1; ++i)
    {
        // Tracks if this row still contains values below threshold
        // If not, computation can be aborted early
        bool row_has_viable_value = false;

        for(unsigned int j = 1; j <= len2; ++j)
        {
            // Substitution cost: 0 if equal, otherwise 2 (heavier penalty than insert/delete)
            unsigned int cost = (trace1[i - 1] == trace2[j - 1]) ? 0 : 2;

            // Standard Levenshtein recurrence:
            // min(delete, insert, substitute)
            d[i][j] = std::min({
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost
            });

            // Check if any value in this row is still below threshold
            if (d[i][j] < cost_threshold)
                row_has_viable_value = true;
        }

        // If entire row exceeds threshold, no need to continue
        if (!row_has_viable_value)
            return cost_threshold;
    }

    // Return final edit distance
    return d[len1][len2];
}


/**
 * Joins a sequence of event strings into a single delimited trace string.
 *
 * Input:
 * - trace: ordered vector of event strings
 * - delimiter: string inserted between consecutive events
 *
 * Behavior:
 * - Returns a single concatenated string
 * - If trace is empty, returns an empty string
 * - Preserves original order of events
 *
 */
std::string join_trace_fast(const std::vector<std::string>& trace, const std::string& delimiter)
{
    if (trace.empty()) return "";

    // Compute exact total size: sum of all event lengths + delimiters between them
    // This allows a single memory allocation for the final string
    std::size_t total_size = (trace.size() - 1) * delimiter.size();
    for (const auto& s : trace) total_size += s.size();

    std::string result;

    // Reserve full capacity up front to avoid dynamic resizing during concatenation
    result.reserve(total_size);

    // Append first element without delimiter
    result += trace[0];

    // Append remaining elements with delimiter in between
    for (std::size_t i = 1; i < trace.size(); ++i)
    {
        result += delimiter;
        result += trace[i];
    }

    // Return final concatenated string
    return result;
}


/**
 * Writes a list of buckets (grouped trace indices) to a CSV file.
 *
 * Input:
 * - buckets: vector of buckets, each bucket contains indices of traces belonging to that group
 * - filename: output file path
 *
 * Behavior:
 * - Creates/overwrites a CSV file
 * - Each row represents one bucket
 * - Outputs bucket ID, number of traces, and trace indices
 * - Trace indices are written as a semicolon-separated list within one CSV field
 *
 */
void export_bucketlist_to_csv(const std::vector<std::vector<size_t>> &buckets, const std::string &filename) {
    std::ofstream file(filename);

    // Check if file could be opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << "\n";
        return;
    }

    // Write CSV header
    file << "Bucket,TraceCount,TraceIndices\n";

    // Iterate over all buckets
    for (size_t i = 0; i < buckets.size(); ++i) {
        const auto &bucket = buckets[i];

        // Write bucket ID and number of traces
        file << i << "," << bucket.size() << ",";

        // Write all trace indices as a semicolon-separated list inside one CSV column
        for (size_t j = 0; j < bucket.size(); ++j) {
            file << bucket[j];
            if (j < bucket.size() - 1)
                file << ";";
        }

        // End of CSV row
        file << "\n";
    }

    // Close file
    file.close();

    std::cout << "BucketList exported to: " << filename << "\n";
}


/**
 * Computes a hash value for a k-mer combined with a seed.
 *
 * Input:
 * - kmer: string representation of a k-mer
 * - seed: integer used to generate different hash variants
 *
 * Behavior:
 * - Concatenates kmer and seed, then hashes the result
 * - Produces different hash values for the same k-mer depending on the seed
 *
 * Purpose:
 * - Used to simulate multiple independent hash functions for MinHash
 */
uint64_t hash_kmer_with_seed(const std::string &kmer, size_t seed) {
    return std::hash<std::string>{}(kmer + std::to_string(seed));
}


/**
 * Computes a bucket index for a set of k-mers using MinHash + LSH aggregation.
 *
 * Input:
 * - kmers: list of k-mers representing a trace
 * - num_hashes: number of hash functions (MinHash signature size)
 * - num_buckets: total number of buckets
 *
 * Behavior:
 * - Builds a MinHash signature by taking the minimum hash per hash function
 * - Combines the signature into a single value using XOR
 * - Maps the result into a bucket index via modulo
 * - Returns 0 if kmers is empty
 *
 * Purpose:
 * - Provides a fast approximate similarity grouping (LSH)
 * - Similar traces (with similar k-mers) are likely assigned to the same bucket
 */
size_t get_bucket_index_from_kmers(const std::vector<std::string> &kmers,
                                   size_t num_hashes,
                                   size_t num_buckets)
{
    // Handle empty input by assigning a default bucket
    if (kmers.empty())
        return 0;

    // Initialize MinHash signature with maximum values
    // Each position corresponds to one hash function
    std::vector<uint64_t> min_hashes(num_hashes, std::numeric_limits<uint64_t>::max());

    // For each k-mer, compute all hash variants and keep the minimum per position
    for (const auto &kmer : kmers)
    {
        for (size_t i = 0; i < num_hashes; ++i)
        {
            uint64_t h = hash_kmer_with_seed(kmer, i);

            // Keep smallest hash value (MinHash principle)
            if (h < min_hashes[i])
                min_hashes[i] = h;
        }
    }

    // Combine MinHash signature into a single value
    // XOR is used as a simple aggregation function
    uint64_t sketch_signature = 0;
    for (auto h : min_hashes)
        sketch_signature ^= h;

    // Map aggregated signature into a bucket range
    return sketch_signature % num_buckets;
}


/**
 * Computes a bucket index from a k-mer sequence using Order Min Hash (OMH).
 *
 * Input:
 * - kmers: ordered k-mers of a trace as std::string_view
 * - bucket_cnt: total number of target buckets, must be > 0
 * - omh_w: number of smallest hashes kept for the OMH signature
 * - omh_seed: seed used to vary hash generation
 *
 * Behavior:
 * - Hashes each k-mer together with its position
 * - Keeps the omh_w smallest hash values
 * - Re-sorts the selected items by original position to preserve order information
 * - Mixes the resulting signature into one key and maps it to a bucket index
 * - Returns 0 for empty input
 *
 * Purpose:
 * - OMH is an order-sensitive variant of MinHash
 * - Similar traces with similar content and similar order are more likely to land in the same bucket
 */
inline std::size_t
omh_bucket_index(const std::vector<std::string_view>& kmers,
                 std::size_t                         bucket_cnt,
                 std::size_t                         omh_w    = 4,
                 std::uint64_t                       omh_seed = 0xCAFEBABEDEADBEEF)
{
    if (bucket_cnt == 0)
        throw std::invalid_argument("bucket_cnt must be > 0");
    if (kmers.empty())
        return 0;

    // Store one hash together with the original k-mer position.
    // The position is needed because OMH keeps order information, unlike plain MinHash.
    struct Item { std::uint64_t h; std::size_t pos; };
    std::vector<Item> items;
    items.reserve(kmers.size());

    const std::hash<std::string_view> hasher{};

    // Constant used for hash mixing.
    // This value is commonly used to decorrelate combined integer/hash inputs.
    constexpr std::uint64_t GOLD = 0x9e3779b97f4a7c15ULL;

    for (std::size_t i = 0; i < kmers.size(); ++i) {
        // Combine k-mer hash with seed and position.
        // Including the position makes the signature sensitive to ordering.
        std::uint64_t h = hasher(kmers[i])
                        ^ (omh_seed + GOLD + (i << 6) + (i >> 2));
        items.push_back({h, i});
    }

    // Keep only the omh_w smallest hashes.
    // nth_element is used so this selection is faster than fully sorting all items.
    if (items.size() > omh_w) {
        std::nth_element(items.begin(),
                         items.begin() + static_cast<long>(omh_w),
                         items.end(),
                         [](auto a, auto b){ return a.h < b.h; });
        items.resize(omh_w);
    }

    // Re-sort the selected items by original position.
    // This is the key OMH step that preserves relative order among selected k-mers.
    std::sort(items.begin(), items.end(),
              [](auto a, auto b){ return a.pos < b.pos; });

    // Fold the OMH signature into one 64-bit key.
    // FNV-1a is used here as a lightweight mixing step to combine the selected hashes.
    constexpr std::uint64_t FNV_OFF = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t FNV_PRM = 0x100000001b3ULL;

    std::uint64_t key = std::accumulate(items.begin(), items.end(), FNV_OFF,
        [](std::uint64_t acc, const Item& it){
            acc ^= it.h;
            acc *= FNV_PRM;
            return acc;
        });

    // Map the mixed signature to the configured bucket range.
    return static_cast<std::size_t>(key % bucket_cnt);
}





/**
 * Build an Interleaved Bloom Filter (IBF) and assign traces to buckets.
 *
 * Input:
 * - ibf: target Interleaved Bloom Filter instance
 * - file_traces: input stream containing one trace per line
 * - file_ibf_load_path: file path used when deserializing an existing IBF (serializing/deserializing not implemented atm)
 * - read_traces: output container storing all parsed traces as event vectors
 * - program_type: controls whether the IBF is built or deserialized (serializing/deserializing not implemented atm)
 * - k: k-mer length
 * - seed: seed for MurmurHash3
 * - kmer_count_per_bin: external per-bin statistics container
 * - verbose: enables debug output
 * - trace_buckets: output structure storing trace indices per bucket
 * - bucketing: bucket assignment strategy
 * - number_buckets: total number of buckets
 * - num_hashes: number of hash functions for LSH-based bucketing
 * - omh_w / omh_seed: parameters for OMH bucketing
 *
 * Behavior:
 * - Optionally deserializes an existing IBF from disk (serializing/deserializing not implemented atm)
 * - Reads traces line by line, splits them into events, and builds k-mers
 * - Assigns each trace to a bucket depending on the selected bucketing strategy
 * - Stores trace-to-bucket assignments and parsed trace content
 * - hashes all k-mers and inserts them into the corresponding IBF bin
 *
 * Purpose:
 * - Central preprocessing step for indexing traces into an IBF-based search structure
 */
bool build_interleaved_bloom_filter(auto & ibf, std::ifstream & file_traces, const char* file_ibf_load_path, std::vector<std::vector<std::string>>& read_traces,
    Execution_Types program_type, int k, int seed, std::vector<int> & kmer_count_per_bin, bool verbose, BucketList &trace_buckets, const std::string& bucketing, unsigned long int number_buckets, size_t num_hashes, size_t omh_w, std::uint64_t omh_seed)
{
    // Load an already-built IBF from disk when running in deserialization mode.
    // (serializing/deserializing not implemented atm)
    if (program_type == Execution_Types::DESERIALIZE)
    {
        std::cout << "Deserializing IBF from file." << std::endl;
        std::ifstream in_file (file_ibf_load_path, std::ios::binary);
        deserialize(in_file, ibf);
        std::cout << "Deserializing IBF complete!" << std::endl;
    }

    if (file_traces.is_open())
    {
        long unsigned int i = 0;

        // Random generator used only for the "random" bucketing strategy.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        // Assigns stable bucket IDs for exact set- and multiset-based bucketing.
        KmerSetBucketFinder set_bucketer;

        std::string str;
        while (std::getline(file_traces, str))
        {
            // Current trace line from the input file.
            std::string trace = str.c_str();

            // Expected delimiter between events inside a trace.
            std::string delimiter = " - ";

            // Split the trace into non-owning event views, then convert to owning strings
            // because downstream components expect std::string containers.
            auto events = split_events_fast(trace);
            std::vector<std::string> trace_events;
            trace_events.reserve(events.size());
            for (auto sv : events)
                trace_events.emplace_back(sv);

            unsigned long int trace_length = trace_events.size();

            // Build unique k-mers from the parsed event sequence.
            std::vector<std::string> kmers = build_kmers_fast_unique(trace_events, k);

            if (kmers.size() > 0)
            {
                // Default bucket assignment uses the trace index itself.
                long unsigned int bucket_index = i;

                if (bucketing == "lsh") {
                    bucket_index = get_bucket_index_from_kmers(kmers, num_hashes, number_buckets);
                }
                else if (bucketing == "random") {
                    bucket_index = static_cast<long unsigned int>(dis(gen) * number_buckets);
                }
                else if (bucketing == "tracelen") {
                    // Bucket by trace length; oversized traces are folded into bucket 0.
                    if (trace_length >= number_buckets) {
                        bucket_index = 0;
                    }
                    else {
                        bucket_index = trace_length;
                    }
                }
                else if (bucketing == "omh") {
                    // Convert k-mers to string_view so OMH can work without extra string copies.
                    auto views = [](const std::vector<std::string>& v){
                        std::vector<std::string_view> sv;
                        sv.reserve(v.size());
                        for (auto& s : v) sv.emplace_back(s);
                        return sv;
                    };

                    bucket_index = omh_bucket_index(views(kmers), number_buckets, omh_w, omh_seed);

                }
                else if (bucketing == "set") {
                    bucket_index = set_bucketer.bucket_for_set(make_kmer_set(kmers));
                }
                else if (bucketing == "multiset") {
                    bucket_index = set_bucketer.bucket_for_multiset(make_kmer_multiset(trace_events, k));
                }
                else if (bucketing == "i") {
                    // Identity bucketing keeps the default one-trace-per-bucket mapping.
                }
                else {
                    throw std::runtime_error{"no valid bucketing selected"};
                }

                // Record which trace index belongs to which bucket.
                trace_buckets[bucket_index].push_back(i);

                // Store the parsed trace for later downstream use.
                read_traces.push_back(trace_events);

                if (verbose) std::cout << "IBF generation:" << trace << std::endl;

                // Only build the IBF when building it from traces.
                // (serializing/deserializing not implemented atm)
                if (program_type != Execution_Types::DESERIALIZE)
                {
                    if (verbose) std::cout << "- " << trace << std::endl;

                    // Hash every k-mer and insert it into the bin corresponding to the assigned bucket.
                    for (int j = 0; j < kmers.size(); j++)
                    {
                        uint64_t hash[2];
                        MurmurHash3_x64_128(kmers[j].c_str(), (uint64_t)strlen(kmers[j].c_str()), seed, hash);

                        if (verbose) std::cout << "    k-mer hash: " << ((uint64_t*)hash)[0] << std::endl;

                        ibf.emplace(((uint64_t*)hash)[0], seqan3::bin_index{bucket_index});
                    }
                }
            }

            i++;
        }
        return true;
    }
    else
    {
        std::cout << "[Error] Something went wrong while reading traces file!" << std::endl;
        return false;
    }
}



// Per-bucket statistics used to rank candidate buckets during IBF search.
//
// Fields:
// - index: bucket ID
// - hit_count: number of query k-mer hashes found in this bucket
// - score: hit count normalized by number of traces in the bucket
// - score2: hit count normalized by number of query k-mers
struct BucketScoreStruct {
    size_t index;
    unsigned short hit_count;
    float score;
    float score2;
};


/**
 * Searches query traces in an Interleaved Bloom Filter and writes the best match for each query trace.
 *
 * Input:
 * - ibf: populated Interleaved Bloom Filter used as search index
 * - file_search: input stream with query traces, one trace per line
 * - file_output_path: CSV output file for search results
 * - read_traces: reference traces previously loaded from the model/input trace file
 * - search_traces: output container storing all raw query traces
 * - k: k-mer length
 * - seed: seed for MurmurHash3
 * - kmer_count_per_bin: external per-bin statistics container
 * - verbose: enables debug output
 * - trace_buckets: maps each bucket to the reference trace indices assigned to it
 * - sorting: bucket ranking mode
 * - shortest_path: shortest trace in process model added to the initial Levenshtein threshold
 * - max_checked_buckets: optional limit for how many candidate buckets are evaluated
 *
 * Behavior:
 * - Reads each query trace, splits it into events, and builds unique k-mers
 * - Hashes the query k-mers and queries the IBF for bucket hit counts
 * - Ranks buckets by the selected score
 * - Compares only promising candidate traces with thresholded Levenshtein distance
 * - Writes the best matching trace and search statistics to CSV
 * - Reuses previous results for duplicate query traces
 */
bool search_in_interleaved_bloom_filter(auto & ibf, std::ifstream & file_search, const char* file_output_path, std::vector<std::vector<std::string>>& read_traces, std::vector<std::string> & search_traces,
    int k, int seed, std::vector<int> & kmer_count_per_bin, bool verbose, const BucketList &trace_buckets, const std::string& sorting, size_t shortest_path, size_t max_checked_buckets)
{
    if (file_search.is_open())
    {
        long unsigned int trace_number = 0;

        // Cache exact duplicate query traces so repeated searches can reuse the previous result.
        std::map<std::string, std::string> already_searched_traces;

        // Output CSV containing one result row per query trace.
        std::ofstream output_csv (file_output_path);

        if (!output_csv.is_open())
        {
            std::cout << "[Error] Could not create output file!" << std::endl;
            return false;
        }

        // Output columns:
        // query trace ; best matching trace ; edit distance ; runtime ; checked buckets ; compared traces
        output_csv << "given trace;found trace;levenshtein distance;calculation time [ms];searched buckets;compared traces\n";

        if (verbose) std::cout << "IBF search:" << std::endl;

        std::string str;
        while (std::getline(file_search, str))
        {
            ++trace_number;
            std::cout << trace_number << std::endl;

            // Measure runtime for the current query trace.
            std::chrono::steady_clock::time_point begin_search = std::chrono::steady_clock::now();

            search_traces.push_back(str.c_str());

            std::string trace = str.c_str();

            if (verbose) std::cout << "- " << trace << std::endl;

            // Duplicate query handling: reuse the previously computed result immediately.
            if (already_searched_traces.find(trace) != already_searched_traces.end())
            {
                if (verbose) std::cout << "    duplicate!" << std::endl;
                std::chrono::steady_clock::time_point end_search = std::chrono::steady_clock::now();
                output_csv << "\"" << trace << "\";" << already_searched_traces[trace] << ";\"" << (std::chrono::duration_cast<std::chrono::microseconds>(end_search - begin_search).count()) / 1000.0 << "\"\n";
                continue;
            }

            // Split the query trace into event views, then materialize owning strings for downstream code.
            auto events = split_events_fast(trace);
            std::vector<std::string> trace_events;
            trace_events.reserve(events.size());
            for (auto sv : events)
                trace_events.emplace_back(sv);

            unsigned long int trace_length = trace_events.size();

            // Delimiter used later when reconstructing the best reference trace as a single string.
            std::string delimiter = " - ";

            // Build the unique k-mer representation of the query trace.
            std::vector<std::string> kmers = build_kmers_fast_unique(trace_events, k);

            // Hash all query k-mers so they can be looked up in the IBF.
            std::vector<uint64_t> hash_values;
            for (int i = 0; i < kmers.size(); i++)
            {
                // MurmurHash3 produces the hash used for IBF lookup.
                uint64_t hash[2];
                MurmurHash3_x64_128(kmers[i].c_str(), (uint64_t)strlen(kmers[i].c_str()), seed, hash);

                if (verbose) std::cout << "    k-mer hash: " << ((uint64_t*)hash)[0] << std::endl;

                // Only the first 64-bit part is used as the IBF key.
                hash_values.push_back(((uint64_t*)hash)[0]);
            }

            // Query all k-mer hashes at once and get one hit count per bucket.
            auto agent = ibf.counting_agent();
            seqan3::counting_vector<short unsigned int> results = agent.bulk_count(hash_values);

            bool all_hits_zero = true;

            // Per-bucket ranking data derived from IBF hit counts.
            std::vector<BucketScoreStruct> bucket_scores;

            for (size_t i = 0; i < results.size(); ++i) {
                if (results[i] > 0) {

                    // Score 1 favors buckets with many hits but penalizes large buckets.
                    float score_hitcount_to_tracecount = static_cast<float>(results[i]) /
                                static_cast<float>(trace_buckets[i].size());

                    // Score 2 measures how much of the query signature is supported by this bucket.
                    float score_hitcount_to_kmerscount = static_cast<float>(results[i]) /
                                static_cast<float>(hash_values.size());

                    bucket_scores.emplace_back(i, results[i], score_hitcount_to_tracecount, score_hitcount_to_kmerscount);

                    all_hits_zero = false;
                }
                else {
                    bucket_scores.emplace_back(BucketScoreStruct{ i, 0, 0.0f, 0.0f });
                }
            }

            // Sort buckets so the most promising candidates are checked first.
            if (sorting == "score") {
                std::sort(bucket_scores.begin(), bucket_scores.end(),
                    [](const BucketScoreStruct &a, const BucketScoreStruct &b) {
                        return a.score > b.score;
                    });
            }
            else if (sorting == "score2") {
                std::sort(bucket_scores.begin(), bucket_scores.end(),
                    [](const BucketScoreStruct &a, const BucketScoreStruct &b) {
                        if (a.score2 == b.score2) return a.score > b.score;
                        return a.score2 > b.score2;
                    });
            }
            else {
                std::sort(bucket_scores.begin(), bucket_scores.end(),
                    [](const BucketScoreStruct &a, const BucketScoreStruct &b) {
                        return a.hit_count > b.hit_count;
                    });
            }

            if (verbose)
            {
                seqan3::debug_stream << "    matches: " << results << std::endl;
            }

            int all_matches_count = 0;
            int compared_traces_count = 0;

            // Use one valid reference trace as the initial baseline so later comparisons can use
            // its levenshtein distance as the current upper bound.
            int closest_trace_index = read_traces.size() -1;

            int closest_alignment_cost = levenshtein_distance_with_threshold(trace_events, read_traces[closest_trace_index], trace_length + shortest_path);
            compared_traces_count += 1;

            bool stop = false;

            for (const auto& bucket : bucket_scores) {

                if (stop){
                    break;
                }

                // Translate the current best edit distance into a minimum number of shared k-mer hits
                // a bucket should have to remain competitive.
                size_t max_cost = static_cast<size_t>(k * closest_alignment_cost);

                // Buckets below this hit threshold are unlikely to improve the current best match.
                size_t threshold = (kmers.size() > max_cost) ? (kmers.size() - max_cost) : 0;

                // Always check at least the first candidate bucket.
                // Afterwards, only inspect buckets that are still plausible under the current bound,
                // unless the IBF produced no hits at all.
                if (all_hits_zero || bucket.hit_count >= threshold || all_matches_count == 0){

                    if (trace_buckets[bucket.index].empty())
                    {
                        continue;
                    }

                    ++all_matches_count;

                    // Compare the query trace with all reference traces in this bucket.
                    for (int trace_index : trace_buckets[bucket.index]) {
                        // Use the current best edit distance as threshold for early termination.
                        int edits = levenshtein_distance_with_threshold(trace_events, read_traces[trace_index], closest_alignment_cost);
                        compared_traces_count += 1;

                        if (edits == 0)
                        {
                            // Exact match found; no better result is possible.
                            closest_trace_index = trace_index;
                            closest_alignment_cost = 0;
                            stop = true;
                            break;
                        }

                        // Keep the currently best reference trace.
                        if (edits < closest_alignment_cost)
                        {
                            closest_trace_index = trace_index;
                            closest_alignment_cost = edits;
                        }
                    }

                    // Optional hard stop after a configured number of checked buckets.
                    if (max_checked_buckets !=0 and all_matches_count >= max_checked_buckets) {
                        stop = true;
                    }
                } else {
                    // When buckets are sorted by descending hit count, all remaining buckets will have
                    // equal or fewer hits; once the threshold is not met, no subsequent bucket can qualify.
                    if (sorting == "hitcount") {
                        break;
                    }
                }
            }

            // Reconstruct the best matching reference trace for CSV output.
            std::string best_trace = join_trace_fast(read_traces[closest_trace_index], " - ");

            std::chrono::steady_clock::time_point end_search = std::chrono::steady_clock::now();

            // Write final result and search statistics for this query trace.
            output_csv << "\"" << trace << "\";\"" << best_trace << "\";\"" << closest_alignment_cost << "\";" << (std::chrono::duration_cast<std::chrono::microseconds>(end_search - begin_search).count()) / 1000.0 << ";" << all_matches_count << ";" << compared_traces_count << "\n";

            // Store result for future duplicate queries.
            already_searched_traces[trace] = "\"" + best_trace + "\";\"" + std::to_string(closest_alignment_cost) + "\"";
        }
        return true;
    }
    else
    {
        std::cout << "Something went wrong while reading search file!" << std::endl;
        return false;
    }
}


/**
 * Builds an Interleaved Bloom Filter from model traces and searches event-log traces against it.
 *
 * Input:
 * - file_traces_path: file containing reference/model traces used to build the IBF
 * - file_search_path: file containing query traces to search
 * - file_output_path: output file for search/alignment results
 * - file_ibf_load_path: input file for IBF deserialization (serializing/deserializing not implemented atm)
 * - file_ibf_save_path: output file for IBF serialization (serializing/deserializing not implemented atm)
 * - program_type: controls whether to build, deserialize, search, or serialize (serializing/deserializing not implemented atm)
 * - k: k-mer length
 * - seed: seed for MurmurHash3
 * - verbose: enables debug output
 * - bucketing: strategy used to assign traces to buckets
 * - number_buckets_calc: strategy for deriving bucket count
 * - num_hashes / omh_w / omh_seed: parameters for LSH / OMH bucketing
 * - sorting / shortest_path / bucket_limit: search-phase configuration
 *
 * Behavior:
 * - Counts traces to determine IBF size
 * - Builds the IBF
 * - Exports bucket assignments
 * - Searches query traces against the IBF and writes results
 *
 * Purpose:
 * - Main orchestration function for the full build + search pipeline
 */
void search_alignments(const char* file_traces_path, const char* file_search_path, const char* file_output_path,
    const char* file_ibf_load_path, const char* file_ibf_save_path,
    Execution_Types program_type, int k, int seed, bool verbose, const std::string& bucketing, const std::string& number_buckets_calc, size_t num_hashes, size_t omh_w, std::uint64_t omh_seed, const std::string& sorting, size_t shortest_path, size_t bucket_limit)
{
    std::ifstream file_search(file_search_path);

    // Open trace source and count lines to estimate how many traces must be indexed
    std::ifstream file_traces(file_traces_path);
    long unsigned int traces_count = std::count(std::istreambuf_iterator<char>(file_traces),
                                                std::istreambuf_iterator<char>(), '\n');
    file_traces.clear();
    file_traces.seekg(0);

    // Auxiliary containers used during build and search
    std::vector<int> kmer_count_per_bin;
    std::vector<std::string> search_traces;
	std::vector<std::vector<std::string>> read_traces;

    // Measure IBF build/deserialization time
    std::chrono::steady_clock::time_point begin_ibf_generation = std::chrono::steady_clock::now();

    // Default bucket count is one bucket per trace
	unsigned long int number_buckets = traces_count;

    // Optional reduction of bucket count to sqrt(n) for coarser grouping and lower index size
	if (number_buckets_calc == "sqrt") {
		number_buckets = static_cast<long unsigned int>(std::sqrt(traces_count));
	}

    // Stores, for each bucket, which trace indices were assigned to it
    BucketList trace_buckets(number_buckets);

    // Construct the IBF with fixed bin size and hash-function count
    seqan3::interleaved_bloom_filter ibf{seqan3::bin_count{number_buckets},
                                         seqan3::bin_size{2048u},
                                         seqan3::hash_function_count{2u}};

    std::cout << "Building IBF..." << std::endl;

    // Build the IBF from traces
    if (!build_interleaved_bloom_filter(ibf, file_traces, file_ibf_load_path, read_traces, program_type, k, seed, kmer_count_per_bin, verbose, trace_buckets, bucketing, number_buckets, num_hashes, omh_w, omh_seed)) return;

    std::chrono::steady_clock::time_point end_ibf_generation = std::chrono::steady_clock::now();

    std::cout << "IBF generation finished in " << (std::chrono::duration_cast<std::chrono::microseconds>(end_ibf_generation - begin_ibf_generation).count()) / 1000.0 << "[ms]" << std::endl;

    // Export bucket contents for inspection/debugging
    std::string bucketlist_path = "../output/temp/bucketlist_" + bucketing + "_" + std::to_string(num_hashes) + "_" + std::to_string(number_buckets) + "_k_" + std::to_string(k) + ".csv";

    export_bucketlist_to_csv(trace_buckets, bucketlist_path);

    // In serialize mode, persist the built IBF and stop before search
    // (serializing/deserializing not implemented atm)
    if (program_type == Execution_Types::SERIALIZE)
    {
        std::cout << "Serializing IBF to file." << std::endl;
        std::ofstream out_file (file_ibf_save_path, std::ios::binary);
        serialize(out_file, ibf);
        std::cout << "Serializing IBF complete!" << std::endl;
        return;
    }

    // Measure IBF search time separately from build time
    std::chrono::steady_clock::time_point begin_ibf_search = std::chrono::steady_clock::now();

    // Search query traces in the IBF and write alignment/search output
    std::cout << "Search in IBF..." << std::endl;
    if (!search_in_interleaved_bloom_filter(ibf, file_search, file_output_path, read_traces, search_traces, k, seed,
        kmer_count_per_bin, verbose, trace_buckets, sorting, shortest_path, bucket_limit)) return;

    std::chrono::steady_clock::time_point end_ibf_search = std::chrono::steady_clock::now();

    std::cout << "IBF search finished in " << (std::chrono::duration_cast<std::chrono::microseconds>(end_ibf_search - begin_ibf_search).count()) / 1000.0 << "[ms]" << std::endl;
    std::cout << "Results saved to " << file_output_path << std::endl;
}


/**
 * @brief Initialize SeqAn3 argument parser
 *
 * @param parser SeqAn3 argument parser
 */
void initialize_argument_parser(seqan3::argument_parser & parser)
{
    parser.info.author = "Florian Marx and Jessica Kloss";
    parser.info.short_description = "Check alignment of model traces with real time trace logs.";
    parser.info.version = "1.0.2";
}


/**
 * Entry point of the application handling argument parsing and execution mode selection.
 *
 * Input:
 * - argc / argv: command-line arguments
 *
 * Behavior:
 * - Parses CLI arguments using seqan3 argument parser
 * - Configures parameters for IBF construction and search
 * - Determines execution mode (build, load, search)
 * - Calls the main pipeline (search_alignments)
 *
 * Purpose:
 * - Provides a flexible CLI interface to control the full workflow
 */
int main(int argc, char* argv[])
{
    // Enable verbose output via CLI flag to print debug information during execution.

    seqan3::argument_parser argument_parser{"alignment_txt", argc, argv};

    initialize_argument_parser(argument_parser);

    // Used to check whether optional file arguments were provided.
    std::filesystem::path empty_file{};

    // k-mer length: number of consecutive events forming one k-mer.
    int k = 2;
    argument_parser.add_option(k, 'k', "kmer", "The amount of events per mere.");

    // Seed for MurmurHash3 to ensure reproducible hashing.
    int seed = 42;
    argument_parser.add_option(seed, 'x', "seed", "Set a custom seed for the hash functions.");

    // Input file containing reference/model traces.
    std::filesystem::path traces_file{};
    argument_parser.add_option(traces_file, 't', "traces", "The input TXT file containing traces to search from.",
                               seqan3::option_spec::standard,
                               seqan3::input_file_validator{{"txt"}});

    // Input file containing query traces to search.
    std::filesystem::path search_file{};
    argument_parser.add_option(search_file, 's', "search", "The input TXT file containing traces to search in.",
                               seqan3::option_spec::standard,
                               seqan3::input_file_validator{{"txt"}});

    // Output file for serialized IBF (build-only mode).
    std::filesystem::path ibf_file_save{};
    argument_parser.add_option(ibf_file_save, 'd', "dump", "This will only generate the IBF data structure and save it to the given file.",
                               seqan3::option_spec::standard,
                               seqan3::output_file_validator{seqan3::output_file_open_options::open_or_create, {"ibf"}});

    // Input file for loading a pre-built IBF.
    std::filesystem::path ibf_file_load{};
    argument_parser.add_option(ibf_file_load, 'l', "load", "This will only check conformance with the given IBF data structure.",
                               seqan3::option_spec::standard,
                               seqan3::input_file_validator{{"ibf"}});

    // Output CSV file for alignment/search results.
    std::filesystem::path output_file{};
    argument_parser.add_option(output_file, 'o', "output", "The output CSV file containing all calculated alignment of traces (Will override existing files!).",
                               seqan3::option_spec::standard,
                               seqan3::output_file_validator{seqan3::output_file_open_options::open_or_create, {"csv"}});

    // Enables verbose logging during IBF construction and search.
    bool verbose = false;
    argument_parser.add_flag(verbose, 'v', "verbose", "Enable verbose messages for generation and search");

    // Delimiter used for splitting events into k-mers (currently not used in fast pipeline).
    std::string delimiter = "--<DELIM>--";
    argument_parser.add_option(delimiter, 'a', "delimiter", "Set delimiter for k-mer generation");

    // Number of hash functions used for MinHash (LSH bucketing).
    int num_hashes = 128;
    argument_parser.add_option(num_hashes, 'n', "num_hashes", "The amount of hashes for minHash");

    // Strategy for determining number of buckets ("n" or "sqrt").
	std::string number_buckets_calc = "sqrt";
	argument_parser.add_option(number_buckets_calc, 'c', "number_buckets_calc", "Set the way the number of buckets is calculated. Either n or sqrt");

    // Bucketing strategy used to group traces.
	std::string bucketing = "omh";
	argument_parser.add_option(bucketing, 'b', "bucketing", "Choose bucketing algo between omh, lsh, random, tracelen, set and i");

    // Number of smallest hashes used in OMH signature.
    // Controls trade-off between accuracy and performance (larger = more stable but slower).
    int omh_w = 3;
    argument_parser.add_option(omh_w, 'w', "omh_w", "Number of smallest hashes used for OMH signature");

    // Seed used for OMH hash mixing to ensure reproducibility.
    std::uint64_t omh_seed = 0x9e3779b97f4a7c15;
    argument_parser.add_option(omh_seed, 'z', "omh_seed", "Seed for OMH hashing");

    // Sorting strategy for ranking buckets during search.
    // "score": normalized by bucket size
    // "score2": normalized by query size
    // "hitcount": raw hit count
    std::string sorting = "score";
    argument_parser.add_option(sorting,'g', "sorting", "Sort buckets by 'score', 'score2' or 'hitcount'");

    // Lower-bound offset added to initial edit-distance threshold.
    int shortest_path = 0;
    argument_parser.add_option(shortest_path, 'p', "shortest_path", "Shortest visible path length in process model");

    // Limits how many buckets are checked per query.
    // Improves performance by early stopping, at the cost of possibly missing the optimal match.
    int bucket_limit = 0;
    argument_parser.add_option(bucket_limit, 'y', "bucket_limit", "Maximum number of checked buckets per trace (0 = no limit, check all buckets)");

    try
    {
        argument_parser.parse();

        Execution_Types program_type;

        // Determine execution mode based on provided CLI arguments.
        if (traces_file != empty_file && ibf_file_save != empty_file)
        {
            // Build IBF and serialize to disk.
            program_type = Execution_Types::SERIALIZE;
        }
        else if (search_file != empty_file && ibf_file_load != empty_file && output_file != empty_file)
        {
            // Load IBF from disk and perform search.
            program_type = Execution_Types::DESERIALIZE;
        }
        else if (traces_file != empty_file && search_file != empty_file && output_file != empty_file)
        {
            // Build IBF and immediately perform search.
            program_type = Execution_Types::NORMAL;
        }

        // Execute main pipeline (build, load, search depending on mode).
        search_alignments(
            traces_file.string().c_str(), search_file.string().c_str(), output_file.string().c_str(),
            ibf_file_load.string().c_str(), ibf_file_save.string().c_str(),
            program_type, k, seed, verbose, bucketing, number_buckets_calc, num_hashes, omh_w, omh_seed, sorting, shortest_path, bucket_limit);
    }
    catch (seqan3::argument_parser_error const & ext)
    {
        seqan3::debug_stream << "[Error] " << ext.what() << "\n";
        return -1;
    }

    return 0;
}