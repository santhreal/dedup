#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![forbid(unsafe_code)]
//! # dedup  -  High-performance dataset deduplication for ML training data
//!
//! A `MinHash` + LSH implementation for finding near-duplicate documents
//! in massive datasets. Designed for streaming operation to handle
//! billions of documents without loading all into memory.
//!
//! ## Quick Start
//!
//! ```rust
//! use dedup::{Config, DedupTransformer};
//!
//! let config = Config::default()
//!     .with_similarity_threshold(0.85)
//!     .with_num_bands(16);
//!
//! let dedup = DedupTransformer::new(config).unwrap();
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
//! │   Shingle   │────▶│   MinHash    │────▶│  LSH Bands  │────▶│   Deduplicate│
//! │  (k-grams)  │     │  (fast hash) │     │  (buckets)  │     │   (filter)   │
//! └─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
//! ```
//!
//! ## `MinHash` + LSH Theory
//!
//! - **Shingling**: Convert documents to sets of k-grams (overlapping subsequences)
//! - **MinHash**: Compress document to a small signature while preserving Jaccard similarity
//! - **LSH**: Band signatures such that similar documents collide in at least one bucket
//! - **Threshold**: Documents with estimated Jaccard ≥ threshold are considered duplicates

#![warn(missing_docs, clippy::pedantic)]

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::todo,
        clippy::unimplemented,
        clippy::panic
    )
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::module_name_repetitions,
    clippy::needless_pass_by_value,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::unnecessary_literal_bound,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]

mod cluster;
mod config;
mod error;
mod lsh;
mod minhash;
pub mod shingle;
mod fast_hash;
mod transform;

pub use config::Config;
pub use error::{Error, Result};
pub use cluster::DuplicateCluster;
pub use lsh::LshIndex;
pub use minhash::{MinHashSignature, MinHasher};
pub use shingle::ShingleIterator;
pub use fast_hash::{hash_bytes, FastHasher};
pub use transform::{DedupTransformer, StatefulDedupTransform};

/// Re-export tenshift types for convenience.
pub mod tenshift {
    pub use tenshift_core::sample::Sample;
    pub use tenshift_core::transform::{Transform, TransformResult};
}

/// Default number of hash functions (signature size).
pub const DEFAULT_SIGNATURE_SIZE: usize = 128;

/// Default shingle size in bytes/characters.
pub const DEFAULT_SHINGLE_SIZE: usize = 5;

/// Default number of LSH bands.
pub const DEFAULT_NUM_BANDS: usize = 16;

/// Default similarity threshold for considering documents as duplicates.
pub const DEFAULT_SIMILARITY_THRESHOLD: f64 = 0.9;

/// Compute the number of rows per band given signature size and num bands.
///
/// Returns `None` if the signature size is not evenly divisible by num bands.
#[must_use]
pub const fn compute_rows_per_band(signature_size: usize, num_bands: usize) -> Option<usize> {
    if num_bands == 0 || signature_size % num_bands != 0 {
        return None;
    }
    Some(signature_size / num_bands)
}

/// Estimate the false positive rate given LSH parameters.
///
/// The probability that two documents with similarity `s` will be
/// marked as candidates for comparison.
#[must_use]
pub fn candidate_probability(similarity: f64, num_bands: usize, rows_per_band: usize) -> f64 {
    if similarity <= 0.0 {
        return 0.0;
    }
    if similarity >= 1.0 {
        return 1.0;
    }
    // P(at least one band matches) = 1 - P(no bands match)
    // P(one band matches) = s^r where r = rows_per_band
    let band_match_prob = similarity.powf(rows_per_band as f64);
    1.0 - (1.0 - band_match_prob).powf(num_bands as f64)
}

/// Find the optimal LSH parameters for a given similarity threshold.
///
/// Returns `(num_bands, rows_per_band)` that maximizes the S-curve
/// steepness around the threshold.
#[must_use]
pub fn optimize_lsh_params(
    signature_size: usize,
    target_threshold: f64,
) -> (usize, usize) {
    // The optimal point is where s^r ≈ 1/b for threshold s
    // This gives us roughly b * s^r = 1 expected matches at threshold.
    //
    // Enumerate every band count that EXACTLY divides `signature_size` (not a
    // hardcoded {4,8,16,32,64} shortlist). A shortlist returns an INVALID
    // (bands, rows) whenever `signature_size` divides none of its entries
    // (e.g. 100: 100/8=12 but 8*12=96 != 100), so the caller would build an
    // LSH index whose bands don't tile the signature. Iterating real divisors
    // guarantees `bands * rows == signature_size` for the returned pair, and
    // the discrimination score naturally rejects the degenerate 1-band /
    // 1-row divisors.
    if signature_size == 0 {
        return (1, 0);
    }
    // Only consider band counts up to a sane cap. Realistic LSH configs never
    // use more than a few hundred bands, and iterating `1..=signature_size`
    // would hang for an adversarial `signature_size` near `usize::MAX`. The
    // cap keeps the search O(MAX_CANDIDATE_BANDS); a valid result is still
    // guaranteed because `num_bands == 1` (rows == signature_size) always
    // divides and is always in range.
    const MAX_CANDIDATE_BANDS: usize = 1024;
    let search_limit = signature_size.min(MAX_CANDIDATE_BANDS);
    let mut best: Option<(f64, usize, usize)> = None;

    for num_bands in 1..=search_limit {
        if signature_size % num_bands != 0 {
            continue;
        }
        let rows_per_band = signature_size / num_bands;

        // Score by how steep the curve is at the threshold
        let p_at_threshold = candidate_probability(target_threshold, num_bands, rows_per_band);
        let p_below = candidate_probability(target_threshold * 0.9, num_bands, rows_per_band);
        let p_above =
            candidate_probability((target_threshold * 1.1).min(1.0), num_bands, rows_per_band);

        // We want high discrimination: low p_below, high p_above
        let score = p_above - p_below - (p_at_threshold - 0.5).abs() * 0.5;

        if best.map_or(true, |(best_score, _, _)| score > best_score) {
            best = Some((score, num_bands, rows_per_band));
        }
    }

    // `signature_size >= 1` always has the divisor 1, so `best` is `Some`.
    let (_, bands, rows) = best.unwrap_or((0.0, 1, signature_size));
    (bands, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rows_per_band() {
        assert_eq!(compute_rows_per_band(128, 16), Some(8));
        assert_eq!(compute_rows_per_band(128, 8), Some(16));
        assert_eq!(compute_rows_per_band(128, 0), None);
        assert_eq!(compute_rows_per_band(128, 3), None);
    }

    #[test]
    fn test_candidate_probability_bounds() {
        assert_eq!(candidate_probability(0.0, 16, 8), 0.0);
        assert_eq!(candidate_probability(1.0, 16, 8), 1.0);
    }

    #[test]
    fn test_candidate_probability_increases_with_similarity() {
        let p1 = candidate_probability(0.5, 16, 8);
        let p2 = candidate_probability(0.8, 16, 8);
        let p3 = candidate_probability(0.95, 16, 8);
        
        assert!(p1 < p2, "probability should increase with similarity");
        assert!(p2 < p3, "probability should increase with similarity");
    }

    #[test]
    fn test_optimize_lsh_params_produces_valid_params() {
        let (bands, rows) = optimize_lsh_params(128, 0.9);
        assert!(bands > 0);
        assert!(rows > 0);
        assert_eq!(bands * rows, 128);
    }

    #[test]
    fn test_optimize_lsh_params_valid_when_indivisible_by_hardcoded_bands() {
        // Regression: the old code only tried band counts {4,8,16,32,64} (all
        // multiples of 4). Any signature_size NOT divisible by 4 matched none,
        // so it returned the untouched seed (8, size/8) whose product != size:
        // e.g. size=6 -> (8, 0), size=50 -> (8, 6) [48 != 50]. Now every result
        // must exactly tile the signature.
        for size in [1_usize, 2, 3, 5, 6, 7, 10, 25, 30, 50, 100, 127, 200] {
            let (bands, rows) = optimize_lsh_params(size, 0.85);
            assert!(bands >= 1, "size {size}: bands must be >= 1, got {bands}");
            assert!(rows >= 1, "size {size}: rows must be >= 1, got {rows}");
            assert_eq!(
                bands * rows,
                size,
                "size {size}: returned ({bands}, {rows}) must tile the signature exactly"
            );
        }
    }
}

/// Compile-checks the README quick-start example as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
