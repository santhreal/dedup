//! Configuration for the deduplication engine.

use crate::error::{Error, Result};

/// Configuration for the deduplication engine.
///
/// This struct controls all parameters for MinHash + LSH deduplication.
/// Use the builder pattern to customize:
///
/// ```rust
/// use dedup::Config;
///
/// let config = Config::default()
///     .with_similarity_threshold(0.85)
///     .with_num_bands(14)
///     .with_shingle_size(4);
/// ```
/// Upper bound on [`Config::signature_size`]. MinHash accuracy gains flatten
/// out by a few hundred hash functions (typical configurations use 64-256),
/// so 65536 is far beyond any useful value. The cap exists to fail closed:
/// the signature is stored per document and the hash coefficients are two
/// `u64` vectors of this length, so an unbounded size turns a hostile or
/// mistaken configuration into gigabytes of allocation.
pub const MAX_SIGNATURE_SIZE: usize = 1 << 16;

/// Configuration for MinHash and LSH deduplication parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Number of hash functions (signature size).
    pub signature_size: usize,
    /// Number of LSH bands.
    pub num_bands: usize,
    /// Size of shingles (k-grams).
    pub shingle_size: usize,
    /// Similarity threshold for duplicates (0.0-1.0).
    pub similarity_threshold: f64,
    /// Maximum document size in bytes.
    pub(crate) max_document_size: usize,
    /// Memory limit in MB for the LSH index.
    pub(crate) memory_limit_in_mb: usize,
    /// Seed for hash function randomization.
    pub(crate) seed: u64,
    /// Whether to store document content in memory.
    pub(crate) store_documents: bool,
}

impl Config {
    /// Create a new configuration with custom parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if parameters are invalid:
    /// - `signature_size` must be divisible by `num_bands`
    /// - `similarity_threshold` must be in (0.0, 1.0]
    /// - `shingle_size` must be at least 1
    pub fn new(
        signature_size: usize,
        num_bands: usize,
        shingle_size: usize,
        similarity_threshold: f64,
    ) -> Result<Self> {
        if signature_size == 0 {
            return Err(Error::InvalidConfig {
                reason: "signature_size must be at least 1".to_string(),
                fix: "use signature_size >= 1".to_string(),
            });
        }
        if signature_size > MAX_SIGNATURE_SIZE {
            return Err(Error::InvalidConfig {
                reason: format!(
                    "signature_size ({signature_size}) exceeds maximum {MAX_SIGNATURE_SIZE}"
                ),
                fix: format!(
                    "use signature_size <= {MAX_SIGNATURE_SIZE}; larger signatures add no accuracy and only exhaust memory"
                ),
            });
        }
        if num_bands == 0 {
            return Err(Error::InvalidConfig {
                reason: "num_bands must be at least 1".to_string(),
                fix: "use num_bands >= 1".to_string(),
            });
        }
        if signature_size % num_bands != 0 {
            return Err(Error::InvalidConfig {
                reason: format!(
                    "signature_size ({signature_size}) must be divisible by num_bands ({num_bands})"
                ),
                fix: "use signature_size = num_bands * rows_per_band".to_string(),
            });
        }
        if shingle_size == 0 {
            return Err(Error::InvalidConfig {
                reason: "shingle_size must be at least 1".to_string(),
                fix: "use shingle_size >= 1".to_string(),
            });
        }
        if similarity_threshold <= 0.0 || similarity_threshold > 1.0 {
            return Err(Error::InvalidConfig {
                reason: format!(
                    "similarity_threshold ({similarity_threshold}) must be in (0.0, 1.0]"
                ),
                fix: "use 0.0 < similarity_threshold <= 1.0".to_string(),
            });
        }

        Ok(Self {
            signature_size,
            num_bands,
            shingle_size,
            similarity_threshold,
            max_document_size: 10 * 1024 * 1024, // 10 MB default
            memory_limit_in_mb: 4096,            // 4 GB default
            seed: 0x9e37_79b9_7f4a_7c15,         // Random seed
            store_documents: false,
        })
    }

    /// Set the similarity threshold.
    #[must_use]
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold.clamp(0.01, 1.0);
        self
    }

    /// Set the number of LSH bands.
    ///
    /// The bands must tile the signature exactly (`signature_size % num_bands
    /// == 0`). Rather than silently dropping a requested count that does not
    /// divide the current `signature_size` (the old behavior, which left the
    /// caller believing their value took effect), this snaps to the divisor of
    /// `signature_size` nearest the request, so the invariant always holds and
    /// the change always takes effect. A request of `0` (no valid band count)
    /// leaves the current value unchanged. Use [`Config::new`] for a fallible,
    /// error-returning construction instead.
    #[must_use]
    pub fn with_num_bands(mut self, num_bands: usize) -> Self {
        if num_bands > 0 {
            self.num_bands = nearest_divisor(self.signature_size, num_bands);
        }
        self
    }

    /// Set the shingle size (k-gram length).
    #[must_use]
    pub fn with_shingle_size(mut self, shingle_size: usize) -> Self {
        if shingle_size > 0 {
            self.shingle_size = shingle_size;
        }
        self
    }

    /// Set the signature size (number of hash functions).
    ///
    /// The signature must be an exact multiple of `num_bands`. Rather than
    /// silently dropping a requested size that is not (the old behavior), this
    /// rounds UP to the next multiple of the current `num_bands`, so the bands
    /// always tile the signature and the change always takes effect. A request
    /// of `0` leaves the current value unchanged. Use [`Config::new`] for a
    /// fallible, error-returning construction instead.
    #[must_use]
    pub fn with_signature_size(mut self, signature_size: usize) -> Self {
        if signature_size > 0 {
            // Clamp before snapping: `div_ceil(bands) * bands` overflows usize
            // for requests near usize::MAX (panic in debug, silent wrap to a
            // tiny size in release), and an uncapped size lets the hasher
            // allocate gigabytes of coefficients. The cap is far beyond any
            // accuracy-useful signature, so real configurations are unchanged.
            let signature_size = signature_size.min(MAX_SIGNATURE_SIZE);
            let bands = self.num_bands.max(1);
            self.signature_size = signature_size.div_ceil(bands) * bands;
        }
        self
    }

    /// Set the maximum document size in bytes.
    #[must_use]
    pub fn with_max_document_size(mut self, max_bytes: usize) -> Self {
        self.max_document_size = max_bytes;
        self
    }

    /// Set the memory limit in MB.
    #[must_use]
    pub fn with_memory_limit(mut self, memory_limit_in_mb: usize) -> Self {
        self.memory_limit_in_mb = memory_limit_in_mb;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable or disable storing document content.
    #[must_use]
    pub fn with_store_documents(mut self, store: bool) -> Self {
        self.store_documents = store;
        self
    }

    /// Calculate rows per band.
    #[must_use]
    pub const fn rows_per_band(&self) -> usize {
        self.signature_size / self.num_bands
    }

    /// Calculate the estimated memory usage per document in bytes.
    #[must_use]
    pub fn estimated_memory_per_document(&self) -> usize {
        // Signature: signature_size * 4 bytes (u32)
        // LSH index overhead: num_bands * pointer overhead
        let signature_bytes = self.signature_size * 4;
        let index_overhead = self.num_bands * 16; // Approximate
        signature_bytes + index_overhead + 64 // Base overhead per document
    }

    /// Calculate the maximum number of documents that fit in memory.
    #[must_use]
    pub fn max_documents_in_memory(&self) -> usize {
        let memory_bytes = self.memory_limit_in_mb.saturating_mul(1024 * 1024);
        let per_doc = self.estimated_memory_per_document();
        memory_bytes / per_doc.max(1)
    }
}

impl Default for Config {
    fn default() -> Self {
        // These parameters give good results for text deduplication
        // Signature size 128, 16 bands = 8 rows per band
        // Threshold ≈ (1/16)^(1/8) ≈ 0.83
        Self {
            signature_size: 128,
            num_bands: 16,
            shingle_size: 5,
            similarity_threshold: 0.9,
            max_document_size: 10 * 1024 * 1024,
            memory_limit_in_mb: 4096,
            seed: 0x9e37_79b9_7f4a_7c15,
            store_documents: false,
        }
    }
}

/// The divisor of `n` closest to `target` (ties resolve to the smaller
/// divisor). Used by [`Config::with_num_bands`] to snap a requested band count
/// onto a value that tiles the signature. `n >= 1` always has the divisor `1`,
/// so the result is always a valid divisor; returns `1` for `n == 0`.
///
/// The candidate scan is capped at [`MAX_BAND_SEARCH`] so a hostile
/// `signature_size` near `usize::MAX` cannot make this loop O(n) and hang;
/// realistic band counts are far below the cap, and `1` is always in range.
fn nearest_divisor(n: usize, target: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let limit = n.min(MAX_BAND_SEARCH);
    let mut best = 1_usize;
    let mut best_dist = target.abs_diff(1);
    let mut d = 1_usize;
    while d <= limit {
        if n % d == 0 {
            let dist = target.abs_diff(d);
            if dist < best_dist {
                best_dist = dist;
                best = d;
            }
        }
        d += 1;
    }
    best
}

/// Upper bound on the band-count divisor search in [`nearest_divisor`]. Well
/// above any realistic LSH band count, so it never changes a real result, but
/// keeps the scan O(1) for a hostile `signature_size`.
const MAX_BAND_SEARCH: usize = 1 << 16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_divisor_snaps_to_closest_divisor() {
        // Divisors of 128: 1,2,4,8,16,32,64,128.
        assert_eq!(nearest_divisor(128, 20), 16); // 16 (d4) closer than 32 (d12)
        assert_eq!(nearest_divisor(128, 30), 32); // 32 (d2) closer than 16 (d14)
        assert_eq!(nearest_divisor(128, 8), 8); // exact divisor unchanged
        assert_eq!(nearest_divisor(100, 7), 5); // divisors 1,2,4,5,10,...: 5 nearest to 7
        assert_eq!(nearest_divisor(0, 9), 1); // degenerate
    }

    #[test]
    fn with_num_bands_snaps_indivisible_request_to_valid_divisor() {
        // Old behavior: 20 does not divide 128, so it was silently ignored and
        // num_bands stayed 16 with the CALLER believing 20 took effect. Now it
        // snaps to the nearest divisor and always tiles the signature.
        let config = Config::default().with_num_bands(20);
        assert_eq!(config.num_bands, 16);
        assert_eq!(config.signature_size % config.num_bands, 0);

        // A request that snaps to a different value than the default.
        let config = Config::default().with_num_bands(30);
        assert_eq!(config.num_bands, 32);
        assert_eq!(config.signature_size % config.num_bands, 0);
    }

    #[test]
    fn with_signature_size_rounds_up_to_multiple_of_bands() {
        // num_bands defaults to 16; 100 is not a multiple, so the old code
        // silently kept 128. Now it rounds up to the next multiple (112).
        let config = Config::default().with_signature_size(100);
        assert_eq!(config.num_bands, 16);
        assert_eq!(config.signature_size, 112);
        assert_eq!(config.signature_size % config.num_bands, 0);

        // An exact multiple is preserved.
        let config = Config::default().with_signature_size(256);
        assert_eq!(config.signature_size, 256);
    }

    #[test]
    fn default_config_valid() {
        let config = Config::default();
        assert_eq!(config.signature_size, 128);
        assert_eq!(config.num_bands, 16);
        assert_eq!(config.rows_per_band(), 8);
    }

    #[test]
    fn new_validates_signature_size() {
        let result = Config::new(100, 16, 5, 0.9);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("divisible"));
    }

    #[test]
    fn new_validates_threshold() {
        let result = Config::new(128, 16, 5, 0.0);
        assert!(result.is_err());
        let result = Config::new(128, 16, 5, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn builder_pattern_works() {
        let config = Config::default()
            .with_similarity_threshold(0.85)
            .with_num_bands(8)
            .with_shingle_size(4);
        
        assert!((config.similarity_threshold - 0.85).abs() < f64::EPSILON);
        assert_eq!(config.num_bands, 8);
        assert_eq!(config.shingle_size, 4);
    }

    #[test]
    fn rows_per_band_calculation() {
        let config = Config::new(256, 32, 5, 0.9).unwrap();
        assert_eq!(config.rows_per_band(), 8);
    }

    #[test]
    fn memory_estimation() {
        let config = Config::default();
        let per_doc = config.estimated_memory_per_document();
        assert!(per_doc > 0);
        
        let max_docs = config.max_documents_in_memory();
        assert!(max_docs > 0);
    }

    #[test]
    fn invalid_shingle_size_rejected() {
        let result = Config::new(128, 16, 0, 0.9);
        assert!(result.is_err());
    }

    #[test]
    fn valid_config_accepts() {
        let config = Config::new(128, 16, 5, 0.9).unwrap();
        assert_eq!(config.signature_size, 128);
        assert_eq!(config.num_bands, 16);
    }

    /// Regression: `Config::new` accepted any `signature_size`, and the value
    /// flowed straight into per-document signature vectors and the hasher's
    /// coefficient vectors. `Config::new(usize::MAX, usize::MAX, ..)` passed
    /// validation and then died on a capacity-overflow panic (or abort) deep
    /// in `LshIndex::new` / `FastHasher::new`. Oversized sizes must now be
    /// rejected up front with an actionable error.
    #[test]
    fn new_rejects_oversized_signature_size() {
        let result = Config::new(MAX_SIGNATURE_SIZE + 16, 16, 5, 0.9);
        let err = result.expect_err("oversized signature_size must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("exceeds maximum"), "error names the bound: {msg}");
        assert!(msg.contains("Fix:"), "error carries a fix: {msg}");

        // The boundary itself stays valid.
        assert!(Config::new(MAX_SIGNATURE_SIZE, 16, 5, 0.9).is_ok());
    }

    /// Regression: `with_signature_size(usize::MAX)` computed
    /// `div_ceil(bands) * bands`, which overflows usize next to the maximum
    /// (panic in debug builds, silent wrap to a tiny size in release). The
    /// request is now clamped to `MAX_SIGNATURE_SIZE` before snapping, so the
    /// builder can neither panic nor silently shrink the signature.
    #[test]
    fn with_signature_size_near_usize_max_cannot_overflow() {
        let config = Config::default().with_signature_size(usize::MAX);
        assert_eq!(config.signature_size, MAX_SIGNATURE_SIZE);
        assert_eq!(config.signature_size % config.num_bands, 0);
    }
}
