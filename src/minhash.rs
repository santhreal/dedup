//! MinHash signature computation.
//!
//! MinHash is a technique for quickly estimating how similar two sets are.
//! It compresses large sets into small signatures while preserving Jaccard
//! similarity, enabling efficient near-duplicate detection.

use std::collections::HashSet;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::shingle::HashedShingleIterator;
use crate::fast_hash::FastHasher;
use tracing::{instrument, warn};

/// A MinHash signature for a document.
///
/// The signature is a vector of hash values (typically 64-256 values).
/// Similar documents will have similar signatures.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MinHashSignature {
    /// The hash values forming this signature.
    pub values: Vec<u32>,
    /// Document index (if part of a collection).
    pub doc_id: usize,
}

impl MinHashSignature {
    /// Create a new signature from raw values.
    pub fn new(values: Vec<u32>, doc_id: usize) -> Self {
        Self { values, doc_id }
    }

    /// Compute estimated Jaccard similarity with another signature.
    ///
    /// The similarity is the fraction of hash values that match between
    /// the two signatures. This approximates the true Jaccard similarity
    /// of the original sets.
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f64 {

        if self.values.len() != other.values.len() || self.values.is_empty() {
            warn!(
                self_len = self.values.len(),
                other_len = other.values.len(),
                "MinHashSignature::similarity called with mismatched or empty signature lengths"
            );
            return 0.0;
        }

        let matches = self
            .values
            .iter()
            .zip(&other.values)
            .filter(|(a, b)| a == b)
            .count();

        matches as f64 / self.values.len() as f64
    }

    /// Get a band of the signature for LSH.
    ///
    /// Returns the slice `values[start .. start + length]`, gracefully clamped
    /// to the signature bounds: a `start` at or past the end yields an empty
    /// slice, and an over-long `length` is truncated to the available tail. It
    /// never panics.
    ///
    /// This clamp is safe (not a silent recall loss) because the callers that
    /// require exact band tiling, the [`LshIndex`](crate::lsh::LshIndex),
    /// validate `signature.len() == num_bands * rows_per_band` before slicing,
    /// so a band is never silently shortened on the indexing path.
    #[must_use]
    pub fn band(&self, start: usize, length: usize) -> &[u32] {
        let end = start.saturating_add(length).min(self.values.len());
        &self.values[start.min(self.values.len())..end]
    }

    /// Compute a band hash for LSH bucketing.
    ///
    /// This combines all values in a band into a single hash value
    /// that can be used as a bucket key.
    #[must_use]
    pub fn band_hash(&self, start: usize, length: usize) -> u64 {
        let band = self.band(start, length);
        
        // FNV-1a inspired hash combination
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for &value in band {
            hash ^= u64::from(value);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        hash
    }

    /// Get the number of hash values in this signature.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if this signature is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// MinHasher computes MinHash signatures from documents.
///
/// This struct maintains the hash function coefficients and provides
/// methods to compute signatures from raw bytes or strings.
pub struct MinHasher {
    /// The fast hash function for MinHash computation.
    hasher: FastHasher,
    /// Shingle size (k-gram length).
    shingle_size: usize,
    /// Signature size (number of hash functions).
    signature_size: usize,
}

impl MinHasher {
    /// Create a new MinHasher from a configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    #[instrument(skip(config), level = "debug")]
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            hasher: FastHasher::new(config.signature_size, config.seed),
            shingle_size: config.shingle_size,
            signature_size: config.signature_size,
        })
    }

    /// Compute MinHash signature for a byte sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if the document is empty or too large.
    #[instrument(skip(self, data), fields(doc_id, data_len = data.len()), level = "debug")]
    pub fn compute(&self, data: &[u8], doc_id: usize) -> Result<MinHashSignature> {
        if data.is_empty() {
            warn!(doc_id, "empty document");
            return Err(Error::EmptyDocument { index: doc_id });
        }

        // Initialize signature with maximum values (we'll take minimum)
        let mut signature = vec![u32::MAX; self.signature_size];

        // Iterate over all shingles and update signature
        let shingle_iter = HashedShingleIterator::new(data, self.shingle_size);
        
        if shingle_iter.len() == 0 {
            warn!(doc_id, shingle_size = self.shingle_size, "document too short for shingle size");
            return Err(Error::EmptyDocument { index: doc_id });
        }

        for shingle_hash in shingle_iter {
            self.hasher.update_signature(&mut signature, shingle_hash);
        }

        Ok(MinHashSignature::new(signature, doc_id))
    }

    /// Compute signature for a string.
    ///
    /// # Errors
    ///
    /// Returns an error if the document is empty or too large.
    #[instrument(skip(self, text), fields(doc_id, text_len = text.len()), level = "debug")]
    pub fn compute_str(&self, text: &str, doc_id: usize) -> Result<MinHashSignature> {
        self.compute(text.as_bytes(), doc_id)
    }

    /// Compute signatures for multiple documents in batch.
    ///
    /// This is more efficient than calling `compute` repeatedly
    /// due to better cache utilization.
    pub fn compute_batch(&self, documents: &[&[u8]], start_id: usize) -> Vec<Result<MinHashSignature>> {
        documents
            .iter()
            .enumerate()
            .map(|(idx, doc)| match start_id.checked_add(idx) {
                Some(doc_id) => self.compute(doc, doc_id),
                // Overflow used to panic in debug and silently wrap (aliasing
                // doc 0) in release. Report the offending entry instead; the
                // rest of the batch still computes.
                None => Err(Error::InvalidConfig {
                    reason: format!(
                        "doc_id overflow: start_id {start_id} plus batch index {idx} exceeds usize::MAX"
                    ),
                    fix: "use a smaller start_id or split the batch".to_string(),
                }),
            })
            .collect()
    }

    /// Compute signature from pre-hashed shingles.
    ///
    /// Useful when shingles are computed elsewhere or cached.
    pub fn compute_from_hashed_shingles(
        &self,
        shingle_hashes: &[u64],
        doc_id: usize,
    ) -> MinHashSignature {
        let mut signature = vec![u32::MAX; self.signature_size];

        for &shingle_hash in shingle_hashes {
            self.hasher.update_signature(&mut signature, shingle_hash);
        }

        MinHashSignature::new(signature, doc_id)
    }

    /// Get the signature size.
    #[must_use]
    pub const fn signature_size(&self) -> usize {
        self.signature_size
    }

    /// Get the shingle size.
    #[must_use]
    pub const fn shingle_size(&self) -> usize {
        self.shingle_size
    }
}

/// Compute exact Jaccard similarity between two sets.
#[must_use]
pub fn exact_jaccard_similarity<T: Ord + Clone + std::hash::Hash>(a: &[T], b: &[T]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let set_a: HashSet<_> = a.iter().cloned().collect();
    let set_b: HashSet<_> = b.iter().cloned().collect();

    let intersection: HashSet<_> = set_a.intersection(&set_b).collect();
    let union: HashSet<_> = set_a.union(&set_b).collect();

    intersection.len() as f64 / union.len() as f64
}

/// Estimate the expected error of MinHash estimation.
///
/// The variance of MinHash similarity estimation is approximately:
/// Var(ŝ) ≈ s(1-s)/k where s is true similarity and k is signature size.
#[must_use]
pub fn expected_error(similarity: f64, signature_size: usize) -> f64 {
    let s = similarity.clamp(0.0, 1.0);
    let k = signature_size as f64;
    (s * (1.0 - s) / k).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn create_hasher() -> MinHasher {
        let config = Config::default();
        MinHasher::new(&config).unwrap()
    }

    #[test]
    fn minhash_signature_similarity_perfect() {
        let sig1 = MinHashSignature::new(vec![1, 2, 3, 4, 5], 0);
        let sig2 = MinHashSignature::new(vec![1, 2, 3, 4, 5], 1);
        
        assert!((sig1.similarity(&sig2) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn minhash_signature_similarity_zero() {
        let sig1 = MinHashSignature::new(vec![1, 2, 3, 4, 5], 0);
        let sig2 = MinHashSignature::new(vec![6, 7, 8, 9, 10], 1);
        
        assert!((sig1.similarity(&sig2) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn minhash_signature_similarity_partial() {
        let sig1 = MinHashSignature::new(vec![1, 2, 3, 4, 5], 0);
        let sig2 = MinHashSignature::new(vec![1, 2, 8, 9, 10], 1);
        
        // 2 out of 5 match = 0.4
        assert!((sig1.similarity(&sig2) - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn band_extraction() {
        let sig = MinHashSignature::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 0);
        let band = sig.band(2, 3);
        assert_eq!(band, &[3, 4, 5]);
    }

    #[test]
    fn band_hash_deterministic() {
        let sig = MinHashSignature::new(vec![1, 2, 3, 4, 5], 0);
        let h1 = sig.band_hash(0, 3);
        let h2 = sig.band_hash(0, 3);
        assert_eq!(h1, h2);
    }

    /// Regression: `compute_batch` assigned doc ids with `start_id + idx`,
    /// which panics in debug builds and silently wraps in release when
    /// `start_id` is near usize::MAX, aliasing later documents onto doc id 0.
    /// The overflowing entry must now surface as an error while the rest of
    /// the batch still computes.
    #[test]
    fn compute_batch_reports_doc_id_overflow() {
        let hasher = create_hasher();
        let docs: &[&[u8]] = &[b"first document", b"second document"];
        let results = hasher.compute_batch(docs, usize::MAX);

        assert_eq!(results.len(), 2);
        let first = results[0].as_ref().expect("index 0 fits at usize::MAX");
        assert_eq!(first.doc_id, usize::MAX);
        let err = results[1].as_ref().expect_err("index 1 must overflow");
        assert!(
            err.to_string().contains("doc_id overflow"),
            "error names the overflow: {err}"
        );
    }

    #[test]
    fn band_hash_different_bands_different() {
        let sig = MinHashSignature::new(vec![1, 2, 3, 4, 5, 6], 0);
        let h1 = sig.band_hash(0, 3);
        let h2 = sig.band_hash(3, 3);
        assert_ne!(h1, h2);
    }

    #[test]
    fn compute_signature_for_document() {
        let hasher = create_hasher();
        let doc = b"hello world this is a test document";
        let sig = hasher.compute(doc, 0).unwrap();
        
        assert_eq!(sig.len(), 128); // Default signature size
    }

    #[test]
    fn similar_documents_have_similar_signatures() {
        let hasher = create_hasher();
        
        let doc1 = b"hello world this is a test document";
        let doc2 = b"hello world this is a test document with extra words";
        
        let sig1 = hasher.compute(doc1, 0).unwrap();
        let sig2 = hasher.compute(doc2, 1).unwrap();
        
        let similarity = sig1.similarity(&sig2);
        // Similar documents should have > 0.5 estimated similarity
        assert!(similarity > 0.5, "similarity was {}", similarity);
    }

    #[test]
    fn different_documents_have_low_similarity() {
        let hasher = create_hasher();
        
        let doc1 = b"the quick brown fox jumps over the lazy dog";
        let doc2 = b"completely different content about various topics";
        
        let sig1 = hasher.compute(doc1, 0).unwrap();
        let sig2 = hasher.compute(doc2, 1).unwrap();
        
        let similarity = sig1.similarity(&sig2);
        // Different documents should have low similarity
        assert!(similarity < 0.3, "similarity was {}", similarity);
    }

    #[test]
    fn empty_document_errors() {
        let hasher = create_hasher();
        let result = hasher.compute(b"", 0);
        assert!(result.is_err());
    }

    #[test]
    fn document_too_short_for_shingle_size() {
        let hasher = create_hasher(); // Default shingle_size = 5
        let result = hasher.compute(b"hi", 0);
        assert!(result.is_err());
    }

    #[test]
    fn compute_str_works() {
        let hasher = create_hasher();
        let sig = hasher.compute_str("hello world", 0).unwrap();
        assert_eq!(sig.len(), 128);
    }

    #[test]
    fn batch_compute() {
        let hasher = create_hasher();
        let docs: Vec<&[u8]> = vec![
            b"document one content",
            b"document two content",
            b"document three content",
        ];
        
        let results = hasher.compute_batch(&docs, 0);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn compute_from_hashed_shingles() {
        let hasher = create_hasher();
        let shingles = vec![1_u64, 2, 3, 4, 5];
        let sig = hasher.compute_from_hashed_shingles(&shingles, 0);
        
        assert_eq!(sig.len(), 128);
    }

    #[test]
    fn exact_jaccard_identical_sets() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];
        assert!((exact_jaccard_similarity(&a, &b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn exact_jaccard_disjoint_sets() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        assert!((exact_jaccard_similarity(&a, &b) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn exact_jaccard_overlapping_sets() {
        let a = vec![1, 2, 3];
        let b = vec![2, 3, 4];
        // Intersection = {2, 3}, Union = {1, 2, 3, 4}
        assert!((exact_jaccard_similarity(&a, &b) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn expected_error_bounds() {
        // At similarity 0.5, error should be highest
        let err_mid = expected_error(0.5, 100);
        let err_low = expected_error(0.1, 100);
        let err_high = expected_error(0.9, 100);
        
        assert!(err_mid > err_low);
        assert!(err_mid > err_high);
    }

    #[test]
    fn signature_is_empty() {
        let sig = MinHashSignature::new(vec![], 0);
        assert!(sig.is_empty());
        
        let sig = MinHashSignature::new(vec![1, 2, 3], 0);
        assert!(!sig.is_empty());
    }

    #[test]
    fn minhash_preserves_similarity() {
        // Test that MinHash approximates Jaccard similarity
        let hasher = create_hasher();
        
        // Create two documents with known overlap
        let doc1 = "the quick brown fox jumps over the lazy dog";
        let doc2 = "the quick brown fox jumps over the lazy cat";
        
        // They share most words (dog vs cat is the main difference)
        let sig1 = hasher.compute_str(doc1, 0).unwrap();
        let sig2 = hasher.compute_str(doc2, 1).unwrap();
        
        let estimated_sim = sig1.similarity(&sig2);
        
        // Should be high but not perfect
        assert!(estimated_sim > 0.5 && estimated_sim < 1.0);
    }
    #[test]
    fn test_exact_jaccard_similarity_and_expected_error() {
        let set1 = vec!["apple", "banana", "cherry"];
        let set2 = vec!["banana", "cherry", "date"];
        // Intersection = 2, Union = 4 -> Jaccard = 0.5
        let sim = exact_jaccard_similarity(&set1, &set2);
        assert!((sim - 0.5).abs() < f64::EPSILON);

        let err = expected_error(0.5, 100);
        assert!((err - 0.05).abs() < 1e-4);

        assert_eq!(exact_jaccard_similarity::<&str>(&[], &[]), 1.0);
        assert_eq!(exact_jaccard_similarity(&["a"], &[]), 0.0);
    }
}
