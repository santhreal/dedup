/// Statistics about the LSH index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LshStats {
    /// Number of LSH bands.
    pub num_bands: usize,
    /// Rows per band.
    pub rows_per_band: usize,
    /// Similarity threshold.
    pub threshold: f64,
    /// Total documents indexed.
    pub doc_count: usize,
    /// Total buckets across all bands.
    pub total_buckets: usize,
    /// Total entries (sum of bucket sizes).
    pub total_entries: usize,
    /// Average bucket size.
    pub avg_bucket_size: f64,
    /// Number of duplicate clusters.
    pub cluster_count: usize,
    /// Number of duplicate documents.
    pub duplicate_count: usize,
}

impl LshStats {
    /// Calculate the estimated false positive rate.
    ///
    /// This estimates the probability that two documents with similarity
    /// below the threshold will be marked as candidates.
    #[must_use]
    pub fn estimated_false_positive_rate(&self) -> f64 {
        // Approximate using average bucket collisions
        if self.total_buckets == 0 {
            return 0.0;
        }
        let avg_docs_per_bucket = self.avg_bucket_size;
        let prob_collision = avg_docs_per_bucket / self.doc_count.max(1) as f64;
        
        // Probability of at least one collision in num_bands. `powf(_ as f64)`
        // (not `powi(_ as i32)`) avoids the usize->i32 downcast, which wraps to
        // a negative exponent for band counts above i32::MAX and silently
        // inverts the result; matches `candidate_probability` in the crate root.
        1.0 - (1.0 - prob_collision).powf(self.num_bands as f64)
    }

    /// Calculate the estimated recall.
    ///
    /// This estimates the probability that two documents with similarity
    /// equal to the threshold will be found as candidates.
    #[must_use]
    pub fn estimated_recall(&self) -> f64 {
        // For documents at exactly the threshold, probability of collision
        // in at least one band
        let s = self.threshold;
        // `powf(_ as f64)` avoids the usize->i32 downcast overflow (see
        // estimated_false_positive_rate) and matches candidate_probability.
        let band_match_prob = s.powf(self.rows_per_band as f64);
        1.0 - (1.0 - band_match_prob).powf(self.num_bands as f64)
    }
}
