//! tenshift Transform integration for deduplication.
//!
//! Provides `DedupTransformer` which implements the tenshift `Transform` trait,
//! allowing deduplication to be used as a pipeline stage.

use std::collections::hash_map::Entry;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::cluster::DuplicateCluster;
use crate::lsh::LshIndex;
use crate::minhash::MinHasher;

use tenshift_core::sample::Sample;

use tracing::{instrument, warn};

/// A transform that deduplicates samples using MinHash + LSH.
///
/// This transform buffers samples to compute signatures and find duplicates.
/// It can operate in two modes:
/// - **Streaming**: Process samples as they arrive, outputting non-duplicates immediately
/// - **Batch**: Buffer all samples, then output deduplicated set
///
/// # Example
///
/// ```rust
/// use dedup::{Config, DedupTransformer};
/// use tenshift_core::sample::Sample;
/// use tenshift_core::transform::Transform;
///
/// let config = Config::default()
///     .with_similarity_threshold(0.9);
///
/// let mut dedup = DedupTransformer::new(config).unwrap();
/// ```
pub struct DedupTransformer {
    /// Configuration.
    config: Config,
    /// MinHash signature computer.
    hasher: MinHasher,
    /// LSH index for finding duplicates.
    index: LshIndex,
    /// Buffered samples waiting for processing.
    buffer: Vec<Sample>,
    /// Whether we're in streaming mode.
    pub streaming: bool,
    /// Next document ID.
    next_doc_id: usize,
    /// Output queue for streaming mode.
    output_queue: Vec<Sample>,
    /// Field name containing text to deduplicate on.
    text_field: String,
    /// Whether to mark duplicates instead of filtering.
    mark_duplicates: bool,
    /// Track bypassed documents globally across batches.
    bypassed_samples: std::collections::HashMap<Vec<u8>, usize>,
}

impl DedupTransformer {
    /// Create a new deduplication transformer.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    #[instrument(skip(config), level = "debug")]
    pub fn new(config: Config) -> Result<Self> {
        let hasher = MinHasher::new(&config)?;
        let index = LshIndex::new(&config)?;

        Ok(Self {
            config,
            hasher,
            index,
            buffer: Vec::new(),
            streaming: false,
            next_doc_id: 0,
            output_queue: Vec::new(),
            text_field: "text".to_string(),
            mark_duplicates: false,
            bypassed_samples: std::collections::HashMap::new(),
        })
    }

    /// Set the field name containing text to deduplicate on.
    #[must_use]
    pub fn with_text_field(mut self, field: impl Into<String>) -> Self {
        self.text_field = field.into();
        self
    }

    /// Enable streaming mode (output non-duplicates immediately).
    #[must_use]
    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.streaming = enabled;
        self
    }

    /// Enable marking duplicates instead of filtering them.
    ///
    /// When enabled, duplicates are tagged with a `is_duplicate` field
    /// instead of being removed from the output.
    #[must_use]
    pub fn with_mark_duplicates(mut self, enabled: bool) -> Self {
        self.mark_duplicates = enabled;
        self
    }

    /// Process a single sample.
    ///
    /// Computes the MinHash signature and adds to the LSH index.
    /// Returns true if the sample is unique, false if it's a duplicate.
    ///
    /// # Errors
    ///
    /// Returns an error if signature computation fails.
    #[instrument(skip(self, sample), level = "debug")]
    pub fn process_sample(&mut self, sample: &Sample) -> Result<bool> {
        // Extract text from the configured field
        let text = self.extract_text(sample)?;
        
        if text.is_empty() {
            // Empty documents are considered unique (can't deduplicate)
            return Ok(true);
        }

        let doc_id = self.next_doc_id;
        self.next_doc_id = self.next_doc_id.saturating_add(1);

        // Compute MinHash signature
        let signature = self.hasher.compute_str(&text, doc_id)?;

        // Add to LSH index and get candidates
        let candidates = self.index.insert(signature)?;

        // Check if any candidate is actually a duplicate
        let mut is_duplicate = false;
        for candidate_id in candidates {
            if let Some(sim) = self.index.verify_similarity(candidate_id, doc_id) {
                if sim >= self.config.similarity_threshold {
                    is_duplicate = true;
                    break;
                }
            }
        }

        Ok(!is_duplicate)
    }

    /// Add a sample to the buffer for batch processing, or process immediately
    /// if streaming mode is enabled.
    pub fn push(&mut self, sample: Sample) {
        if self.streaming {
            let doc_id = self.next_doc_id;
            self.next_doc_id = self.next_doc_id.saturating_add(1);

            let mut is_dup = false;
            let mut processed = false;

            if let Ok(text) = self.extract_text(&sample) {
                if !text.is_empty() {
                    if let Ok(sig) = self.hasher.compute_str(&text, doc_id) {
                        if let Ok(candidates) = self.index.insert(sig) {
                            processed = true;
                            for candidate_id in candidates {
                                if let Some(sim) = self.index.verify_similarity(candidate_id, doc_id) {
                                    if sim >= self.config.similarity_threshold {
                                        is_dup = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !processed {
                match sample.get(&self.text_field) {
                    Some(text) => {
                        let raw_bytes = text.as_bytes().to_vec();
                        if let Entry::Vacant(slot) = self.bypassed_samples.entry(raw_bytes) {
                            slot.insert(doc_id);
                        } else {
                            is_dup = true;
                        }
                    }
                    None => {}
                }
            }

            if self.mark_duplicates {
                let tag_val: u8 = if is_dup { 1 } else { 0 };
                let tagged = sample.with(
                    "is_duplicate",
                    tenshift_core::sample::Tensor::u8(vec![tag_val], vec![1]),
                );
                self.output_queue.push(tagged);
            } else if !is_dup {
                self.output_queue.push(sample);
            }
        } else {
            self.buffer.push(sample);
        }
    }

    /// Drain any pending output samples produced in streaming mode.
    pub fn drain_streaming(&mut self) -> Vec<Sample> {
        std::mem::take(&mut self.output_queue)
    }

    /// Process all buffered samples and return deduplicated results.
    ///
    /// This computes signatures for all samples, builds the LSH index,
    /// finds clusters, and returns only unique samples.
    pub fn finish_batch(&mut self) -> Vec<Sample> {
        let mut result = std::mem::take(&mut self.output_queue);

        if self.buffer.is_empty() {
            return result;
        }

        // Assign document IDs for this batch to avoid collisions with prior inserts
        let start_doc_id = self.next_doc_id;
        let batch_end = start_doc_id.saturating_add(self.buffer.len());
        
        let mut uninserted_docs = Vec::new();

        // Process all samples and insert signatures with global doc ids
        for (i, sample) in self.buffer.iter().enumerate() {
            let doc_id = start_doc_id.saturating_add(i);
            let mut inserted = false;
            
            if let Ok(text) = self.extract_text(sample) {
                if !text.is_empty() {
                    // Compute signature; if document is too short, treat as unique
                    if let Ok(sig) = self.hasher.compute_str(&text, doc_id) {
                        if self.index.insert(sig).is_ok() {
                            inserted = true;
                        }
                    }
                }
            }
            
            if !inserted {
                // Document bypassed LSH (empty text, hash error, or no text field).
                match sample.get(&self.text_field) {
                    Some(text) => {
                        let raw_bytes = text.as_bytes().to_vec();
                        if let Entry::Vacant(slot) = self.bypassed_samples.entry(raw_bytes) {
                            slot.insert(doc_id);
                            uninserted_docs.push(doc_id);
                        }
                    }
                    None => {
                        uninserted_docs.push(doc_id);
                    }
                }
            }
        }

        // Advance global doc id counter
        self.next_doc_id = batch_end;

        // Find clusters first (this populates the index's cluster data)
        self.index.find_clusters();
        
        // Get unique indices from LSH and append our bypassed docs
        let mut unique_indices = self.index.get_unique_indices();
        unique_indices.extend(uninserted_docs);
        
        if self.mark_duplicates {
            let unique_set: std::collections::HashSet<usize> = unique_indices.into_iter().collect();
            result.reserve(self.buffer.len());
            for i in 0..self.buffer.len() {
                let doc_id = start_doc_id.saturating_add(i);
                let is_dup = !unique_set.contains(&doc_id);
                let tag_val: u8 = if is_dup { 1 } else { 0 };
                let mut sample = std::mem::take(&mut self.buffer[i]);
                sample = sample.with(
                    "is_duplicate",
                    tenshift_core::sample::Tensor::u8(vec![tag_val], vec![1]),
                );
                result.push(sample);
            }
        } else {
            result.reserve(unique_indices.len());
            for doc_id in unique_indices {
                if doc_id >= start_doc_id && doc_id < batch_end {
                    let buf_idx = doc_id - start_doc_id;
                    result.push(std::mem::take(&mut self.buffer[buf_idx]));
                }
            }
        }

        // Clear buffer since we've processed all samples
        self.buffer.clear();

        result
    }

    /// Get duplicate clusters.
    ///
    /// Returns all detected duplicate clusters. Call after `finish_batch()`
    /// for complete results.
    #[must_use]
    pub fn clusters(&mut self) -> &[DuplicateCluster] {
        self.index.find_clusters()
    }

    /// Get statistics about the deduplication process.
    #[must_use]
    pub fn stats(&self) -> crate::lsh::LshStats {
        self.index.stats()
    }

    /// Get the number of unique documents found.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.index.doc_count() - self.index.duplicate_count()
    }

    /// Get the number of duplicate documents found.
    #[must_use]
    pub fn duplicate_count(&self) -> usize {
        self.index.duplicate_count()
    }

    /// Reset the transformer state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.output_queue.clear();
        self.next_doc_id = 0;
        self.bypassed_samples.clear();
        // Clear the index in place. This previously rebuilt via `LshIndex::new`
        // and SILENTLY kept the old populated index when construction errored
        // (`if let Ok(index) = ...`), leaving a stale index while every other
        // field was reset -> downstream doc_id collisions against ghost entries.
        // `clear()` is infallible and cannot leave a stale/half-reset index, so
        // there is no error to swallow (Law-10).
        self.index.clear();
    }

    /// Extract text from a sample's configured field.
    #[instrument(skip(self, sample), level = "trace")]
    fn extract_text(&self, sample: &Sample) -> Result<String> {
        if let Some(tensor) = sample.get(&self.text_field) {
            // Try to interpret as UTF-8 text
            match tensor.dtype() {
                tenshift_core::sample::DType::U8 | tenshift_core::sample::DType::Bytes => {
                    let bytes = tensor.as_bytes();
                    match std::str::from_utf8(bytes) {
                        Ok(s) => Ok(s.to_string()),
                        Err(_) => Err(Error::InvalidConfig {
                            reason: format!("field '{}' is not valid UTF-8", self.text_field),
                            fix: "ensure text fields contain valid UTF-8".to_string(),
                        }),
                    }
                }
                _ => {
                    warn!(field = %self.text_field, "field is not a text field");
                    Err(Error::InvalidConfig {
                        reason: format!("field '{}' is not a text field", self.text_field),
                        fix: "use U8 or Bytes dtype for text fields".to_string(),
                    })
                }
            }
        } else {
            warn!(field = %self.text_field, "sample missing text field");
            Err(Error::InvalidConfig {
                reason: format!("sample missing text field '{}'", self.text_field),
                fix: format!("ensure samples have a '{}' field", self.text_field),
            })
        }
    }

}


pub mod stateful;
#[cfg(test)]
mod tests;

pub use stateful::StatefulDedupTransform;
