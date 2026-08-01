

use crate::cluster::DuplicateCluster;
use crate::config::Config;
use crate::error::Result;
use tenshift_core::sample::Sample;
use tracing::instrument;

use super::DedupTransformer;

/// A stateful deduplication transform that buffers samples.
///
/// This is the recommended transform for deduplication as it properly
/// handles the stateful nature of duplicate detection.
pub struct StatefulDedupTransform {
    /// Inner transformer.
    inner: DedupTransformer,
    /// Whether we've finished processing.
    finished: bool,
}

impl StatefulDedupTransform {
    /// Create a new stateful deduplication transform.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    #[instrument(skip(config), level = "debug")]
    pub fn new(config: Config) -> Result<Self> {
        Ok(Self {
            inner: DedupTransformer::new(config)?,
            finished: false,
        })
    }

    /// Set the text field name.
    #[must_use]
    pub fn with_text_field(mut self, field: impl Into<String>) -> Self {
        self.inner = self.inner.with_text_field(field);
        self
    }

    /// Enable marking duplicates.
    #[must_use]
    pub fn with_mark_duplicates(mut self, enabled: bool) -> Self {
        self.inner = self.inner.with_mark_duplicates(enabled);
        self
    }

    /// Get duplicate clusters.
    #[must_use]
    pub fn clusters(&mut self) -> &[DuplicateCluster] {
        self.inner.clusters()
    }

    /// Get statistics.
    #[must_use]
    pub fn stats(&self) -> crate::lsh::LshStats {
        self.inner.stats()
    }
}

impl tenshift_core::transform::StatefulTransform for StatefulDedupTransform {
    fn push(&mut self, sample: Sample) -> Vec<Sample> {
        self.inner.push(sample);
        Vec::new() // Buffer all samples, output on finish
    }

    fn finish(&mut self) -> Vec<Sample> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        self.inner.finish_batch()
    }

    fn name(&self) -> &str {
        "stateful_dedup"
    }
}

