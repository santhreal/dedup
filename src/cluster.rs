/// A cluster of duplicate documents.
#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateCluster {
    /// Cluster ID (sequential from 0).
    pub id: usize,
    /// Document indices in this cluster.
    pub indices: Vec<usize>,
    /// Representative index (first in cluster).
    pub representative: usize,
}

impl DuplicateCluster {
    /// Create a new duplicate cluster.
    pub fn new(id: usize, representative: usize) -> Self {
        Self {
            id,
            indices: vec![representative],
            representative,
        }
    }

    /// Add an index to the cluster.
    pub fn add(&mut self, index: usize) {
        self.indices.push(index);
    }

    /// Get the number of documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if cluster has no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Check if cluster has duplicates.
    #[must_use]
    pub fn is_duplicate(&self) -> bool {
        self.len() > 1
    }

    /// Check if this cluster contains a document.
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        self.indices.contains(&index)
    }
}
