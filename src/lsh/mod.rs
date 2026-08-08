//! Locality Sensitive Hashing (LSH) for efficient candidate pair detection.
//!
//! LSH bands the MinHash signature such that similar documents are likely to
//! collide in at least one bucket. This reduces the O(n²) pairwise comparison
//! to O(n) bucket lookups.

use crate::cluster::DuplicateCluster;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::minhash::MinHashSignature;

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use tracing::{instrument, warn};

/// LSH index for finding candidate similar document pairs.
///
/// Documents are inserted into buckets based on their band hashes.
/// Documents in the same bucket are candidate pairs for similarity checking.
pub struct LshIndex {
    /// Number of bands.
    num_bands: usize,
    /// Rows per band.
    rows_per_band: usize,
    /// Similarity threshold.
    threshold: f64,
    /// Bucket storage: band_index -> bucket_hash -> doc_indices
    buckets: Vec<HashMap<u64, Vec<usize>>>,
    /// All signatures stored for verification.
    signatures: std::collections::BTreeMap<usize, MinHashSignature>,
    /// Document count.
    doc_count: usize,
    /// Duplicate clusters found.
    clusters: Vec<DuplicateCluster>,
    /// Map from doc index to cluster id.
    doc_to_cluster: HashMap<usize, usize>,
    /// Next cluster ID.
    next_cluster_id: usize,
    /// Maximum document ID seen (tracks range of document IDs).
    max_doc_id: usize,
}

impl LshIndex {
    /// Create a new LSH index from configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration has incompatible parameters.
    #[instrument(skip(config), level = "debug")]
    pub fn new(config: &Config) -> Result<Self> {
        if config.signature_size % config.num_bands != 0 {
            warn!(
                signature_size = config.signature_size,
                num_bands = config.num_bands,
                "signature_size not divisible by num_bands"
            );
            return Err(Error::InvalidConfig {
                reason: format!(
                    "signature_size ({}) not divisible by num_bands ({})",
                    config.signature_size, config.num_bands
                ),
                fix: "ensure signature_size = num_bands * rows_per_band".to_string(),
            });
        }

        let rows_per_band = config.signature_size / config.num_bands;
        let buckets: Vec<HashMap<u64, Vec<usize>>> = 
            (0..config.num_bands).map(|_| HashMap::new()).collect();

        Ok(Self {
            num_bands: config.num_bands,
            rows_per_band,
            threshold: config.similarity_threshold,
            buckets,
            signatures: std::collections::BTreeMap::new(),
            doc_count: 0,
            clusters: Vec::new(),
            doc_to_cluster: HashMap::new(),
            next_cluster_id: 0,
            max_doc_id: 0,
        })
    }

    /// Clear all indexed documents and derived cluster state, keeping the band
    /// and row structure and the similarity threshold.
    ///
    /// This is the infallible way to empty an index for reuse. Unlike rebuilding
    /// via [`Self::new`] (which is fallible and, when a caller swallowed its
    /// error, could leave a stale populated index while surrounding state was
    /// reset), `clear()` cannot fail and cannot leave the index half-reset.
    pub fn clear(&mut self) {
        for band in &mut self.buckets {
            band.clear();
        }
        self.signatures.clear();
        self.doc_count = 0;
        self.clusters.clear();
        self.doc_to_cluster.clear();
        self.next_cluster_id = 0;
        self.max_doc_id = 0;
    }

    /// Insert a signature into the LSH index.
    ///
    /// Returns a list of candidate similar document indices that collided
    /// in at least one bucket.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] when `doc_id` exceeds the configured maximum
    /// (prevents unbounded allocation from adversarial identifiers).
    #[instrument(skip(self, signature), fields(doc_id = signature.doc_id), level = "debug")]
    pub fn insert(&mut self, signature: MinHashSignature) -> Result<Vec<usize>> {
        let doc_id = signature.doc_id;

        // Guard against adversarial doc_ids that would cause unbounded allocation.
        // 100M documents × ~1KB per signature = ~100GB, which is the practical limit.
        const MAX_DOC_ID: usize = 100_000_000;
        if doc_id > MAX_DOC_ID {
            return Err(Error::InvalidConfig {
                reason: format!("doc_id {doc_id} exceeds maximum {MAX_DOC_ID}"),
                fix: "use sequential doc_ids starting from 0".to_string(),
            });
        }

        // Reject malformed signatures. band_hash slices [start, start+rows) and
        // silently CLAMPS an out-of-range end, so a signature shorter than
        // num_bands * rows_per_band would be indexed under clamped/overlapping
        // band hashes, corrupting candidate recall without any error. Validate the
        // length up front instead.
        let expected_len = self.num_bands * self.rows_per_band;
        if signature.len() != expected_len {
            return Err(Error::InvalidConfig {
                reason: format!(
                    "signature length {} does not match index configuration ({} bands x {} rows = {expected_len})",
                    signature.len(),
                    self.num_bands,
                    self.rows_per_band
                ),
                fix: "generate signatures with the signature_size the index was configured for"
                    .to_string(),
            });
        }

        // Determine whether a signature was present before
        let had_signature = self.signatures.contains_key(&doc_id);

        // If there was an old signature, remove this doc_id from its buckets
        if had_signature {
            if let Some(old_sig) = self.signatures.remove(&doc_id) {
                for band_idx in 0..self.num_bands {
                    let start = band_idx * self.rows_per_band;
                    let old_hash = old_sig.band_hash(start, self.rows_per_band);
                    if let Some(vec) = self.buckets[band_idx].get_mut(&old_hash) {
                        vec.retain(|&id| id != doc_id);
                        if vec.is_empty() {
                            self.buckets[band_idx].remove(&old_hash);
                        }
                    }
                }
            }
        }

        // Store the new signature and update counts
        self.signatures.insert(doc_id, signature.clone());
        if !had_signature {
            self.doc_count += 1;
        }
        self.max_doc_id = self.max_doc_id.max(doc_id);

        // Index changed: invalidate any cached clusters
        if !self.clusters.is_empty() || !self.doc_to_cluster.is_empty() {
            self.clusters.clear();
            self.doc_to_cluster.clear();
            self.next_cluster_id = 0;
        }

        let mut candidates = std::collections::HashSet::new();

        // For each band, compute bucket hash and find collisions
        for band_idx in 0..self.num_bands {
            let start = band_idx * self.rows_per_band;
            let band_hash = signature.band_hash(start, self.rows_per_band);

            let bucket = &mut self.buckets[band_idx];

            match bucket.entry(band_hash) {
                Entry::Occupied(mut entry) => {
                    // Add all existing documents as candidates
                    for &existing_id in entry.get() {
                        if existing_id != doc_id {
                            candidates.insert(existing_id);
                        }
                    }
                    // Cap at 10K entries per bucket to prevent OOM from hash collisions.
                    //
                    // The old `!entry.get().contains(&doc_id)` guard was a linear
                    // O(bucket) scan on EVERY insert, making insertion into large
                    // colliding buckets O(N^2). It was also redundant: `doc_id`
                    // cannot already be in this bucket here - a fresh doc was never
                    // inserted, and a re-inserted doc_id had its old band entries
                    // removed by the `had_signature` cleanup above (lines ~130-143),
                    // so within one band it is pushed at most once per insert. Only
                    // the size cap remains, keeping the push O(1).
                    const MAX_BUCKET_SIZE: usize = 10_000;
                    if entry.get().len() < MAX_BUCKET_SIZE {
                        entry.get_mut().push(doc_id);
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(vec![doc_id]);
                }
            }
        }

        Ok(candidates.into_iter().collect())
    }

    /// Query for candidate similar documents.
    ///
    /// Returns document indices that collided with the given signature
    /// in at least one LSH bucket.
    pub fn query(&self, signature: &MinHashSignature) -> Vec<usize> {
        let expected_len = self.num_bands * self.rows_per_band;
        if signature.len() != expected_len {
            warn!(
                sig_len = signature.len(),
                expected_len = expected_len,
                "LshIndex::query called with signature length mismatched with index configuration"
            );
            return Vec::new();
        }

        let mut candidates = std::collections::HashSet::new();
        for band_idx in 0..self.num_bands {
            let start = band_idx * self.rows_per_band;
            let band_hash = signature.band_hash(start, self.rows_per_band);

            if let Some(bucket) = self.buckets[band_idx].get(&band_hash) {
                for &doc_id in bucket {
                    if doc_id != signature.doc_id {
                        candidates.insert(doc_id);
                    }
                }
            }
        }

        candidates.into_iter().collect()
    }

    /// Verify similarity between two documents using their signatures.
    #[must_use]
    pub fn verify_similarity(&self, doc_a: usize, doc_b: usize) -> Option<f64> {
        let sig_a = self.signatures.get(&doc_a)?;
        let sig_b = self.signatures.get(&doc_b)?;
        Some(sig_a.similarity(sig_b))
    }

    /// Find all duplicate clusters in the index.
    ///
    /// This performs pairwise verification of all candidate pairs
    /// and groups documents into clusters.
    #[instrument(skip(self), level = "debug")]
    pub fn find_clusters(&mut self) -> &[DuplicateCluster] {
        if !self.clusters.is_empty() {
            return &self.clusters;
        }

        // Path-compressed union-find over doc ids. Skipping verify_similarity
        // for pairs already in the same component prunes the redundant all-pairs
        // work in dense buckets (toward near-linear vs O(n^2)) while producing
        // identical connected components: every component-MERGING edge is still
        // verified (find differs -> verify -> union); only redundant intra-
        // component edges are skipped, which cannot change the components.
        let mut parent: HashMap<usize, usize> =
            self.signatures.keys().map(|&d| (d, d)).collect();

        for (doc_id, signature) in &self.signatures {
            let doc_id = *doc_id;

            // Get candidates from LSH
            let candidates = self.query(signature);

            for &candidate_id in &candidates {
                if candidate_id <= doc_id {
                    continue; // Avoid duplicate checking
                }

                // Already in the same component: this edge is redundant for
                // connected components, so skip the similarity computation.
                if uf_find(&mut parent, doc_id) == uf_find(&mut parent, candidate_id) {
                    continue;
                }

                // Verify actual similarity, and union on a real edge.
                if let Some(sim) = self.verify_similarity(doc_id, candidate_id) {
                    if sim >= self.threshold {
                        uf_union(&mut parent, doc_id, candidate_id);
                    }
                }
            }
        }

        // Group doc ids by their component root. Sort each group and order the
        // groups by their minimum member so cluster ids are assigned
        // deterministically (the previous HashMap-seeded BFS numbered clusters
        // in nondeterministic order; the partition itself is unchanged).
        let all_docs: Vec<usize> = self.signatures.keys().copied().collect();
        let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
        for doc_id in all_docs {
            let root = uf_find(&mut parent, doc_id);
            components.entry(root).or_default().push(doc_id);
        }
        let mut groups: Vec<Vec<usize>> = components.into_values().collect();
        for g in &mut groups {
            g.sort_unstable();
        }
        groups.sort_unstable_by_key(|g| g[0]);

        for cluster_docs in groups {
            if cluster_docs.len() > 1 {
                // Minimum doc_id is the representative (deterministic).
                let mut cluster = DuplicateCluster::new(self.next_cluster_id, cluster_docs[0]);
                self.doc_to_cluster.insert(cluster_docs[0], self.next_cluster_id);
                for &doc in &cluster_docs[1..] {
                    cluster.add(doc);
                    self.doc_to_cluster.insert(doc, self.next_cluster_id);
                }
                self.clusters.push(cluster);
                self.next_cluster_id += 1;
            }
        }

        &self.clusters
    }

    /// Get the cluster for a document index.
    #[must_use]
    pub fn get_cluster_for_doc(&self, doc_id: usize) -> Option<&DuplicateCluster> {
        let cluster_id = self.doc_to_cluster.get(&doc_id)?;
        self.clusters.get(*cluster_id)
    }

    /// Check if a document is a duplicate (belongs to any cluster).
    #[must_use]
    pub fn is_duplicate(&self, doc_id: usize) -> bool {
        self.doc_to_cluster.contains_key(&doc_id)
    }

    /// Get all unique documents (first in each cluster + non-duplicate documents).
    pub fn get_unique_indices(&self) -> Vec<usize> {
        let mut unique: Vec<usize> = Vec::new();
        let mut in_cluster = std::collections::HashSet::new();

        // Add representatives from clusters
        for cluster in &self.clusters {
            unique.push(cluster.representative);
            for &idx in &cluster.indices {
                in_cluster.insert(idx);
            }
        }

        // Add non-clustered documents (only those that actually have signatures)
        for &doc_id in self.signatures.keys() {
            if !in_cluster.contains(&doc_id) {
                unique.push(doc_id);
            }
        }

        unique.sort_unstable();
        unique
    }

    /// Get the total number of documents indexed.
    #[must_use]
    pub const fn doc_count(&self) -> usize {
        self.doc_count
    }

    /// Get the number of duplicate clusters.
    #[must_use]
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// Get the number of duplicate documents (documents in clusters, excluding representatives).
    #[must_use]
    pub fn duplicate_count(&self) -> usize {
        self.clusters.iter().map(|c| c.len().saturating_sub(1)).sum()
    }

    /// Get statistics about the LSH index.
    pub fn stats(&self) -> LshStats {
        let total_buckets: usize = self.buckets.iter().map(std::collections::HashMap::len).sum();
        let total_entries: usize = self.buckets.iter().map(|b| b.values().map(std::vec::Vec::len).sum::<usize>()).sum();

        LshStats {
            num_bands: self.num_bands,
            rows_per_band: self.rows_per_band,
            threshold: self.threshold,
            doc_count: self.doc_count,
            total_buckets,
            total_entries,
            avg_bucket_size: if total_buckets > 0 {
                total_entries as f64 / total_buckets as f64
            } else {
                0.0
            },
            cluster_count: self.clusters.len(),
            duplicate_count: self.duplicate_count(),
        }
    }

    /// Estimate memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let signature_bytes = self.signatures.len() * (std::mem::size_of::<usize>() + std::mem::size_of::<MinHashSignature>() + 32); // Approximate BTreeMap overhead
        let bucket_bytes: usize = self.buckets.iter()
            .map(|b| {
                b.capacity() * (std::mem::size_of::<u64>() + std::mem::size_of::<Vec<usize>>()) +
                b.values().map(|v| v.capacity() * std::mem::size_of::<usize>()).sum::<usize>()
            })
            .sum();
        let cluster_bytes = self.clusters.len() * std::mem::size_of::<DuplicateCluster>();
        
        signature_bytes + bucket_bytes + cluster_bytes
    }
}

/// Path-compressed union-find `find` over a doc-id -> parent map.
///
/// Returns the component root of `x`, compressing the path so future lookups
/// are near-constant. A doc id absent from `parent` is treated as its own root
/// (it never panics on a missing key).
fn uf_find(parent: &mut HashMap<usize, usize>, x: usize) -> usize {
    let mut root = x;
    while let Some(&p) = parent.get(&root) {
        if p == root {
            break;
        }
        root = p;
    }
    // Point every node on the path directly at the root.
    let mut cur = x;
    while let Some(&p) = parent.get(&cur) {
        if p == root {
            break;
        }
        parent.insert(cur, root);
        cur = p;
    }
    root
}

/// Union the components of `a` and `b`, keeping the smaller root as the
/// representative so component roots stay deterministic across runs.
fn uf_union(parent: &mut HashMap<usize, usize>, a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        let (keep, drop) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent.insert(drop, keep);
    }
}

pub mod stats;
#[cfg(test)]
mod tests;

pub use stats::LshStats;
