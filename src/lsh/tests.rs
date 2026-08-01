    use super::*;
    use crate::cluster::DuplicateCluster;
    use crate::config::Config;
    use crate::minhash::MinHashSignature;

    fn create_index() -> LshIndex {
        let config = Config::default();
        LshIndex::new(&config).unwrap()
    }

    fn create_signature(values: Vec<u32>, doc_id: usize) -> MinHashSignature {
        MinHashSignature::new(values, doc_id)
    }

    #[test]
    fn lsh_index_creation() {
        let index = create_index();
        assert_eq!(index.doc_count(), 0);
        assert_eq!(index.num_bands, 16);
    }

    #[test]
    fn insert_and_query() {
        let mut index = create_index();
        let sig = create_signature((0..128).map(|i| i as u32).collect(), 0);
        
        let candidates = index.insert(sig.clone()).unwrap();
        assert!(candidates.is_empty());
        
        // Same signature should collide
        let sig2 = create_signature((0..128).map(|i| i as u32).collect(), 1);
        let candidates = index.insert(sig2).unwrap();
        assert!(candidates.contains(&0));
    }

    #[test]
    fn query_returns_candidates() {
        let mut index = create_index();
        let sig1 = create_signature(vec![1; 128], 0);
        index.insert(sig1.clone()).unwrap();
        
        let sig2 = create_signature(vec![1; 128], 1);
        let candidates = index.query(&sig2);
        
        assert!(candidates.contains(&0));
    }

    #[test]
    fn verify_similarity_exact() {
        let mut index = create_index();
        let sig1 = create_signature(vec![1; 128], 0);
        index.insert(sig1).unwrap();
        
        let sig2 = create_signature(vec![1; 128], 1);
        index.insert(sig2).unwrap();
        
        let sim = index.verify_similarity(0, 1).unwrap();
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn find_clusters_detects_duplicates() {
        let mut index = create_index();
        
        // Insert identical signatures
        for i in 0..5 {
            let sig = create_signature(vec![42; 128], i);
            index.insert(sig).unwrap();
        }
        
        let clusters = index.find_clusters();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 5);
    }

    #[test]
    fn get_unique_indices() {
        let mut index = create_index();
        
        // Create a cluster of duplicates
        for i in 0..3 {
            let sig = create_signature(vec![42; 128], i);
            index.insert(sig).unwrap();
        }
        
        // Add a unique document
        let unique_sig = create_signature((0..128).map(|i| i as u32).collect(), 3);
        index.insert(unique_sig).unwrap();
        
        index.find_clusters();
        let unique = index.get_unique_indices();
        
        // Should get representative from cluster + unique doc
        assert_eq!(unique.len(), 2);
        assert!(unique.contains(&0)); // Representative
        assert!(unique.contains(&3)); // Unique
    }

    #[test]
    fn cluster_info() {
        let mut cluster = DuplicateCluster::new(0, 0);
        cluster.add(1);
        cluster.add(2);

        assert_eq!(cluster.len(), 3);
        assert!(cluster.contains(1));
        assert!(!cluster.contains(99));
    }

    #[test]
    fn duplicate_cluster() {
        let mut cluster = DuplicateCluster::new(0, 5);
        cluster.add(6);
        cluster.add(7);
        
        assert_eq!(cluster.len(), 3);
        assert!(cluster.is_duplicate());
        assert!(cluster.contains(6));
    }

    #[test]
    fn stats_calculation() {
        let mut index = create_index();
        
        for i in 0..10 {
            let sig = create_signature(vec![i as u32; 128], i);
            index.insert(sig).unwrap();
        }
        
        let stats = index.stats();
        assert_eq!(stats.doc_count, 10);
        assert!(stats.total_buckets > 0);
    }

    #[test]
    fn estimated_recall_at_threshold() {
        let config = Config::default();
        let index = LshIndex::new(&config).unwrap();
        let stats = index.stats();
        
        let recall = stats.estimated_recall();
        // At threshold 0.9 with 16 bands of 8 rows each
        // s^r = 0.9^8 ≈ 0.43
        // 1 - (1 - 0.43)^16 ≈ 0.999
        assert!(recall > 0.9);
    }

    #[test]
    fn invalid_config_rejected() {
        let config = Config::new(100, 16, 5, 0.9).unwrap_err();
        assert!(config.to_string().contains("divisible"));
    }

    #[test]
    fn memory_usage_reported() {
        let mut index = create_index();
        
        for i in 0..100 {
            let sig = create_signature(vec![i as u32; 128], i);
            index.insert(sig).unwrap();
        }
        
        let mem = index.memory_usage();
        assert!(mem > 0);
    }

    #[test]
    fn duplicate_count_correct() {
        let mut index = create_index();
        
        // 3 duplicates + 1 representative = cluster of 4
        for i in 0..4 {
            let sig = create_signature(vec![42; 128], i);
            index.insert(sig).unwrap();
        }
        
        index.find_clusters();
        assert_eq!(index.duplicate_count(), 3);
    }

    #[test]
    fn insert_rejects_wrong_length_signature() {
        let mut index = create_index();
        // Config::default => 16 bands x 8 rows = 128 expected; 64 is malformed.
        let bad = create_signature((0..64).map(|i| i as u32).collect(), 0);
        let result = index.insert(bad);
        assert!(
            matches!(result, Err(crate::error::Error::InvalidConfig { .. })),
            "a signature whose length != num_bands*rows_per_band must be rejected, got {result:?}"
        );
        // A correctly-sized signature is still accepted.
        let good = create_signature((0..128).map(|i| i as u32).collect(), 1);
        assert!(index.insert(good).is_ok());
    }

    #[test]
    fn dense_component_forms_one_cluster_without_losing_nodes() {
        // 20 identical signatures form a fully-connected component. The BFS must
        // enqueue each node exactly once (mark-on-enqueue) and still collect all
        // 20 into a single cluster - proving the quadratic-queue fix preserves
        // correctness.
        let mut index = create_index();
        for i in 0..20 {
            index.insert(create_signature(vec![7; 128], i)).unwrap();
        }
        let clusters = index.find_clusters();
        assert_eq!(clusters.len(), 1, "all identical docs form one cluster");
        assert_eq!(clusters[0].len(), 20, "no node dropped or duplicated");
    }

    #[test]
    fn mixed_corpus_partitions_into_exact_connected_components() {
        // Two disjoint duplicate groups plus a singleton. The union-find path in
        // find_clusters skips verify_similarity for already-connected pairs, so
        // this asserts that pruning still yields the exact connected components:
        // {0,1,2}, {3,4}, and doc 5 alone (excluded, since singletons are not
        // clusters). This is the differential check that the O(n^2)->near-linear
        // change preserves the partition a full all-pairs BFS would produce.
        let mut index = create_index();
        for i in 0..3 {
            index.insert(create_signature(vec![7; 128], i)).unwrap();
        }
        for i in 3..5 {
            index.insert(create_signature(vec![9; 128], i)).unwrap();
        }
        index.insert(create_signature(vec![42; 128], 5)).unwrap();

        let clusters = index.find_clusters();
        assert_eq!(clusters.len(), 2, "two duplicate groups, singleton excluded");

        // Groups are ordered by minimum member: cluster 0 = {0,1,2}, cluster 1 = {3,4}.
        assert_eq!(clusters[0].len(), 3);
        assert!(clusters[0].contains(0) && clusters[0].contains(1) && clusters[0].contains(2));
        assert!(!clusters[0].contains(3) && !clusters[0].contains(5));

        assert_eq!(clusters[1].len(), 2);
        assert!(clusters[1].contains(3) && clusters[1].contains(4));
        assert!(!clusters[1].contains(0) && !clusters[1].contains(5));

        // The unique doc 5 belongs to no cluster.
        assert!(
            !clusters.iter().any(|c| c.contains(5)),
            "unique doc 5 must not be clustered"
        );
    }

    #[test]
    fn reinserting_same_doc_id_never_duplicates_it_in_a_bucket() {
        // Removing the per-insert `contains(&doc_id)` scan is only safe because a
        // re-inserted doc_id is first stripped from its old buckets. Re-insert
        // the SAME doc_id with different signatures and assert no bucket ever
        // holds it twice - the invariant the removed guard used to enforce.
        let mut index = create_index();
        for round in 0..5u32 {
            index
                .insert(create_signature(vec![round; 128], 7))
                .unwrap();
        }
        for band in &index.buckets {
            for ids in band.values() {
                let count = ids.iter().filter(|&&id| id == 7).count();
                assert!(
                    count <= 1,
                    "doc_id 7 must appear at most once per bucket, found {count}"
                );
            }
        }
        // The doc is still indexed exactly once overall (one live signature).
        assert_eq!(index.doc_count(), 1, "re-insert must not inflate doc_count");
        // Its latest signature (all values == 4) still collides on query.
        let candidates = index.query(&create_signature(vec![4u32; 128], 99));
        assert_eq!(
            candidates,
            vec![7],
            "the latest signature must resolve back to doc_id 7"
        );
    }

    #[test]
    fn clear_empties_the_index_leaving_no_stale_entries() {
        // `DedupTransformer::reset` used to rebuild the index via LshIndex::new
        // and silently keep the OLD populated index on a build error. `clear()`
        // replaces that: it must empty every bucket, signature, and cluster in
        // place so a reused index has no ghost entries to collide against.
        let mut index = create_index();
        for i in 0..50 {
            index.insert(create_signature(vec![7u32; 128], i)).unwrap();
        }
        index.find_clusters(); // populate cluster caches too
        assert_eq!(index.doc_count(), 50);

        index.clear();

        assert_eq!(index.doc_count(), 0, "doc_count must be zeroed");
        assert!(
            index.buckets.iter().all(std::collections::HashMap::is_empty),
            "every band bucket must be emptied"
        );
        assert!(index.signatures.is_empty(), "signatures must be emptied");
        assert!(index.clusters.is_empty(), "cluster cache must be emptied");
        // The band structure is preserved (not torn down), so the index is
        // immediately reusable.
        assert_eq!(index.num_bands, 16, "band structure is preserved");

        // Reuse after clear: a fresh doc collides only with newly inserted docs,
        // never with the 50 cleared ghosts.
        index.insert(create_signature(vec![7u32; 128], 0)).unwrap();
        let candidates = index.query(&create_signature(vec![7u32; 128], 99));
        assert_eq!(
            candidates,
            vec![0],
            "after clear, only the one re-inserted doc collides - no stale entries"
        );
    }

    #[test]
    fn inserting_100k_similar_documents_completes_under_two_seconds() {
        // The removed O(bucket) `contains` scan made inserting into colliding
        // buckets more expensive per insert. Insert 100k documents arranged as
        // near-duplicate groups (each doc has same-group dupes, distinct across
        // groups) and assert the whole load finishes well under two seconds -
        // the scalability the contains-scan removal protects. Groups are kept
        // small so the (inherent, separate) per-insert candidate collection does
        // not dominate the measurement.
        const GROUPS: u32 = 25_000;
        const COPIES: usize = 4;

        let mut index = create_index();
        let start = std::time::Instant::now();
        for g in 0..GROUPS {
            for c in 0..COPIES {
                let doc_id = g as usize * COPIES + c;
                index
                    .insert(create_signature(vec![g; 128], doc_id))
                    .unwrap();
            }
        }
        let elapsed = start.elapsed();

        assert_eq!(index.doc_count(), GROUPS as usize * COPIES, "all 100k indexed");
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "inserting 100k similar documents took {elapsed:?}; the O(N) contains scan \
             (now removed) made this O(N^2)"
        );

        // Real-value collision check: each of a group's 100 copies collides with
        // the other 99, so querying the group signature returns exactly 99
        // candidates (self excluded).
        let candidates = index.query(&create_signature(vec![0u32; 128], 0));
        assert_eq!(
            candidates.len(),
            COPIES - 1,
            "each copy must collide with the other 99 in its group"
        );
    }
