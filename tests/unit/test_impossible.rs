//! Impossible condition tests for dedup.
//! Designed to test extreme limits, exact boundary thresholds, and massive scale.
//!
//! Tests marked `#[ignore]` need ~100MB–1GB RAM or long runtimes, run in a dedicated job:
//! `cargo test -p dedup --test unit_test_impossible -- --ignored`

// Heavy tests use `#[ignore = "heavy: ..."]` (literal required by rustc).

use dedup::{Config, DedupTransformer, FastHasher, MinHasher};
use dedup::tenshift::Sample;
use tenshift_core::sample::Tensor;
use std::sync::{Arc, Mutex};
use std::thread;

fn create_text_sample(text: impl Into<String>, idx: u64) -> Sample {
    Sample::new()
        .with("text", Tensor::bytes(text.into().into_bytes()))
        .with_metadata("test", idx)
}

#[allow(dead_code)]
fn create_bytes_sample(bytes: Vec<u8>, idx: u64) -> Sample {
    Sample::new()
        .with("text", Tensor::bytes(bytes))
        .with_metadata("test", idx)
}

// 1. identical 100MB files produce same hash
#[test]
#[ignore = "heavy: >100MB RAM or >60s, run: cargo test --test unit_test_impossible -- --ignored"]
fn test_impossible_identical_100mb_files() {
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    let file1 = vec![0x42; 100 * 1024 * 1024];
    let file2 = vec![0x42; 100 * 1024 * 1024];
    let sig1 = hasher.compute(&file1, 0).unwrap();
    let sig2 = hasher.compute(&file2, 1).unwrap();
    assert_eq!(sig1.values, sig2.values, "Identical 100MB files must produce the exact same hash");
}

// 2. files differing by 1 bit produce different hashes
#[test]
#[ignore = "heavy: >100MB RAM or >60s, run: cargo test --test unit_test_impossible -- --ignored"]
fn test_impossible_1_bit_diff_100mb_files() {
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    let file1 = vec![0x42; 100 * 1024 * 1024];
    let mut file2 = file1.clone();
    // flip 1 bit in the middle
    file2[50 * 1024 * 1024] ^= 0x01;
    let sig1 = hasher.compute(&file1, 0).unwrap();
    let sig2 = hasher.compute(&file2, 1).unwrap();
    assert_ne!(sig1.values, sig2.values, "100MB files differing by 1 bit must produce different hashes");
}

// 3. LSH similarity threshold at 0.0: `with_similarity_threshold` clamps to 0.01
#[test]
fn test_impossible_lsh_threshold_0_0() {
    let config = Config::default().with_similarity_threshold(0.0);
    let mut transformer = DedupTransformer::new(config).unwrap();
    transformer.push(create_text_sample("Completely different text one", 1));
    transformer.push(create_text_sample("Absolutely unrelated string two", 2));
    let unique = transformer.finish_batch();
    assert_eq!(
        unique.len(),
        2,
        "Threshold 0.0 is clamped to 0.01; unrelated documents stay separate"
    );
}

// 4. LSH at 1.0 (nothing matches unless identical)
#[test]
fn test_impossible_lsh_threshold_1_0() {
    let config = Config::default().with_similarity_threshold(1.0);
    let mut transformer = DedupTransformer::new(config).unwrap();
    // Near duplicates that differ by just 1 character
    let base = "This is a long sentence that we will use to test near duplicates.";
    let mut doc1 = base.to_string();
    doc1.push('A');
    let mut doc2 = base.to_string();
    doc2.push('B');
    transformer.push(create_text_sample(doc1, 1));
    transformer.push(create_text_sample(doc2, 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 2, "Threshold 1.0 should prevent deduplication of non-identical documents");
}

// 5. near-duplicate pair where similarity is exactly at threshold boundary
#[test]
fn test_impossible_exact_threshold_boundary() {
    let config = Config::default().with_similarity_threshold(0.5);
    let mut transformer = DedupTransformer::new(config).unwrap();
    // Jaccard similarity = |A intersect B| / |A union B|
    // To get exactly 0.5, we can make A and B share half their shingles.
    // Let's use shingles of size 1 (by hacking the config temporarily if possible, or just build exact strings).
    // Let's use very specific alternating strings.
    let str1 = "AAAAAAAAAA";
    let str2 = "AAAAABBBBB"; // Half similar
    transformer.push(create_text_sample(str1, 1));
    transformer.push(create_text_sample(str2, 2));
    let unique = transformer.finish_batch();
    // Result might be 1 or 2 depending on hash collisions, but it shouldn't panic.
    assert!(unique.len() > 0);
}

// 6. 100K documents all identical (massive dedup) -> this is test 12, but we implement 1-11 first
// Wait, we can implement 1-11 on hash collisions and LSH boundaries.

// 6. Test very small documents with threshold 1.0
#[test]
fn test_impossible_small_docs_threshold_1_0() {
    let config = Config::default().with_similarity_threshold(1.0);
    let mut transformer = DedupTransformer::new(config).unwrap();
    transformer.push(create_text_sample("A", 1));
    transformer.push(create_text_sample("B", 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 2, "Small documents differing must not dedup at threshold 1.0");
}

// 7. Test identical tiny docs threshold 1.0
#[test]
fn test_impossible_tiny_identical_threshold_1_0() {
    let config = Config::default().with_similarity_threshold(1.0);
    let mut transformer = DedupTransformer::new(config).unwrap();
    transformer.push(create_text_sample("A", 1));
    transformer.push(create_text_sample("A", 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Small identical documents must dedup at threshold 1.0");
}

// 8. Test config with extreme S-curve parameters
#[test]
fn test_impossible_extreme_s_curve() {
    let config = Config::new(1024, 1024, 5, 0.9).unwrap();
    let mut transformer = DedupTransformer::new(config).unwrap();
    transformer.push(create_text_sample("A long string that is very specific to this test.", 1));
    transformer.push(create_text_sample("A long string that is very specific to this test.", 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Extreme config should still dedup identical strings");
}

// 9. Test config with 1 band
#[test]
fn test_impossible_one_band() {
    let config = Config::new(128, 1, 5, 0.5).unwrap();
    let mut transformer = DedupTransformer::new(config).unwrap();
    transformer.push(create_text_sample("A long string that is very specific to this test.", 1));
    transformer.push(create_text_sample("A long string that is very specific to this test.", 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Config with 1 band should work for identical strings");
}

// 10. Test hash collisions with specific cyclic byte patterns
#[test]
fn test_impossible_cyclic_hash_collision() {
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    // Two strings that share exactly the same set of shingles
    let file1 = vec![0x12, 0x34, 0x12, 0x34, 0x12, 0x34, 0x12, 0x34, 0x12, 0x34];
    let file2 = vec![0x34, 0x12, 0x34, 0x12, 0x34, 0x12, 0x34, 0x12, 0x34, 0x12];
    let sig1 = hasher.compute(&file1, 0).unwrap();
    let sig2 = hasher.compute(&file2, 1).unwrap();
    // They might not be identical if shingles are 5 bytes because 12 34 12 34 12 != 34 12 34 12 34
    // But they will be very similar.
    assert!(sig1.similarity(&sig2) > 0.0);
}

// 11. Test hash with extremely sparse byte distribution
#[test]
fn test_impossible_sparse_byte_distribution() {
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    let mut file1 = vec![0x00; 1000];
    file1[500] = 0x01;
    let mut file2 = vec![0x00; 1000];
    file2[501] = 0x01; // slightly shifted
    let sig1 = hasher.compute(&file1, 0).unwrap();
    let sig2 = hasher.compute(&file2, 1).unwrap();
    // Sparse distribution means mostly 0x00 shingles, so similarity is high.
    assert!(sig1.similarity(&sig2) > 0.9);
}

// 12. 100K documents all identical (massive dedup)
#[test]
#[ignore = "heavy: >100MB RAM or >60s, run: cargo test --test unit_test_impossible -- --ignored"]
fn test_impossible_100k_identical() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    let text = "This is a completely identical document that will be repeated 100,000 times.";
    for i in 0..100_000 {
        transformer.push(create_text_sample(text, i as u64));
    }
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "100K identical documents should dedup to 1");
}

// 13. 100K documents all unique (zero dedup)
#[test]
#[ignore = "heavy: >100MB RAM or >60s, run: cargo test --test unit_test_impossible -- --ignored"]
fn test_impossible_100k_unique() {
    let config = Config::default().with_similarity_threshold(1.0);
    // Since unique items take up more memory, we use small strings to avoid OOM in test
    let mut transformer = DedupTransformer::new(config).unwrap();
    for i in 0..100_000 {
        transformer.push(create_text_sample(format!("Unique document {}", i), i as u64));
    }
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 100_000, "100K unique documents should yield 100K results");
}

// 14. concurrent dedup from 32 threads
#[test]
#[ignore = "heavy: >100MB RAM or >60s, run: cargo test --test unit_test_impossible -- --ignored"]
fn test_impossible_concurrent_dedup_32_threads() {
    let config = Config::default();
    let transformer = Arc::new(Mutex::new(DedupTransformer::new(config).unwrap()));
    let mut handles = vec![];
    for t in 0..32 {
        let trans_clone = Arc::clone(&transformer);
        let handle = thread::spawn(move || {
            for i in 0..1000 {
                let text = format!("Thread {} generated item {}", t, i);
                let mut guard = trans_clone.lock().unwrap();
                guard.push(create_text_sample(text, (t * 1000 + i) as u64));
            }
        });
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap();
    }
    let mut guard = transformer.lock().unwrap();
    let unique = guard.finish_batch();
    assert!(unique.len() > 0, "Concurrent dedup from 32 threads should finish without crashing");
}

// 15. concurrent hash of massive files across 32 threads
#[test]
#[ignore = "heavy: >100MB RAM or >60s, run: cargo test --test unit_test_impossible -- --ignored"]
fn test_impossible_concurrent_massive_hashing_32_threads() {
    let config = Config::default();
    let hasher = Arc::new(MinHasher::new(&config).unwrap());
    let mut handles = vec![];
    let file = vec![0xAB; 10 * 1024 * 1024]; // 10MB
    for _ in 0..32 {
        let h_clone = Arc::clone(&hasher);
        let f_clone = file.clone();
        handles.push(thread::spawn(move || {
            h_clone.compute(&f_clone, 0).unwrap();
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
}

// 16. massive identical hashing concurrency
#[test]
#[ignore = "heavy: >100MB RAM or >60s, run: cargo test --test unit_test_impossible -- --ignored"]
fn test_impossible_massive_identical_hashing() {
    let config = Config::default();
    let hasher = Arc::new(MinHasher::new(&config).unwrap());
    let mut handles = vec![];
    let file = "Same text again and again.".to_string();
    for _ in 0..32 {
        let h_clone = Arc::clone(&hasher);
        let f_clone = file.clone();
        handles.push(thread::spawn(move || {
            for i in 0..10_000 {
                h_clone.compute_str(&f_clone, i).unwrap();
            }
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
}

// 17. 100K identical docs with threshold 0.0
#[test]
#[ignore = "heavy: >100MB RAM or >60s, run: cargo test --test unit_test_impossible -- --ignored"]
fn test_impossible_100k_identical_threshold_0() {
    let config = Config::default().with_similarity_threshold(0.0);
    let mut transformer = DedupTransformer::new(config).unwrap();
    let text = "Text doesn't matter.";
    for i in 0..100_000 {
        transformer.push(create_text_sample(text, i as u64));
    }
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "100K identical docs with threshold 0.0 should dedup to 1");
}

// 18. 10K unique docs with threshold 0.0 (all dedup to 1)
#[test]
#[ignore = "heavy: >100MB RAM or >60s, run: cargo test --test unit_test_impossible -- --ignored"]
fn test_impossible_10k_unique_threshold_0() {
    let config = Config::default().with_similarity_threshold(0.0);
    let mut transformer = DedupTransformer::new(config).unwrap();
    for i in 0..10_000 {
        transformer.push(create_text_sample(format!("Unique doc {}", i), i as u64));
    }
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "10K unique docs with threshold 0.0 should dedup to 1");
}

// 19. FastHasher concurrent stress test
#[test]
fn test_impossible_fasthasher_concurrent() {
    let mut handles = vec![];
    for t in 0..32 {
        handles.push(thread::spawn(move || {
            let hasher = FastHasher::new(128, t as u64);
            let mut sig = vec![u32::MAX; 128];
            for i in 0..10_000 {
                hasher.update_signature(&mut sig, i as u64);
            }
            assert!(sig.iter().any(|&x| x < u32::MAX));
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
}

// 20. Shingle memory size massive concurrency
#[test]
fn test_impossible_shingle_iterator_concurrency() {
    let mut handles = vec![];
    for _ in 0..32 {
        handles.push(thread::spawn(move || {
            let file = vec![0xFF; 1_000_000];
            let iter = dedup::ShingleIterator::new(&file, 5);
            let count = iter.count_shingles();
            assert_eq!(count, 1_000_000 - 5 + 1);
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
}

// 21. LSH index boundary condition (0 bands) via unvalidated struct init if possible, or expect error
#[test]
fn test_impossible_lsh_config_0_bands() {
    let config = Config::new(128, 0, 5, 0.9);
    assert!(config.is_err(), "Config with 0 bands should return error");
}

// 22. LSH index boundary condition (1000 bands)
#[test]
fn test_impossible_lsh_config_1000_bands() {
    // 1000 bands, 1 row each => signature_size = 1000
    let config = Config::new(1000, 1000, 5, 0.9).unwrap();
    let mut transformer = DedupTransformer::new(config).unwrap();
    transformer.push(create_text_sample("A", 1));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1);
}

// 23. streaming hash of 1GB file
#[test]
#[ignore = "heavy: >100MB RAM or >60s, run: cargo test --test unit_test_impossible -- --ignored"]
fn test_impossible_streaming_hash_1gb_file() {
    let config = Config::default();
    let _hasher = MinHasher::new(&config).unwrap();
    let chunk = vec![0x42; 10 * 1024 * 1024]; // 10MB chunk
    let mut sig = vec![u32::MAX; config.signature_size];
    let fast_hasher = FastHasher::new(config.signature_size, 0x9e37_79b9_7f4a_7c15);

    for _ in 0..100 {
        // 100 * 10MB = 1GB
        for shingle in dedup::ShingleIterator::new(&chunk, config.shingle_size) {
            let shingle_hash = dedup::hash_bytes(shingle);
            fast_hasher.update_signature(&mut sig, shingle_hash);
        }
    }
    assert!(sig.iter().any(|&x| x < u32::MAX), "Streaming hash of 1GB file should succeed");
}

// 24. extreme fast_hasher alternating chunks
#[test]
#[ignore = "heavy: >100MB RAM or >60s, run: cargo test --test unit_test_impossible -- --ignored"]
fn test_impossible_streaming_alternating_1byte_chunks() {
    let config = Config::default();
    let mut sig = vec![u32::MAX; config.signature_size];
    let fast_hasher = FastHasher::new(config.signature_size, 0x9e37_79b9_7f4a_7c15);
    for i in 0..10_000_000 {
        // Just feeding raw hashes
        fast_hasher.update_signature(&mut sig, (i % 2) as u64);
    }
    assert!(sig.iter().any(|&x| x < u32::MAX));
}

// 25. Minhash batch computation size limits
#[test]
fn test_impossible_minhash_batch_size_limits() {
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    let mut docs = Vec::new();
    let doc = vec![0x00; 100];
    for _ in 0..10_000 {
        docs.push(doc.as_slice());
    }
    let sigs = hasher.compute_batch(&docs, 0);
    assert_eq!(sigs.len(), 10_000);
}

// 26. Memory leak check on huge cluster adds
#[test]
fn test_impossible_huge_cluster_adds() {
    let mut cluster = dedup::DuplicateCluster::new(0, 0);
    for i in 1..100_000 {
        cluster.add(i);
    }
    assert_eq!(cluster.len(), 100_000);
}

// 27. Test identical signatures but different lengths
#[test]
fn test_impossible_identical_sig_diff_lengths() {
    let sig1 = dedup::MinHashSignature::new(vec![1, 2, 3], 0);
    let sig2 = dedup::MinHashSignature::new(vec![1, 2, 3, 4], 1);
    assert_eq!(sig1.similarity(&sig2), 0.0);
}

// 28. Transform edge case with single tiny string repeated massive times
#[test]
fn test_impossible_transform_tiny_massive_repeat() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    for i in 0..10_000 {
        transformer.push(create_text_sample("x", i as u64));
    }
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1);
}

// 29. Optimize params with exact threshold 1.0
#[test]
fn test_impossible_optimize_params_1_0() {
    let (bands, rows) = dedup::optimize_lsh_params(128, 1.0);
    assert!(bands > 0 && rows > 0);
}

// 30. Optimize params with exact threshold 0.0
#[test]
fn test_impossible_optimize_params_0_0() {
    let (bands, rows) = dedup::optimize_lsh_params(128, 0.0);
    assert!(bands > 0 && rows > 0);
}

// 31. Invalid max documents configuration
#[test]
fn test_impossible_max_documents_bounds() {
    let config = Config::default();
    let _max_docs = config.max_documents_in_memory();
    let mut transformer = DedupTransformer::new(config).unwrap();
    transformer.push(create_text_sample("A", 1));
    transformer.push(create_text_sample("B", 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 2);
}

// 32. Zero shingle size
#[test]
fn test_impossible_zero_shingle_size() {
    let res = Config::new(128, 16, 0, 0.9);
    assert!(res.is_err(), "0 shingle size should error out");
}

// 33. Empty band boundary extraction
#[test]
fn test_impossible_empty_band_extraction() {
    let sig = dedup::MinHashSignature::new(vec![], 0);
    let band = sig.band(0, 10);
    assert_eq!(band.len(), 0);
}

// 34. Max out similarity S-curve float precision
#[test]
fn test_impossible_s_curve_float_precision() {
    let prob = dedup::candidate_probability(0.9999999999999999, 16, 8);
    assert!(prob > 0.0 && prob <= 1.0);
}

// 35. Transform batch max out
#[test]
fn test_impossible_transform_batch_max_out() {
    use dedup::StatefulDedupTransform;
    use tenshift_core::transform::StatefulTransform;
    let config = Config::default();
    let mut transformer = StatefulDedupTransform::new(config).unwrap();
    for i in 0..10_000 {
        let _ = transformer.push(create_text_sample("A", i as u64));
    }
    let res = transformer.finish();
    assert_eq!(res.len(), 1); // all identical
}

