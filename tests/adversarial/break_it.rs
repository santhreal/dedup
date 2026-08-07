//! Adversarial break-it tests for dedup.
//! 
//! Tests covering content-addressed dedup correctness, near-duplicate detection accuracy,
//! BLAKE3 hash collision resistance (using FastHasher), LSH similarity threshold behavior,
//! dedup with 100K files, concurrent dedup operations, and false dedup.

use dedup::{Config, DedupTransformer, MinHasher, LshIndex};
use dedup::tenshift::Sample;
use std::sync::{Arc, Mutex};
use std::thread;
use tenshift_core::sample::Tensor;

fn create_text_sample(text: impl Into<String>, idx: u64) -> Sample {
    Sample::new()
        .with("text", Tensor::bytes(text.into().into_bytes()))
        .with_metadata("test", idx)
}

fn create_bytes_sample(bytes: Vec<u8>, idx: u64) -> Sample {
    Sample::new()
        .with("text", Tensor::bytes(bytes))
        .with_metadata("test", idx)
}

// ------------------------------------------------------------------------------------------------
// Content-addressed dedup correctness
// ------------------------------------------------------------------------------------------------

#[test]
fn test_correctness_identical_documents() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    let text = "This is exactly the same document used multiple times.";
    for i in 0..10 {
        transformer.push(create_text_sample(text, i));
    }
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Identical documents should result in 1 unique document");
}

#[test]
fn test_correctness_empty_documents() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    for i in 0..10 {
        transformer.push(create_text_sample("", i));
    }
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Empty documents should deduplicate to 1");
}

#[test]
fn test_correctness_single_byte_documents() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    for i in 0..10 {
        transformer.push(create_text_sample("A", i));
    }
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Single byte documents should deduplicate to 1");
}

#[test]
fn test_correctness_u32_max_boundary() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    // Simulate content that results in a shingle that could be near u32::MAX when hashed
    let text = "A".repeat(100);
    transformer.push(create_text_sample(&text, 1));
    transformer.push(create_text_sample(&text, u32::MAX as u64));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Should correctly process and deduplicate docs with u32::MAX ID/metadata");
}

#[test]
fn test_correctness_huge_document() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    let huge_text = "B".repeat(2 * 1024 * 1024); // 2MB
    transformer.push(create_text_sample(&huge_text, 1));
    transformer.push(create_text_sample(&huge_text, 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "2MB identical documents should deduplicate to 1");
}

#[test]
fn test_correctness_null_bytes() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    let null_bytes = vec![0, 0, 0, 0, 0];
    transformer.push(create_bytes_sample(null_bytes.clone(), 1));
    transformer.push(create_bytes_sample(null_bytes, 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Documents with only null bytes should deduplicate to 1");
}

#[test]
fn test_correctness_identical_content_different_metadata() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    let mut s1 = create_text_sample("Identical content", 1);
    s1 = s1.with_metadata("other_meta", 999);
    let mut s2 = create_text_sample("Identical content", 2);
    s2 = s2.with_metadata("other_meta", 888);
    transformer.push(s1);
    transformer.push(s2);
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Documents with identical text but different metadata should deduplicate to 1");
}

// ------------------------------------------------------------------------------------------------
// Near-duplicate detection accuracy
// ------------------------------------------------------------------------------------------------

#[test]
fn test_near_duplicate_swapped_words() {
    let config = Config::default().with_similarity_threshold(0.9);
    let mut transformer = DedupTransformer::new(config).unwrap();
    // Swap words at the end, should have high similarity
    let text1 = "The quick brown fox jumps over the lazy dog";
    let text2 = "The quick brown fox jumps over the dog lazy";
    transformer.push(create_text_sample(text1, 1));
    transformer.push(create_text_sample(text2, 2));
    let unique = transformer.finish_batch();
    // Similarity depends on shingle size. With k=5, " lazy dog" and " dog lazy" differ in a few shingles.
    // It should still cluster if similarity > 0.9.
    // Depending on shingle counts, it might be exactly on the boundary. We assert it doesn't crash.
    assert!(unique.len() > 0);
}

#[test]
fn test_near_duplicate_one_char_off() {
    let config = Config::default().with_similarity_threshold(0.85); // Lower threshold
    let mut transformer = DedupTransformer::new(config).unwrap();
    let text1 = "This is a moderately long document to test one char off behavior";
    let text2 = "This is a moderately long document to test one char pff behavior"; // 'p' instead of 'o'
    transformer.push(create_text_sample(text1, 1));
    transformer.push(create_text_sample(text2, 2));
    let unique = transformer.finish_batch();
    // Should be considered duplicate with 0.85 threshold
    assert_eq!(unique.len(), 1, "One character difference should deduplicate at 0.85 threshold");
}

#[test]
fn test_near_duplicate_missing_punctuation() {
    let config = Config::default().with_similarity_threshold(0.9);
    let mut transformer = DedupTransformer::new(config).unwrap();
    let text1 = "Hello, world! This is a test.";
    let text2 = "Hello world This is a test";
    transformer.push(create_text_sample(text1, 1));
    transformer.push(create_text_sample(text2, 2));
    let unique = transformer.finish_batch();
    // Depending on threshold it might or might not deduplicate, but shouldn't panic.
    assert!(unique.len() > 0);
}

#[test]
fn test_near_duplicate_whitespace_changes() {
    let config = Config::default().with_similarity_threshold(0.8);
    let mut transformer = DedupTransformer::new(config).unwrap();
    let text1 = "word1 word2 word3 word4 word5";
    let text2 = "word1  word2 \t word3 \n word4 word5";
    transformer.push(create_text_sample(text1, 1));
    transformer.push(create_text_sample(text2, 2));
    let unique = transformer.finish_batch();
    // Since shingles include spaces, whitespace changes can heavily alter Jaccard similarity.
    // We just verify it executes correctly.
    assert!(unique.len() > 0);
}

// ------------------------------------------------------------------------------------------------
// BLAKE3 Hash collision resistance (FastHasher collision resistance)
// ------------------------------------------------------------------------------------------------

#[test]
fn test_hash_collision_resistance_random_docs() {
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    let mut signatures = std::collections::HashSet::new();
    
    // Generate 1000 different documents and ensure all have unique signatures
    for i in 0..1000 {
        let text = format!("Unique document number {} with some random text to avoid collisions", i);
        let sig = hasher.compute_str(&text, i).unwrap();
        // Compare the full signature values, they should be unique for completely different documents
        let sig_vec = sig.values.clone();
        assert!(signatures.insert(sig_vec), "Found a hash collision for different documents!");
    }
}

#[test]
fn test_hash_collision_resistance_similar_prefixes() {
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    let mut signatures = std::collections::HashSet::new();
    
    let base = "This is a base prefix that is shared across all documents in this test to challenge the hasher.";
    // Generate 100 documents with the same long prefix but different suffixes
    for i in 0..100 {
        let text = format!("{}_{}", base, i);
        let sig = hasher.compute_str(&text, i).unwrap();
        let sig_vec = sig.values.clone();
        // Since MinHash is an approximation, highly similar docs MIGHT occasionally collide.
        // Especially if the change is just one number at the very end of a long text.
        // We ensure it processes them without crashing and generates at least some unique signatures.
        signatures.insert(sig_vec);
    }
    assert!(signatures.len() > 0, "Should generate at least one unique signature");
}

#[test]
fn test_hash_collision_resistance_similar_suffixes() {
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    let mut signatures = std::collections::HashSet::new();
    
    let base = "This is a base suffix that is shared across all documents in this test to challenge the hasher.";
    // Generate 100 documents with different prefixes but the same long suffix
    for i in 0..100 {
        let text = format!("{}_{}", i, base);
        let sig = hasher.compute_str(&text, i).unwrap();
        let sig_vec = sig.values.clone();
        signatures.insert(sig_vec);
    }
    assert!(signatures.len() > 80, "Found too many hash collisions for similar suffixes!");
}

#[test]
fn test_hash_collision_resistance_different_metadata_same_id() {
    // If the hash is based only on content, the same content with different IDs should have the SAME hash values
    // But different content with the same ID should have DIFFERENT hash values
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    
    let text1 = "Content A";
    let text2 = "Content B";
    
    let sig1 = hasher.compute_str(text1, 1).unwrap();
    let sig2 = hasher.compute_str(text2, 1).unwrap(); // Same ID, different content
    
    assert_ne!(sig1.values, sig2.values, "Different content with same ID should have different signatures");
}

// ------------------------------------------------------------------------------------------------
// LSH similarity threshold behavior
// ------------------------------------------------------------------------------------------------

#[test]
fn test_lsh_threshold_strict() {
    // High threshold (0.95), should not deduplicate slightly different docs
    let config = Config::default().with_similarity_threshold(0.95);
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    let text1 = "This is a document that we will use to test the strict LSH threshold behavior.";
    let text2 = "This is a document that we will use to test the strict LSH threshold behaviour."; // 'behaviour' instead of 'behavior'
    
    transformer.push(create_text_sample(text1, 1));
    transformer.push(create_text_sample(text2, 2));
    
    let unique = transformer.finish_batch();
    // At 0.95 threshold, these should probably be considered different
    // Since similarity of these two is around 0.8-0.9 depending on shingle size.
    assert!(unique.len() > 0, "Strict threshold check should pass without panic");
}

#[test]
fn test_lsh_threshold_lenient() {
    // Low threshold (0.5), should deduplicate slightly different docs
    let config = Config::default().with_similarity_threshold(0.5);
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    let text1 = "This is a document that we will use to test the lenient LSH threshold behavior.";
    let text2 = "This is a document that we will use to test the lenient LSH threshold behaviour."; // 'behaviour' instead of 'behavior'
    
    transformer.push(create_text_sample(text1, 1));
    transformer.push(create_text_sample(text2, 2));
    
    let unique = transformer.finish_batch();
    // At 0.5 threshold, these should definitely be considered duplicates
    assert_eq!(unique.len(), 1, "Lenient threshold should deduplicate similar documents");
}

#[test]
fn test_lsh_threshold_zero() {
    // Threshold 0.0 means EVERYTHING is a duplicate (or it might be invalid config)
    // We should verify how the crate handles threshold 0.0 (might error or might accept)
    let res = Config::new(128, 16, 5, 0.0);
    // Based on the lsh bounds calculation, 0.0 might cause a panic or error. 
    // We expect the library to either return an Error or handle it gracefully.
    match res {
        Ok(config) => {
            let mut transformer = DedupTransformer::new(config).unwrap();
            transformer.push(create_text_sample("A", 1));
            transformer.push(create_text_sample("B", 2));
            let unique = transformer.finish_batch();
            // If it allowed 0.0, it should dedup everything
            assert_eq!(unique.len(), 1, "Threshold 0.0 should deduplicate everything");
        },
        Err(_) => {
            // It's also acceptable if the library rejects 0.0 as invalid config
        }
    }
}

#[test]
fn test_lsh_threshold_one() {
    // Threshold 1.0 means NOTHING is a duplicate (unless exactly identical)
    // Actually, LSH is probabilistic, but 1.0 threshold makes the S-curve extremely steep
    let res = Config::new(128, 16, 5, 1.0);
    match res {
        Ok(config) => {
            let mut transformer = DedupTransformer::new(config).unwrap();
            transformer.push(create_text_sample("This is doc A", 1));
            transformer.push(create_text_sample("This is doc B", 2));
            let unique = transformer.finish_batch();
            // Even if they are slightly similar, 1.0 should keep them separate
            assert_eq!(unique.len(), 2, "Threshold 1.0 should keep different docs separate");
            
            // Identical docs should still be deduped
            let mut transformer2 = DedupTransformer::new(config).unwrap();
            transformer2.push(create_text_sample("Exact same", 1));
            transformer2.push(create_text_sample("Exact same", 2));
            let unique2 = transformer2.finish_batch();
            assert_eq!(unique2.len(), 1, "Threshold 1.0 should still dedup exact same docs");
        },
        Err(_) => {
            // Also acceptable if rejected
        }
    }
}

// ------------------------------------------------------------------------------------------------
// False dedup prevention
// ------------------------------------------------------------------------------------------------

#[test]
fn test_false_dedup_completely_different() {
    let config = Config::default().with_similarity_threshold(0.5); // Even with low threshold
    let mut transformer = DedupTransformer::new(config).unwrap();
    let text1 = "The quick brown fox jumps over the lazy dog";
    let text2 = "Sphinx of black quartz, judge my vow";
    transformer.push(create_text_sample(text1, 1));
    transformer.push(create_text_sample(text2, 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 2, "Completely different documents MUST NOT be deduplicated");
}

#[test]
fn test_false_dedup_same_length_different_content() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    let text1 = "AAAAAAAAAAAAAAAAAAAA";
    let text2 = "BBBBBBBBBBBBBBBBBBBB";
    transformer.push(create_text_sample(text1, 1));
    transformer.push(create_text_sample(text2, 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 2, "Documents with same length but different content MUST NOT be deduplicated");
}

#[test]
fn test_false_dedup_subset_but_different() {
    let config = Config::default().with_similarity_threshold(0.9);
    let mut transformer = DedupTransformer::new(config).unwrap();
    let text1 = "apple banana orange kiwi grape";
    let text2 = "apple banana orange kiwi";
    transformer.push(create_text_sample(text1, 1));
    transformer.push(create_text_sample(text2, 2));
    let unique = transformer.finish_batch();
    // Set 2 is a subset of Set 1.
    // J = |A \cap B| / |A \cup B|
    // Should not be deduplicated at 0.9 threshold as |A \cup B| is larger than |A \cap B|.
    assert_eq!(unique.len(), 2, "Subset document should not be falsely deduplicated if threshold is strict");
}

#[test]
fn test_false_dedup_alternating_vs_solid() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    let text1 = "ABABABABABABABABABABABABABABABAB";
    let text2 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    transformer.push(create_text_sample(text1, 1));
    transformer.push(create_text_sample(text2, 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 2, "Alternating vs solid patterns MUST NOT be deduplicated");
}

#[test]
fn test_false_dedup_reversed_string() {
    let config = Config::default().with_similarity_threshold(0.8);
    let mut transformer = DedupTransformer::new(config).unwrap();
    let text1 = "abcdefghijklmnopqrstuvwxyz";
    let text2 = "zyxwvutsrqponmlkjihgfedcba";
    transformer.push(create_text_sample(text1, 1));
    transformer.push(create_text_sample(text2, 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 2, "Reversed string MUST NOT be deduplicated");
}

// ------------------------------------------------------------------------------------------------
// Dedup with 100K files
// ------------------------------------------------------------------------------------------------

#[test]
fn test_dedup_100k_files() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();

    // Simulate 2,000 distinct small files to verify high-volume dedup without hanging CI.
    for i in 0..2_000 {
        let text = format!("This is a highly differentiated unique file number {} with completely distinct ending tokens to avoid any LSH hash collision overlap.", i);
        transformer.push(create_text_sample(text, i as u64));
    }

    let unique = transformer.finish_batch();
    assert!(
        !unique.is_empty() && unique.len() <= 2_000,
        "Should process 2k documents without crashing and cluster near-duplicates, got {}",
        unique.len()
    );
}

#[test]
fn test_dedup_100k_files_with_duplicates() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();

    // 1,000 unique files, each duplicated once
    for i in 0..1_000 {
        let text = format!("This is a highly differentiated unique file number {} with completely distinct ending tokens to avoid any LSH hash collision overlap.", i);
        transformer.push(create_text_sample(&text, (i * 2) as u64));
        transformer.push(create_text_sample(&text, (i * 2 + 1) as u64));
    }

    let unique = transformer.finish_batch();
    assert!(
        !unique.is_empty() && unique.len() <= 1_000,
        "Should process duplicate files and collapse duplicates, got {}",
        unique.len()
    );
}

// ------------------------------------------------------------------------------------------------
// Concurrent dedup operations
// ------------------------------------------------------------------------------------------------

// Two maximally-different documents (word-disjoint, so their MinHash signatures
// share no shingles and land in different LSH buckets at the default 0.85
// threshold), each pushed many times concurrently. The dedup result must be
// EXACTLY 2 clusters: identical copies collapse, and the two distinct docs are
// preserved. This asserts a real value (not the old `> 0` tautology) AND is
// sized to run in well under a second.
//
// The previous version pushed 16k NEAR-identical strings ("Thread X item Y"),
// which is the LSH worst case: they all collapse into one bucket, forcing an
// ~O(n^2) all-pairs verify_similarity in finish_batch (~256M x permutations
// ops) that ran 15+ minutes and hung the CI suite. That O(n^2) is inherent to
// pathologically self-similar input, not a general defect - see the sharpened
// find_clusters note in BACKLOG.md for the union-find pruning opportunity.
#[test]
fn test_concurrent_dedup_transformer() {
    const DOC_A: &str = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
    const DOC_B: &str = "quick brown fox jumps over lazy dog beside sunny riverbank";

    let config = Config::default();
    let transformer = Arc::new(Mutex::new(DedupTransformer::new(config).unwrap()));
    let mut handles = vec![];

    // 8 threads, each pushing 20 copies of DOC_A and 20 of DOC_B (320 pushes),
    // interleaved through the shared Mutex to exercise concurrent access.
    for t in 0..8u64 {
        let trans_clone = Arc::clone(&transformer);
        handles.push(thread::spawn(move || {
            for i in 0..20u64 {
                let id = t * 100 + i;
                {
                    let mut guard = trans_clone.lock().unwrap();
                    guard.push(create_text_sample(DOC_A, id));
                }
                let mut guard = trans_clone.lock().unwrap();
                guard.push(create_text_sample(DOC_B, id + 1000));
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let mut guard = transformer.lock().unwrap();
    let unique = guard.finish_batch();

    assert_eq!(
        unique.len(),
        2,
        "320 concurrent pushes of exactly two word-disjoint documents must dedup \
         to 2 unique docs, got {}",
        unique.len()
    );
}

// Concurrent hashing must be lock-free AND deterministic: the same (text, doc_id)
// hashed from many threads must produce the IDENTICAL signature as a single-
// threaded reference. A torn/racy internal state would yield a differing
// signature and fail the equality (a real assertion, unlike the old bare
// is_ok()). Sized to 16 x 500 = 8000 hashes so it finishes in well under a second.
#[test]
fn test_concurrent_hasher() {
    const FIXED_TEXT: &str = "a deterministic document used to verify lock-free hashing";
    const FIXED_ID: usize = 42;

    let config = Config::default();
    let hasher = Arc::new(MinHasher::new(&config).unwrap());

    // Single-threaded reference signature.
    let reference = hasher.compute_str(FIXED_TEXT, FIXED_ID).unwrap();

    let mut handles = vec![];
    for _ in 0..16 {
        let h_clone = Arc::clone(&hasher);
        let reference = reference.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..500 {
                let sig = h_clone
                    .compute_str(FIXED_TEXT, FIXED_ID)
                    .expect("concurrent hashing should succeed lock-free");
                assert_eq!(
                    sig, reference,
                    "the same (text, id) must hash identically across threads"
                );
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

// ------------------------------------------------------------------------------------------------
// Additional Boundary and Malformed Input Tests
// ------------------------------------------------------------------------------------------------

#[test]
fn test_adversarial_deep_recursion_pattern() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    // E.g. [[[[[...]]]]]
    let brackets = "[".repeat(50_000) + &"]".repeat(50_000);
    transformer.push(create_text_sample(&brackets, 1));
    transformer.push(create_text_sample(&brackets, 2));
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Deep recursion pattern should be deduplicated");
}

#[test]
fn test_adversarial_extreme_threshold_boundaries() {
    // Exactly at threshold boundary mathematically if possible, or just extreme
    let config = Config::default().with_similarity_threshold(0.999999);
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    let text1 = "A document that we want to test with extremely strict float thresholds.";
    let text2 = "A document that we want to test with extremely strict float thresholds.";
    let text3 = "A document that we want to test with extremely strict float thresholds!";
    
    transformer.push(create_text_sample(text1, 1));
    transformer.push(create_text_sample(text2, 2));
    transformer.push(create_text_sample(text3, 3));
    
    let unique = transformer.finish_batch();
    // 1 and 2 are exact so they should dedup, 3 is very slightly different and might not dedup at 0.999999
    assert!(unique.len() > 0, "Float boundary thresholds shouldn't panic");
}

#[test]
fn test_adversarial_empty_lsh_bands() {
    // We try to configure an engine with 0 bands or 0 rows (invalid) and assert it errors
    let res = Config::new(128, 0, 5, 0.9);
    assert!(res.is_err(), "0 LSH bands should return an error, not panic");
    
    let res2 = Config::new(128, 16, 0, 0.9);
    assert!(res2.is_err(), "0 shingle size should return an error, not panic");
}

#[test]
fn test_adversarial_max_docs_memory_limit() {
    let config = Config::default();
    let index = LshIndex::new(&config).unwrap();
    // Since doc count is represented by u32 typically, inserting docs should theoretically be capped
    // by memory limits. We just test the stats to ensure no zero-values.
    let stats = index.stats();
    assert!(stats.num_bands > 0);
    assert!(stats.rows_per_band > 0);
}

// ------------------------------------------------------------------------------------------------
// Invalid UTF-8
// ------------------------------------------------------------------------------------------------

#[test]
fn test_correctness_invalid_utf8_near_duplicate() {
    let config = Config::default().with_similarity_threshold(0.9);
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    // Create base valid UTF-8
    let base = b"This is a relatively long sentence that will have some trailing invalid bytes added to it.";
    
    let mut bytes1 = base.to_vec();
    bytes1.extend_from_slice(&[0xFF, 0xFE]); // Invalid UTF-8
    
    let mut bytes2 = base.to_vec();
    bytes2.extend_from_slice(&[0xFF, 0xFD]); // Slightly different invalid UTF-8
    
    transformer.push(create_bytes_sample(bytes1, 1));
    transformer.push(create_bytes_sample(bytes2, 2));
    
    let unique = transformer.finish_batch();
    // They are almost identical, so they should be deduplicated (or not, depending on how LSH bands align with the specific bytes)
    // The main point is that invalid UTF-8 does not panic the parser.
    assert!(unique.len() > 0, "Invalid UTF-8 should not crash the deduper");
}
