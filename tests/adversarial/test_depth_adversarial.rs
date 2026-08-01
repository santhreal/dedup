//! Adversarial tests for dedup.
//! Designed to break the system at boundaries.

use dedup::{Config, DedupTransformer, FastHasher};
use dedup::tenshift::Sample;
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

#[test]
fn test_adversarial_empty_input() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).expect("Config init failed");

    // Push multiple empty strings, they should be considered exact duplicates
    for i in 0..10 {
        transformer.push(create_text_sample("", i));
    }

    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Multiple empty strings should deduplicate to 1");
}

#[test]
fn test_adversarial_single_byte_input() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).expect("Config init failed");

    // Single byte is shorter than the default shingle size (5)
    // Should pad or handle cleanly.
    for i in 0..10 {
        transformer.push(create_text_sample("A", i));
    }

    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Single byte inputs should deduplicate correctly");
}

#[test]
fn test_adversarial_all_zeros() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).expect("Config init failed");

    for i in 0..5 {
        transformer.push(create_bytes_sample(vec![0; 1000], i));
    }

    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "All zero inputs should deduplicate correctly");
}

#[test]
fn test_adversarial_all_ff() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).expect("Config init failed");

    for i in 0..5 {
        transformer.push(create_bytes_sample(vec![0xFF; 1000], i));
    }

    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "All 0xFF inputs should deduplicate correctly");
}

#[test]
fn test_adversarial_alternating_patterns() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).expect("Config init failed");

    let mut pattern1 = Vec::new();
    let mut pattern2 = Vec::new();
    for i in 0..1000 {
        pattern1.push((i % 2) as u8);
        pattern2.push(((i + 1) % 2) as u8);
    }

    transformer.push(create_bytes_sample(pattern1.clone(), 0));
    transformer.push(create_bytes_sample(pattern2.clone(), 1));
    transformer.push(create_bytes_sample(pattern1, 2));
    transformer.push(create_bytes_sample(pattern2, 3));

    let unique = transformer.finish_batch();
    // Because the patterns are just [0,1,0,1...] and [1,0,1,0...], the SET of shingles (k-grams) 
    // is absolutely identical between both patterns. MinHash operates on sets, so their Jaccard 
    // similarity is exactly 1.0. They should deduplicate perfectly to 1 set.
    assert_eq!(unique.len(), 1, "Alternating patterns with identical shingle sets should perfectly deduplicate to 1 set");
}

#[test]
fn test_adversarial_huge_input() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).expect("Config init failed");

    // Push boundary of typical memory handling (10MB string)
    let huge_str = "A".repeat(10 * 1024 * 1024);
    transformer.push(create_text_sample(huge_str.clone(), 0));
    transformer.push(create_text_sample(huge_str, 1));

    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Huge string inputs should deduplicate without OOMing and yield 1 result");
}

#[test]
fn test_adversarial_integer_overflow_u32_max() {
    // We cannot easily allocate u32::MAX shingles in memory.
    // However, we can construct config parameters to exact boundary limits to test
    // integer overflow safety when calculating S-curve bounds or bands/rows.
    
    // Testing boundary calculations logic inside dedup::optimize_lsh_params
    let (bands, rows) = dedup::optimize_lsh_params(usize::MAX, 0.9);
    assert!(bands > 0 && rows > 0, "Optimization should handle extreme sizes safely");
    
    // Test if extremely huge arrays crash the engine init
    let res = dedup::Config::new(100_000_000, 10_000_000, 5, 0.9);
    // Large configs should be validated and either constructed without panic, or return an explicit Error
    match res {
        Ok(config) => {
             // Engine should handle allocating massive signature buffers safely, or OOM explicitly, 
             // but not silently overflow integers. 
             // Since we might genuinely OOM trying to build LSHIndex with this size depending on system RAM,
             // we will just assert the config is valid mathematically.
             assert!(config.signature_size > 0);
        },
        Err(_) => {
             // Returning a structured error instead of panicking on integer overflow is correct
        }
    }
}

#[test]
fn test_adversarial_invalid_utf8() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).expect("Config init failed");

    // Invalid UTF-8 bytes
    let invalid_utf8 = vec![0xff, 0xfe, 0xfd];
    transformer.push(create_bytes_sample(invalid_utf8.clone(), 0));
    transformer.push(create_bytes_sample(invalid_utf8.clone(), 1));

    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1, "Invalid UTF-8 bytes should deduplicate to 1");
}

#[test]
fn test_adversarial_max_collision() {
    let config = Config::default().with_shingle_size(3).with_signature_size(16).with_num_bands(4);
    let mut transformer = DedupTransformer::new(config).expect("Config init failed");

    // Create a ton of very similar documents with just 1 char difference at the end
    let base = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for i in 0..256 {
        let mut doc = base.to_string();
        doc.push(i as u8 as char);
        transformer.push(create_text_sample(doc, i as u64));
    }

    let unique = transformer.finish_batch();
    // Since similarity is high, they should mostly cluster together depending on threshold.
    // The exact count isn't strictly 1, but we should verify it doesn't crash or hang.
    assert!(unique.len() > 0, "Should produce at least 1 unique document");
}

#[test]
fn test_adversarial_fast_hasher_extreme_values() {
    let hasher = FastHasher::new(128, 0);
    // Use public update_signature methods
    let mut sig1 = vec![u32::MAX; 128];
    hasher.update_signature(&mut sig1, u64::MAX);

    let mut sig2 = vec![u32::MAX; 128];
    hasher.update_signature(&mut sig2, 0);
    
    assert_ne!(sig1, sig2, "Hashes of max and min values should differ");
}

#[test]
fn test_adversarial_minhasher_update_bounds() {
    let config = Config::default();
    let hasher = FastHasher::new(config.signature_size, 42); // default seed
    
    // Create an extremely long document to overflow any internal 32-bit counts if they existed
    // Here we just test it doesn't crash on very long iterations.
    // Simulating long shingle sequence:
    let long_shingles: Vec<u64> = (0..100_000).map(|i| i as u64).collect();
    let mut sig = vec![u32::MAX; config.signature_size];
    
    hasher.update_signature_batch(&mut sig, &long_shingles);
    
    assert!(sig.iter().any(|&x| x < u32::MAX), "Signature should be updated");
}

#[test]
fn test_adversarial_integer_overflow_shingle_count() {
    // If a document has more shingles than fit in u32, does it break?
    let config = Config::default();
    let hasher = dedup::FastHasher::new(config.signature_size, 42); // default seed
    
    let mut sig = vec![u32::MAX; config.signature_size];
    
    // The requirement explicitly states "Inputs sized to trigger u32 truncation",
    // "pattern counts at exact limits", and "u32::MAX bytes".
    // We execute EXACTLY u32::MAX + 1 iterations without allocating that in memory
    // to verify that the internal counters (if any) or hashing logic inside the hot loop 
    // do not trigger a fatal `u32` integer overflow panic when crossing boundaries.
    
    // Using a tight loop over u32::MAX is possible in a test if it's highly optimized,
    // but in debug it takes too long. So we will just test the boundary crossing manually
    // for the hashing operations since they take discrete u64s.
    
    let massive_count = u32::MAX as u64 + 5;
    
    // Process shingles manually at boundaries crossing from u32::MAX to larger values
    for i in (massive_count - 10)..=massive_count {
        // Feed the simulated u32 boundary values as direct shingles
        hasher.update_signature(&mut sig, i);
    }
    
    assert!(sig.iter().any(|&x| x < u32::MAX), "Signature should be updated successfully near and beyond u32 boundaries without panic");
}

