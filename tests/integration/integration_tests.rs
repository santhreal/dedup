//! Integration tests for the dedup crate.
//!
//! These tests verify end-to-end deduplication workflows.

use dedup::{Config, DedupTransformer, LshIndex, MinHasher};
use dedup::tenshift::Sample;
use tenshift_core::sample::Tensor;

fn create_text_sample(text: impl Into<String>, idx: u64) -> Sample {
    let text = text.into();
    Sample::new()
        .with("text", Tensor::bytes(text.into_bytes()))
        .with_metadata("test", idx)
}

#[test]
fn test_end_to_end_deduplication() {
    let config = Config::default()
        .with_similarity_threshold(0.8)
        .with_num_bands(8);
    
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    // Create documents with known duplicates
    let docs = vec![
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog", // Exact duplicate
        "The quick brown fox jumps over the lazy cat", // Near duplicate
        "Completely different content about machine learning",
        "Another unique document about rust programming",
    ];
    
    for (i, doc) in docs.iter().enumerate() {
        transformer.push(create_text_sample(*doc, i as u64));
    }
    
    let unique = transformer.finish_batch();
    let clusters = transformer.clusters();
    
    // Should have 4 unique: 2 near-duplicates merged, 3 others unique
    assert_eq!(unique.len(), 4, "Expected 4 unique documents");
    
    // Should have 1 cluster with 2 documents
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 2);
}

#[test]
fn test_exact_duplicate_detection() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    // Insert 10 identical documents
    for i in 0..10 {
        transformer.push(create_text_sample("identical content", i));
    }
    
    let unique = transformer.finish_batch();
    
    // Only 1 should remain
    assert_eq!(unique.len(), 1);
    assert_eq!(transformer.duplicate_count(), 9);
}

#[test]
fn test_similarity_threshold_effect() {
    let high_threshold = Config::default()
        .with_similarity_threshold(0.95)
        .with_num_bands(20);
    
    let low_threshold = Config::default()
        .with_similarity_threshold(0.5)
        .with_num_bands(8);
    
    let docs = vec![
        "hello world test content here",
        "hello world test content there", // Similar but not identical
    ];
    
    // With high threshold, both should be unique
    let mut transformer_high = DedupTransformer::new(high_threshold).unwrap();
    for (i, doc) in docs.iter().enumerate() {
        transformer_high.push(create_text_sample(*doc, i as u64));
    }
    let unique_high = transformer_high.finish_batch();
    
    // With low threshold, they might be considered duplicates
    let mut transformer_low = DedupTransformer::new(low_threshold).unwrap();
    for (i, doc) in docs.iter().enumerate() {
        transformer_low.push(create_text_sample(*doc, i as u64));
    }
    let unique_low = transformer_low.finish_batch();
    
    // High threshold should keep more documents
    assert!(unique_high.len() >= unique_low.len());
}

#[test]
fn test_empty_dataset() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    let unique = transformer.finish_batch();
    assert!(unique.is_empty());
}

#[test]
fn test_single_document() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    transformer.push(create_text_sample("only document", 0));
    
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 1);
    assert_eq!(transformer.duplicate_count(), 0);
}

#[test]
fn test_all_unique_documents() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    // Create 50 completely different documents
    for i in 0..50 {
        let doc = format!("unique document number {} with distinct content", i);
        transformer.push(create_text_sample(doc, i));
    }
    
    let unique = transformer.finish_batch();
    
    // All should remain unique
    assert_eq!(unique.len(), 50);
    assert_eq!(transformer.duplicate_count(), 0);
}

#[test]
fn test_large_scale_deduplication() {
    let config = Config::default()
        .with_signature_size(128)
        .with_num_bands(16);
    
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    // Create dataset with known duplicate ratio
    let total_docs = 1000;
    let duplicate_ratio = 0.3; // 30% duplicates
    
    let base_content = "base document content for deduplication testing";
    
    for i in 0..total_docs {
        let content = if (i as f64 / total_docs as f64) < duplicate_ratio {
            base_content.to_string()
        } else {
            format!("unique content for document {}", i)
        };
        transformer.push(create_text_sample(content, i as u64));
    }
    
    let unique = transformer.finish_batch();
    let stats = transformer.stats();
    
    // Verify deduplication worked
    assert!(unique.len() < total_docs);
    assert!(stats.duplicate_count > 0);
    assert_eq!(stats.doc_count, total_docs);
}

#[test]
fn test_unicode_text_deduplication() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    let docs = vec![
        "你好世界，这是一个测试文档",
        "你好世界，这是一个测试文档", // Exact duplicate
        "こんにちは世界、これはテストです",
        "🎉 Emoji test document 🚀",
        "🎉 Emoji test document 🚀", // Emoji duplicate
    ];
    
    for (i, doc) in docs.iter().enumerate() {
        transformer.push(create_text_sample(*doc, i as u64));
    }
    
    let unique = transformer.finish_batch();
    
    // Should have 3 unique: Chinese, Japanese, Emoji
    assert_eq!(unique.len(), 3);
}

#[test]
fn test_minhash_consistency() {
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    
    let doc = "consistent document for testing";
    
    // Compute signature multiple times
    let sig1 = hasher.compute_str(doc, 0).unwrap();
    let sig2 = hasher.compute_str(doc, 1).unwrap();
    let sig3 = hasher.compute_str(doc, 2).unwrap();
    
    // All should be identical
    assert_eq!(sig1.values, sig2.values);
    assert_eq!(sig2.values, sig3.values);
}

#[test]
fn test_lsh_candidate_generation() {
    let config = Config::default();
    let mut index = LshIndex::new(&config).unwrap();
    
    let hasher = MinHasher::new(&config).unwrap();
    
    // Create similar documents
    let doc1 = "the quick brown fox jumps over the lazy dog";
    let doc2 = "the quick brown fox jumps over the lazy cat";
    
    let sig1 = hasher.compute_str(doc1, 0).unwrap();
    let sig2 = hasher.compute_str(doc2, 1).unwrap();
    
    // Insert first document
    let candidates1 = index.insert(sig1).unwrap();
    assert!(candidates1.is_empty());
    
    // Query with second document
    let candidates2 = index.insert(sig2).unwrap();
    
    // Should find first document as candidate
    assert!(candidates2.contains(&0));
}

#[test]
fn test_config_variations() {
    // Test various configuration combinations
    let configs = vec![
        Config::new(64, 8, 3, 0.9).unwrap(),
        Config::new(128, 16, 5, 0.85).unwrap(),
        Config::new(256, 32, 7, 0.8).unwrap(),
    ];
    
    for config in configs {
        let mut transformer = DedupTransformer::new(config).unwrap();
        
        transformer.push(create_text_sample("test document one", 0));
        transformer.push(create_text_sample("test document two", 1));
        
        let unique = transformer.finish_batch();
        assert_eq!(unique.len(), 2);
    }
}

#[test]
fn test_cluster_info() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    // Create a cluster
    let cluster_docs = vec![
        "document in cluster A",
        "document in cluster A",
        "document in cluster A",
    ];
    
    for (i, doc) in cluster_docs.iter().enumerate() {
        transformer.push(create_text_sample(*doc, i as u64));
    }
    
    transformer.finish_batch();
    let clusters = transformer.clusters();
    
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 3);
    assert_eq!(clusters[0].representative, 0);
}

#[test]
fn test_streaming_mode() {
    let config = Config::default();
    let transformer = DedupTransformer::new(config)
        .unwrap()
        .with_streaming(true);
    
    // In streaming mode, non-duplicates should be output immediately
    assert!(transformer.streaming);
}



#[test]
fn test_memory_estimation() {
    let config = Config::default();
    let transformer = DedupTransformer::new(config).unwrap();
    
    let stats = transformer.stats();
    
    // Should have valid statistics even with no docs
    assert_eq!(stats.doc_count, 0);
}

#[test]
fn test_shingle_size_variations() {
    let configs = vec![
        Config::default().with_shingle_size(3),
        Config::default().with_shingle_size(5),
        Config::default().with_shingle_size(7),
    ];
    
    for config in configs {
        let mut transformer = DedupTransformer::new(config).unwrap();
        transformer.push(create_text_sample("test document content", 0));
        let unique = transformer.finish_batch();
        assert_eq!(unique.len(), 1);
    }
}

#[test]
fn test_different_text_fields() {
    let config = Config::default();
    
    // Default field is "text"
    let _transformer1 = DedupTransformer::new(config).unwrap();
    
    // Custom field
    let _transformer2 = DedupTransformer::new(config.clone())
        .unwrap()
        .with_text_field("content");
}

#[test]
fn test_edge_cases() {
    let config = Config::default();
    let mut transformer = DedupTransformer::new(config).unwrap();
    
    // Very short document
    transformer.push(create_text_sample("hi", 0));
    
    // Very long document
    let long_doc = "word ".repeat(10000);
    transformer.push(create_text_sample(&long_doc, 1));
    
    // Document with special characters
    transformer.push(create_text_sample("!@#$%^&*()_+{}|:<>?~`-=[]\\;',./", 2));
    
    let unique = transformer.finish_batch();
    assert_eq!(unique.len(), 3);
}

#[test]
fn test_reproducibility() {
    let config = Config::default().with_seed(12345);
    
    let mut transformer1 = DedupTransformer::new(config).unwrap();
    let mut transformer2 = DedupTransformer::new(config).unwrap();
    
    let docs = vec![
        "document one",
        "document two",
        "document one", // Duplicate
    ];
    
    for (i, doc) in docs.iter().enumerate() {
        transformer1.push(create_text_sample(*doc, i as u64));
        transformer2.push(create_text_sample(*doc, i as u64));
    }
    
    let unique1 = transformer1.finish_batch();
    let unique2 = transformer2.finish_batch();
    
    assert_eq!(unique1.len(), unique2.len());
    assert_eq!(transformer1.duplicate_count(), transformer2.duplicate_count());
}
