//! Basic deduplication example.
//!
//! Demonstrates how to use the dedup crate to find and remove duplicates.

use dedup::{Config, DedupTransformer};
use dedup::tenshift::Sample;
use tenshift_core::sample::Tensor;

fn main() {
    // Create configuration
    let config = Config::default()
        .with_similarity_threshold(0.85)
        .with_num_bands(16);

    println!("Deduplication Example");
    println!("=====================");
    println!("Signature size: {}", config.signature_size);
    println!("Number of bands: {}", config.num_bands);
    println!("Similarity threshold: {}", config.similarity_threshold);
    println!();

    // Create transformer
    let mut transformer = DedupTransformer::new(config).expect("Failed to create transformer");

    // Sample documents
    let documents = vec![
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",                    // Exact duplicate
        "The quick brown fox jumps over the lazy cat",                    // Near duplicate
        "Machine learning is transforming how we build software",
        "Machine learning is transforming how we build applications",     // Near duplicate
        "Rust provides memory safety without garbage collection",
        "Python is great for data science and machine learning",
        "The quick brown fox jumps over the lazy dog",                    // Another duplicate
    ];

    println!("Input documents: {}", documents.len());
    for (i, doc) in documents.iter().enumerate() {
        println!("  [{}] {}", i, &doc[..doc.len().min(50)]);
    }
    println!();

    // Process documents
    for (i, doc) in documents.iter().enumerate() {
        let sample = Sample::new()
            .with("text", Tensor::bytes(doc.as_bytes().to_vec()))
            .with_metadata("input", i as u64);
        
        transformer.push(sample);
    }

    // Get deduplicated results
    let unique_samples = transformer.finish_batch();
    let stats = transformer.stats();

    // Report results
    println!("Results");
    println!("=======");
    println!("Unique documents: {}", unique_samples.len());
    println!("Duplicate documents: {}", stats.duplicate_count);
    println!();

    println!("LSH Statistics:");
    println!("  Total buckets: {}", stats.total_buckets);
    println!("  Average bucket size: {:.2}", stats.avg_bucket_size);
    println!("  Estimated recall at threshold: {:.2}%", stats.estimated_recall() * 100.0);
}
