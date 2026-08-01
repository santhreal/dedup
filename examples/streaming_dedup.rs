//! Streaming deduplication example with tenshift integration.
//!
//! Demonstrates how to use dedup as part of a data processing pipeline.

use dedup::{Config, StatefulDedupTransform};
use tenshift_core::sample::{Sample, Tensor};
use tenshift_core::transform::StatefulTransform;

fn main() {
    println!("Streaming Deduplication Example");
    println!("================================\n");

    // Create configuration optimized for streaming
    let config = Config::default()
        .with_similarity_threshold(0.9)
        .with_signature_size(128)
        .with_num_bands(16)
        .with_shingle_size(5);

    let mut transform = StatefulDedupTransform::new(config)
        .expect("Failed to create transform")
        .with_text_field("content");

    // Simulate incoming documents
    let incoming_docs = vec![
        ("doc_001", "Introduction to Machine Learning"),
        ("doc_002", "Introduction to Machine Learning"),  // Duplicate
        ("doc_003", "Advanced Topics in Deep Learning"),
        ("doc_004", "Introduction to Machine Learning"),  // Another duplicate
        ("doc_005", "Rust Programming Best Practices"),
        ("doc_006", "Rust Best Practices for Production"), // Near duplicate
        ("doc_007", "Data Pipeline Architecture Patterns"),
    ];

    let num_input = incoming_docs.len();
    println!("Processing {} documents...\n", num_input);

    // Push all documents through the transform
    for (id, content) in &incoming_docs {
        let sample = Sample::new()
            .with("content", Tensor::bytes(content.as_bytes().to_vec()))
            .with("doc_id", Tensor::bytes(id.as_bytes().to_vec()))
            .with_metadata("source", 0);

        let output = transform.push(sample);
        
        // In non-streaming mode, output is empty until finish
        if !output.is_empty() {
            println!("Immediate output: {} samples", output.len());
        }
    }

    // Get clusters first
    let clusters = transform.clusters().to_vec();
    // Then get stats
    let stats = transform.stats();
    // Finish and get deduplicated results
    let deduplicated = transform.finish();

    println!("Deduplication Results");
    println!("=====================");
    println!("Input documents: {}", num_input);
    println!("Output documents: {}", deduplicated.len());
    println!("Duplicates removed: {}", num_input - deduplicated.len());
    println!();

    println!("Duplicate Clusters:");
    for cluster in &clusters {
        println!("  Cluster {}: {:?}", cluster.id, cluster.indices);
    }
    println!();

    println!("Remaining unique documents:");
    for sample in &deduplicated {
        if let Some(tensor) = sample.get("doc_id") {
            let doc_id = String::from_utf8_lossy(tensor.as_bytes());
            println!("  - {}", doc_id);
        }
    }

    println!();
    println!("Performance Statistics:");
    println!("  Documents processed: {}", stats.doc_count);
    println!("  Duplicate clusters: {}", stats.cluster_count);
    println!("  Estimated recall: {:.1}%", stats.estimated_recall() * 100.0);
}
