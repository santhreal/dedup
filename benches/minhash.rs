//! Benchmarks for MinHash and LSH operations.
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use dedup::{Config, MinHasher, LshIndex};
use dedup::shingle::normalize_text;

fn generate_documents(count: usize, length: usize) -> Vec<String> {
    let base_words = vec![
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "data", "science", "rust", "programming",
        "neural", "networks", "deep", "artificial", "intelligence",
    ];
    
    (0..count)
        .map(|i| {
            let mut words = Vec::new();
            for j in 0..length {
                let word_idx = (i + j) % base_words.len();
                words.push(base_words[word_idx]);
            }
            // Add some variations
            if i % 10 == 0 {
                words.push("duplicate");
            }
            words.join(" ")
        })
        .collect()
}

fn bench_minhash_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash_compute");
    
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    
    let doc_short = "short document";
    let doc_medium = "the quick brown fox jumps over the lazy dog and then runs away";
    let doc_long = "word ".repeat(1000);
    
    group.bench_function("short_doc", |b| {
        b.iter(|| {
            black_box(hasher.compute_str(black_box(doc_short), 0).unwrap());
        });
    });
    
    group.bench_function("medium_doc", |b| {
        b.iter(|| {
            black_box(hasher.compute_str(black_box(doc_medium), 0).unwrap());
        });
    });
    
    group.bench_function("long_doc", |b| {
        b.iter(|| {
            black_box(hasher.compute_str(black_box(&doc_long), 0).unwrap());
        });
    });
    
    group.finish();
}

fn bench_signature_size_variations(c: &mut Criterion) {
    let mut group = c.benchmark_group("signature_size");
    
    let doc = "the quick brown fox jumps over the lazy dog ".repeat(10);
    
    for size in [64, 128, 256].iter() {
        let config = Config::default().with_signature_size(*size);
        let hasher = MinHasher::new(&config).unwrap();
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(hasher.compute_str(black_box(&doc), 0).unwrap());
            });
        });
    }
    
    group.finish();
}

fn bench_lsh_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_insert");
    
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    let docs = generate_documents(100, 50);
    let signatures: Vec<_> = docs
        .iter()
        .enumerate()
        .map(|(i, doc)| hasher.compute_str(doc, i).unwrap())
        .collect();
    
    group.bench_function("insert_100_docs", |b| {
        b.iter(|| {
            let mut index = LshIndex::new(&config).unwrap();
            for sig in &signatures {
                black_box(index.insert(sig.clone()));
            }
        });
    });
    
    group.finish();
}

fn bench_similarity_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");
    
    let config = Config::default();
    let hasher = MinHasher::new(&config).unwrap();
    
    let doc1 = "the quick brown fox jumps over the lazy dog";
    let doc2 = "the quick brown fox jumps over the lazy cat";
    
    let sig1 = hasher.compute_str(doc1, 0).unwrap();
    let sig2 = hasher.compute_str(doc2, 1).unwrap();
    
    group.bench_function("signature_similarity", |b| {
        b.iter(|| {
            black_box(sig1.similarity(black_box(&sig2)));
        });
    });
    
    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    for doc_count in [10, 100, 1000].iter() {
        let docs = generate_documents(*doc_count, 50);
        
        group.bench_with_input(
            BenchmarkId::new("batch", doc_count),
            doc_count,
            |b, _| {
                b.iter(|| {
                    let config = Config::default();
                    let hasher = MinHasher::new(&config).unwrap();
                    let mut index = LshIndex::new(&config).unwrap();
                    
                    for (i, doc) in docs.iter().enumerate() {
                        let sig = hasher.compute_str(doc, i).unwrap();
                        index.insert(sig);
                    }
                    
                    black_box(index.find_clusters());
                });
            },
        );
    }
    
    group.finish();
}

fn bench_text_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_normalization");
    
    let text_short = "Hello World";
    let text_medium = "The Quick Brown Fox Jumps Over The Lazy Dog";
    let text_long = &"The Quick Brown Fox ".repeat(100);
    
    group.bench_function("short", |b| {
        b.iter(|| {
            black_box(normalize_text(black_box(text_short)));
        });
    });
    
    group.bench_function("medium", |b| {
        b.iter(|| {
            black_box(normalize_text(black_box(text_medium)));
        });
    });
    
    group.bench_function("long", |b| {
        b.iter(|| {
            black_box(normalize_text(black_box(text_long)));
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_minhash_computation,
    bench_signature_size_variations,
    bench_lsh_insertion,
    bench_similarity_estimation,
    bench_batch_processing,
    bench_text_normalization
);
criterion_main!(benches);
