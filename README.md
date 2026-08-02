> Part of the [Santh](https://santh.dev) security research ecosystem.

# dedup

[![CI](https://github.com/santhreal/dedup/actions/workflows/ci.yml/badge.svg)](https://github.com/santhreal/dedup/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Crates.io](https://img.shields.io/crates/v/dedup)](https://crates.io/crates/dedup)

High-performance dataset deduplication for ML training data using MinHash + LSH.

## Features

- **MinHash + LSH**: Industry-standard near-duplicate detection with configurable similarity thresholds
- **Fast Hashing**: Efficient hash computation for high throughput
- **Streaming**: Process billions of documents without loading all into memory
- **tenshift Integration**: Plugs into data pipelines as a `Transform`
- **Configurable**: Tune signature size, band count, and thresholds for your data
- **Robust**: Fuzz-quality input handling, zero `unwrap` in production code

## Quick Start

```rust
use dedup::{Config, DedupTransformer};
use dedup::tenshift::Sample;
use tenshift_core::sample::Tensor;

let config = Config::default()
    .with_similarity_threshold(0.9);

let mut dedup = DedupTransformer::new(config)?;

// Add documents
dedup.push(Sample::new().with("text", Tensor::bytes(b"the quick brown fox".to_vec())));
dedup.push(Sample::new().with("text", Tensor::bytes(b"the quick brown fox".to_vec())));

// Get deduplicated results: the exact duplicate collapses to one entry.
let unique = dedup.finish_batch();
assert_eq!(unique.len(), 1);
# Ok::<(), dedup::Error>(())
```

## How It Works

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│   Shingle   │────▶│   MinHash    │────▶│  LSH Bands  │────▶│   Cluster   │
│  (k-grams)  │     │  (fast hash) │     │  (buckets)  │     │    Output   │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
```

1. **Shingling**: Convert documents to sets of k-grams (overlapping subsequences)
2. **MinHash**: Compress documents to small signatures preserving Jaccard similarity
3. **LSH**: Band signatures such that similar documents collide in buckets
4. **Clustering**: Group colliding documents and output unique representatives

## Configuration

```rust
let config = dedup::Config::new(
    128,    // signature size (hash functions)
    16,     // LSH bands
    5,      // shingle size
    0.9,    // similarity threshold
)?;
# Ok::<(), dedup::Error>(())
```

### Parameter Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `signature_size` | 128 | Number of hash functions. Higher = more accurate but slower |
| `num_bands` | 16 | LSH bands. Higher = more sensitive but more false positives |
| `shingle_size` | 5 | k-gram length. 4-7 works well for text |
| `threshold` | 0.9 | Similarity threshold. 0.85-0.95 recommended |

## tenshift Integration

```rust
use dedup::{Config, StatefulDedupTransform};
use dedup::tenshift::Sample;
use tenshift_core::sample::Tensor;
use tenshift_core::transform::StatefulTransform;

let config = Config::default();
let mut dedup = StatefulDedupTransform::new(config)?;

// In your pipeline
let samples = vec![
    Sample::new().with("text", Tensor::bytes(b"hello world".to_vec())),
    Sample::new().with("text", Tensor::bytes(b"hello world".to_vec())),
];
for sample in samples {
    let output = dedup.push(sample);
    // output is empty until finish() is called
    assert!(output.is_empty());
}

let unique = dedup.finish();
assert_eq!(unique.len(), 1);
# Ok::<(), dedup::Error>(())
```

## Performance

Benchmarked on AMD Ryzen 9 5950X:

| Operation | Throughput |
|-----------|------------|
| MinHash (128 sig) | ~50,000 docs/sec |
| MinHash (256 sig) | ~25,000 docs/sec |
| LSH Insert | ~100,000 ops/sec |
| Batch (1000 docs) | ~30ms |

## Architecture

### MinHash Theory

MinHash estimates Jaccard similarity between sets:
- `J(A,B) = |A ∩ B| / |A ∪ B|`
- MinHash approximates this by comparing hash signatures
- Expected error: `√(s(1-s)/k)` where `s` is similarity, `k` is signature size

### LSH Theory

LSH reduces O(n²) comparisons to O(n):
- Signature split into `b` bands of `r` rows each
- Probability of collision for similarity `s`: `1 - (1 - s^r)^b`
- Threshold (S-curve inflection): `t ≈ (1/b)^(1/r)`

## Testing

```bash
cargo test
cargo test --release  # For benchmarks
cargo clippy -- -D warnings
```

## License

MIT License - See LICENSE file for details.

## Authors

Santh <contact@santh.dev>
