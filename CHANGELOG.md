# Changelog

## [0.1.3] - 2026-08-07

### Added
- Re-exported `exact_jaccard_similarity` and `expected_error` in `lib.rs` as public utilities.

### Fixed
- Log `warn!` on MinHash signature length mismatch in `similarity` computation.
- Updated crate author metadata to `Santh <64453045+santhreal@users.noreply.github.com>`.
- Set package status to `beta` in `package.metadata.santh`.
- Optimized heavy adversarial test execution in `break_it.rs` and `test_depth_adversarial.rs`.

## 0.1.1

- Fail closed on oversized configurations: `Config::new` rejects a
  `signature_size` above the new `MAX_SIGNATURE_SIZE` (65536) with an
  actionable error instead of dying on a capacity-overflow panic deep in
  `LshIndex::new` or `FastHasher::new`.
- `Config::with_signature_size` clamps before snapping to a band multiple;
  a request near `usize::MAX` previously overflowed in `div_ceil * bands`
  (panic in debug, silent wrap in release).
- `FastHasher::new` clamps a hostile `num_hashes` to `MAX_SIGNATURE_SIZE`
  instead of aborting on allocation failure.
- `MinHasher::compute_batch` reports a per-entry error on doc-id overflow
  (`start_id` near `usize::MAX`) instead of panicking in debug or silently
  aliasing doc 0 in release.
- `DedupTransformer::finish_batch` uses saturating doc-id arithmetic,
  matching `process_sample`.

## 0.1.0  -  2025-04-12

- Initial release.
- MinHash + LSH near-duplicate detection for text datasets.
- `DedupTransformer` and `StatefulDedupTransform` for tenshift pipeline integration.
- Configurable signature size, band count, shingle size, and similarity threshold.
- Fast hash family using Mersenne-prime modular arithmetic.
- Adversarial hardening: bucket size limits and doc-id bounds.

## [0.1.2] - 2026-08-02

### Fixed
- README examples were stale or did not compile against the real API. They are rewritten and wired as doctests, so documentation drift now fails `cargo test`.

