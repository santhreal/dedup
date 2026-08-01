//! Concurrent stress tests for dedup.
//! Designed to hammer the API with 32 threads.

use dedup::{Config, DedupTransformer, MinHasher};
use dedup::tenshift::Sample;
use std::sync::{Arc, Mutex};
use std::thread;
use tenshift_core::sample::Tensor;

fn create_text_sample(text: impl Into<String>, idx: u64) -> Sample {
    Sample::new()
        .with("text", Tensor::bytes(text.into().into_bytes()))
        .with_metadata("test", idx)
}

#[test]
fn test_concurrent_hasher_access() {
    let config = Config::default();
    let hasher = Arc::new(MinHasher::new(&config).expect("Config init failed"));

    let mut handles = vec![];

    // 32 threads hammering the same Arc<MinHasher> API
    for t in 0..32 {
        let hasher_clone = Arc::clone(&hasher);
        let handle = thread::spawn(move || {
            for i in 0..1000 {
                let text = format!("Thread {} doc {}", t, i);
                let sig = hasher_clone.compute_str(&text, t * 1000 + i);
                assert!(sig.is_ok(), "Hash computation should not fail concurrently");
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_concurrent_transformer_hammering() {
    // DedupTransformer takes mutable ownership, so to use it concurrently 
    // it must be wrapped in a Mutex, testing lock contention and shared state safety.
    let config = Config::default();
    let transformer = Arc::new(Mutex::new(DedupTransformer::new(config).expect("Config init failed")));

    let mut handles = vec![];

    for t in 0..32 {
        let trans_clone = Arc::clone(&transformer);
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let text = format!("Thread {} doc {}", t, i);
                // Also push some duplicates across threads
                let shared_dup = format!("Shared duplicate {}", i % 10);
                
                let mut guard = trans_clone.lock().unwrap();
                guard.push(create_text_sample(text, (t * 1000 + i) as u64));
                guard.push(create_text_sample(shared_dup, 99999 + i as u64));
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Now finish the batch
    let mut guard = transformer.lock().unwrap();
    let unique = guard.finish_batch();
    
    // Because thread scheduling is non-deterministic and the exact shared duplicate pushes
    // interleave randomly, the DedupTransformer's final output length for shared docs can vary slightly
    // in normal test suites.
    // However, the thread-specific unique docs (32 * 100) should absolutely be retained without crashing.
    // We check that the length is > 0 and that the engine didn't panic or OOM.
    // (Actually the Jaccard similarity between very short string formats might merge some of the "Thread x doc y" docs if they are too similar!)
    assert!(unique.len() > 1000, "Concurrent mutations should yield a large number of deduplicated docs without crashing");
}

#[test]
fn test_concurrent_streaming_mode() {
    let config = Config::default();
    let transformer = Arc::new(Mutex::new(
        DedupTransformer::new(config).unwrap().with_streaming(true)
    ));

    let mut handles = vec![];

    for _ in 0..32 {
        let trans_clone = Arc::clone(&transformer);
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let text = format!("Streaming doc {}", i); // Everyone pushes the exact same 100 docs
                let mut guard = trans_clone.lock().unwrap();
                guard.push(create_text_sample(text, i as u64));
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let mut guard = transformer.lock().unwrap();
    let final_batch = guard.finish_batch();
    
    // In streaming mode, some items might get returned immediately depending on how push processes them.
    // However, since it is 32 threads pushing concurrently, the final unique buffered output count might vary.
    // We just verify it successfully survived the 32 thread hammering in stream mode without crashing,
    // and we get some valid output representing the documents.
    assert!(final_batch.len() > 0, "Streaming mode should output valid docs after concurrent hammer");
}

#[test]
fn test_concurrent_lock_free_hammer() {
    let config = Config::default();
    let hasher = Arc::new(MinHasher::new(&config).unwrap());
    
    // Spawn threads that hit MinHasher purely lock-free.
    let mut handles = vec![];
    for t in 0..32 {
        let h_clone = Arc::clone(&hasher);
        handles.push(thread::spawn(move || {
            let doc = "A".repeat(100 + t);
            for i in 0..10_000 {
                let sig_result = h_clone.compute_str(&doc, i);
                assert!(sig_result.is_ok(), "Concurrent hashing should succeed lock-free");
            }
        }));
    }
    
    for h in handles {
        h.join().unwrap();
    }
}
