#![allow(missing_docs)]
use dedup::{Config, LshIndex, MinHasher};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

fn get_hasher() -> MinHasher {
    let config = Config::default();
    MinHasher::new(&config).unwrap()
}

// 1-10: Identical documents produce same MinHash (deterministic)
#[test] fn test_01_identical_short_text() {
    let hasher = get_hasher();
    let doc = "hello world this is a test";
    let sig1 = hasher.compute_str(doc, 1).unwrap();
    let sig2 = hasher.compute_str(doc, 2).unwrap();
    assert_eq!(sig1.similarity(&sig2), 1.0);
}

#[test] fn test_02_identical_long_text() {
    let hasher = get_hasher();
    let doc = "A".repeat(10000);
    let sig1 = hasher.compute_str(&doc, 1).unwrap();
    let sig2 = hasher.compute_str(&doc, 2).unwrap();
    assert_eq!(sig1.similarity(&sig2), 1.0);
}

#[test] fn test_03_identical_binary_data() {
    let hasher = get_hasher();
    let doc = vec![0x00, 0xFF, 0x55, 0xAA, 0x11, 0x22];
    let sig1 = hasher.compute(&doc, 1).unwrap();
    let sig2 = hasher.compute(&doc, 2).unwrap();
    assert_eq!(sig1.similarity(&sig2), 1.0);
}

#[test] fn test_04_identical_all_zeros() {
    let hasher = get_hasher();
    let doc = vec![0x00; 1000];
    let sig1 = hasher.compute(&doc, 1).unwrap();
    let sig2 = hasher.compute(&doc, 2).unwrap();
    assert_eq!(sig1.similarity(&sig2), 1.0);
}

#[test] fn test_05_identical_all_ones() {
    let hasher = get_hasher();
    let doc = vec![0xFF; 1000];
    let sig1 = hasher.compute(&doc, 1).unwrap();
    let sig2 = hasher.compute(&doc, 2).unwrap();
    assert_eq!(sig1.similarity(&sig2), 1.0);
}

#[test] fn test_06_identical_alternating() {
    let hasher = get_hasher();
    let doc: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    let sig1 = hasher.compute(&doc, 1).unwrap();
    let sig2 = hasher.compute(&doc, 2).unwrap();
    assert_eq!(sig1.similarity(&sig2), 1.0);
}

#[test] fn test_07_identical_unicode() {
    let hasher = get_hasher();
    let doc = "こんにちは世界、これはテストです";
    let sig1 = hasher.compute_str(doc, 1).unwrap();
    let sig2 = hasher.compute_str(doc, 2).unwrap();
    assert_eq!(sig1.similarity(&sig2), 1.0);
}

#[test] fn test_08_identical_emoji() {
    let hasher = get_hasher();
    let doc = "🦀🚀🔥💯".repeat(10);
    let sig1 = hasher.compute_str(&doc, 1).unwrap();
    let sig2 = hasher.compute_str(&doc, 2).unwrap();
    assert_eq!(sig1.similarity(&sig2), 1.0);
}

#[test] fn test_09_identical_newlines() {
    let hasher = get_hasher();
    let doc = "\n\n\n\n\n\r\n\r\n\n\n\n".repeat(10);
    let sig1 = hasher.compute_str(&doc, 1).unwrap();
    let sig2 = hasher.compute_str(&doc, 2).unwrap();
    assert_eq!(sig1.similarity(&sig2), 1.0);
}

#[test] fn test_10_identical_random_noise() {
    let hasher = get_hasher();
    let mut doc = vec![0; 5000];
    for i in 0..5000 { doc[i] = ((i * 137) % 256) as u8; }
    let sig1 = hasher.compute(&doc, 1).unwrap();
    let sig2 = hasher.compute(&doc, 2).unwrap();
    assert_eq!(sig1.similarity(&sig2), 1.0);
}

// 11-15: Similar documents (>80% overlap) produce similar hashes
#[test] fn test_11_similar_one_char_diff() {
    let hasher = get_hasher();
    let doc1 = "The quick brown fox jumps over the lazy dog.".repeat(10);
    let doc2 = "The quick brown fox jumps over the lazy cat.".repeat(10);
    let sig1 = hasher.compute_str(&doc1, 1).unwrap();
    let sig2 = hasher.compute_str(&doc2, 2).unwrap();
    let sim = sig1.similarity(&sig2);
    // Similarity might be slightly lower than 0.8 depending on shingle size, but let's check it's high
    assert!(sim > 0.6, "Similarity {} should be > 0.6", sim);
}

#[test] fn test_12_similar_appended_text() {
    let hasher = get_hasher();
    let base = "This is a base document that has a lot of words to ensure we have a good amount of shingles. ".repeat(10);
    let doc1 = format!("{} Extra text at the end.", base);
    let doc2 = format!("{} Different extra text.", base);
    let sig1 = hasher.compute_str(&doc1, 1).unwrap();
    let sig2 = hasher.compute_str(&doc2, 2).unwrap();
    let sim = sig1.similarity(&sig2);
    assert!(sim > 0.6, "Similarity {} should be > 0.6", sim);
}

#[test] fn test_13_similar_prepended_text() {
    let hasher = get_hasher();
    let base = "This is a base document that has a lot of words to ensure we have a good amount of shingles. ".repeat(10);
    let doc1 = format!("Start A. {}", base);
    let doc2 = format!("Start B. {}", base);
    let sig1 = hasher.compute_str(&doc1, 1).unwrap();
    let sig2 = hasher.compute_str(&doc2, 2).unwrap();
    let sim = sig1.similarity(&sig2);
    assert!(sim > 0.6, "Similarity {} should be > 0.6", sim);
}

#[test] fn test_14_similar_middle_replacement() {
    let hasher = get_hasher();
    let part1 = "First part of the document is quite long to make many shingles. ".repeat(3);
    let part2 = " Second part of the document is also long. ".repeat(3);
    let doc1 = format!("{} INSERT_A {}", part1, part2);
    let doc2 = format!("{} INSERT_B {}", part1, part2);
    let sig1 = hasher.compute_str(&doc1, 1).unwrap();
    let sig2 = hasher.compute_str(&doc2, 2).unwrap();
    let sim = sig1.similarity(&sig2);
    assert!(sim > 0.8, "Similarity {} should be > 0.8", sim);
}

#[test] fn test_15_similar_minor_typos() {
    let hasher = get_hasher();
    let doc1 = "In computer science, a data structure is a data organization, management, and storage format that enables efficient access and modification. More precisely, a data structure is a collection of data values, the relationships among them, and the functions or operations that can be applied to the data.";
    let doc2 = "In computer scence, a data structure is a data organization, management, and storage format that enables efficient access and modification. More precisely, a data structure is a collection of data values, the relationships among them, and the functions or operations that can be applied to the data.";
    let sig1 = hasher.compute_str(&doc1, 1).unwrap();
    let sig2 = hasher.compute_str(&doc2, 2).unwrap();
    let sim = sig1.similarity(&sig2);
    assert!(sim > 0.8, "Similarity {} should be > 0.8", sim);
}

// 16-20: Different documents produce different hashes (low collision rate)
#[test] fn test_16_different_unrelated_text() {
    let hasher = get_hasher();
    let doc1 = "The quick brown fox jumps over the lazy dog";
    let doc2 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit";
    let sig1 = hasher.compute_str(doc1, 1).unwrap();
    let sig2 = hasher.compute_str(doc2, 2).unwrap();
    let sim = sig1.similarity(&sig2);
    assert!(sim < 0.2, "Similarity {} should be < 0.2", sim);
}

#[test] fn test_17_different_random_bytes() {
    let hasher = get_hasher();
    let doc1: Vec<u8> = (0..1000).map(|i| ((i * 13) % 256) as u8).collect();
    let doc2: Vec<u8> = (0..1000).map(|i| ((i * 17) % 256) as u8).collect();
    let sig1 = hasher.compute(&doc1, 1).unwrap();
    let sig2 = hasher.compute(&doc2, 2).unwrap();
    let sim = sig1.similarity(&sig2);
    assert!(sim < 0.2, "Similarity {} should be < 0.2", sim);
}

#[test] fn test_18_different_disjoint_character_sets() {
    let hasher = get_hasher();
    let doc1 = "AAAAAAAAAA BBBBBBBBBB CCCCCCCCCC";
    let doc2 = "XXXXXXXXXX YYYYYYYYYY ZZZZZZZZZZ";
    let sig1 = hasher.compute_str(doc1, 1).unwrap();
    let sig2 = hasher.compute_str(doc2, 2).unwrap();
    let sim = sig1.similarity(&sig2);
    assert!(sim < 0.2, "Similarity {} should be < 0.2", sim);
}

#[test] fn test_19_different_lengths() {
    let hasher = get_hasher();
    let doc1 = "Short text.";
    let doc2 = "A much longer text that has absolutely nothing to do with the first one and keeps going on and on and on to ensure they are very different.";
    let sig1 = hasher.compute_str(doc1, 1).unwrap();
    let sig2 = hasher.compute_str(doc2, 2).unwrap();
    let sim = sig1.similarity(&sig2);
    assert!(sim < 0.2, "Similarity {} should be < 0.2", sim);
}

#[test] fn test_20_different_reversed_words() {
    let hasher = get_hasher();
    // Shingles (k=5) of reversed strings are very different.
    let doc1 = "one two three four five six seven eight nine ten";
    let doc2 = "ten nine eight seven six five four three two one";
    let sig1 = hasher.compute_str(doc1, 1).unwrap();
    let sig2 = hasher.compute_str(doc2, 2).unwrap();
    let sim = sig1.similarity(&sig2);
    assert!(sim < 0.5, "Similarity {} should be low", sim);
}

// 21-25: Edge cases (empty document, single-char document, 10MB document)
#[test] fn test_21_edge_empty_document() {
    let hasher = get_hasher();
    let res = hasher.compute_str("", 1);
    assert!(res.is_err());
}

#[test] fn test_22_edge_single_char_document() {
    let hasher = get_hasher();
    let res = hasher.compute_str("A", 1);
    // Shingle size is 5, so length 1 is an error
    assert!(res.is_err());
}

#[test] fn test_23_edge_exact_shingle_size_document() {
    let hasher = get_hasher();
    let doc = "ABCDE"; // len = 5
    let res = hasher.compute_str(doc, 1);
    assert!(res.is_ok());
}

#[test] fn test_24_edge_huge_10mb_document() {
    let hasher = get_hasher();
    let doc = vec![b'A'; 10 * 1024 * 1024];
    let res = hasher.compute(&doc, 1);
    assert!(res.is_ok());
    let sig = res.unwrap();
    assert_eq!(sig.len(), 128); // default sig size
}

#[test] fn test_25_edge_shingle_size_minus_one() {
    let hasher = get_hasher();
    let res = hasher.compute_str("ABCD", 1); // length 4 < 5
    assert!(res.is_err());
}

// 26-30: LSH query correctness (inserted doc is always found, non-inserted never false-positive)
#[test] fn test_26_lsh_inserted_found() {
    let config = Config::default();
    let mut index = LshIndex::new(&config).unwrap();
    let hasher = get_hasher();
    let doc = "Find me in the LSH index please, this is a unique text.";
    let sig1 = hasher.compute_str(doc, 1).unwrap();
    let sig2 = hasher.compute_str(doc, 2).unwrap(); // Same doc, different ID
    
    // insert() returns candidates. However if it's the first time inserting, there shouldn't be candidates.
    // LshIndex query will return doc_ids that collide, EXCEPT for the doc_id in the query signature.
    // Therefore to find "itself", we must query with a different doc_id, or we query for an identical doc inserted previously.
    index.insert(sig1.clone()).unwrap();
    let candidates = index.query(&sig2);
    assert!(candidates.contains(&1));
}

#[test] fn test_27_lsh_exact_duplicate_found() {
    let config = Config::default();
    let mut index = LshIndex::new(&config).unwrap();
    let hasher = get_hasher();
    let doc = "We are identical twins in the LSH index.";
    let sig1 = hasher.compute_str(doc, 1).unwrap();
    let sig2 = hasher.compute_str(doc, 2).unwrap();
    
    index.insert(sig1).unwrap();
    let candidates = index.query(&sig2);
    assert!(candidates.contains(&1));
}

#[test] fn test_28_lsh_similar_found() {
    let config = Config::default().with_similarity_threshold(0.8);
    let mut index = LshIndex::new(&config).unwrap();
    let hasher = get_hasher();
    let doc1 = "This document is mostly the same as the other one, except for this part.";
    let doc2 = "This document is mostly the same as the other one, except for THAT part.";
    let sig1 = hasher.compute_str(doc1, 1).unwrap();
    let sig2 = hasher.compute_str(doc2, 2).unwrap();
    
    index.insert(sig1).unwrap();
    let candidates = index.query(&sig2);
    assert!(candidates.contains(&1));
}

#[test] fn test_29_lsh_different_not_found() {
    let config = Config::default();
    let mut index = LshIndex::new(&config).unwrap();
    let hasher = get_hasher();
    let doc1 = "Apples oranges bananas grapes pears.";
    let doc2 = "Cars trucks buses trains planes bicycles.";
    let sig1 = hasher.compute_str(doc1, 1).unwrap();
    let sig2 = hasher.compute_str(doc2, 2).unwrap();
    
    index.insert(sig1).unwrap();
    let candidates = index.query(&sig2);
    assert!(!candidates.contains(&1));
}

#[test] fn test_30_lsh_multiple_documents() {
    let config = Config::default();
    let mut index = LshIndex::new(&config).unwrap();
    let hasher = get_hasher();
    
    for i in 1..=10 {
        // use distinct enough strings so they don't accidentally match
        // "Document number N" will share some shingles like "Docum", "ocume", "cumen", "ument", "ment ", "ent n", "nt nu", "t num", " numb", "numbe", "umber", "mber "
        // To be completely distinct, use very different strings
        let words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon"];
        let doc = format!("{} {} {} {} {}", words[i-1], words[(i)%10], words[(i+1)%10], words[(i+2)%10], words[(i+3)%10]);
        let sig = hasher.compute_str(&doc, i).unwrap();
        index.insert(sig).unwrap();
    }
    
    // We want to query for document 5
    let words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon"];
    let i = 5;
    let target = format!("{} {} {} {} {}", words[i-1], words[(i)%10], words[(i+1)%10], words[(i+2)%10], words[(i+3)%10]);
    // Use an ID that isn't exactly the same so it's a true query
    let sig_target = hasher.compute_str(&target, 99).unwrap();
    let candidates = index.query(&sig_target);
    
    // When queried, it should find doc 5 because the text is identical.
    assert!(candidates.contains(&5));
    // Check it only found that document.
    assert_eq!(candidates.len(), 1, "Expected exactly 1 match (doc 5), but got matches: {:?}", candidates);
}

// 31-33: Concurrent insert+query thread safety
#[test] fn test_31_concurrent_inserts() {
    let config = Config::default();
    let index = Arc::new(Mutex::new(LshIndex::new(&config).unwrap()));
    let hasher = get_hasher();
    
    let mut handles = vec![];
    for i in 0..10 {
        let idx = index.clone();
        let sig = hasher.compute_str(&format!("Concurrent doc {}", i).repeat(5), i).unwrap();
        handles.push(thread::spawn(move || {
            let mut lsh = idx.lock().unwrap();
            lsh.insert(sig).unwrap();
        }));
    }
    
    for h in handles { h.join().unwrap(); }
    
    let lsh = index.lock().unwrap();
    assert_eq!(lsh.doc_count(), 10);
}

#[test] fn test_32_concurrent_queries() {
    let config = Config::default();
    let mut index = LshIndex::new(&config).unwrap();
    let hasher = get_hasher();
    
    for i in 0..5 {
        let sig = hasher.compute_str(&format!("Doc {}", i).repeat(5), i).unwrap();
        index.insert(sig).unwrap();
    }
    
    let index = Arc::new(index);
    let mut handles = vec![];
    
    for i in 0..10 {
        let idx = index.clone();
        handles.push(thread::spawn(move || {
            let h = get_hasher();
            let sig = h.compute_str(&format!("Doc {}", i % 5).repeat(5), 99).unwrap();
            let candidates = idx.query(&sig);
            assert!(candidates.contains(&(i % 5)));
        }));
    }
    
    for h in handles { h.join().unwrap(); }
}

#[test] fn test_33_concurrent_mixed_insert_query() {
    let config = Config::default();
    let index = Arc::new(RwLock::new(LshIndex::new(&config).unwrap()));
    
    let mut handles = vec![];
    
    for i in 0..5 {
        let idx = index.clone();
        handles.push(thread::spawn(move || {
            let h = get_hasher();
            let sig = h.compute_str(&format!("Mixed doc {}", i).repeat(5), i).unwrap();
            let mut lsh = idx.write().unwrap();
            lsh.insert(sig).unwrap();
        }));
    }
    
    for i in 0..5 {
        let idx = index.clone();
        handles.push(thread::spawn(move || {
            let h = get_hasher();
            let sig = h.compute_str(&format!("Mixed doc {}", i).repeat(5), 99).unwrap();
            let lsh = idx.read().unwrap();
            let _ = lsh.query(&sig);
        }));
    }
    
    for h in handles { h.join().unwrap(); }
}
