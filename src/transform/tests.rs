    use super::*;
    use tenshift_core::sample::{Sample, Tensor};
    use tenshift_core::transform::StatefulTransform;

    fn create_text_sample(text: impl Into<String>, idx: u64) -> Sample {
        let text = text.into();
        Sample::new()
            .with("text", Tensor::bytes(text.into_bytes()))
            .with_metadata("test", idx)
    }

    #[test]
    fn transformer_creation() {
        let config = Config::default();
        let transformer = DedupTransformer::new(config);
        assert!(transformer.is_ok());
    }

    #[test]
    fn process_unique_document() {
        let config = Config::default();
        let mut transformer = DedupTransformer::new(config).unwrap();
        
        let sample = create_text_sample("unique document content", 0);
        let is_unique = transformer.process_sample(&sample).unwrap();
        
        assert!(is_unique);
    }

    #[test]
    fn process_duplicate_document() {
        let config = Config::default()
            .with_similarity_threshold(0.8);
        let mut transformer = DedupTransformer::new(config).unwrap();
        
        let sample1 = create_text_sample("hello world test document content here", 0);
        let sample2 = create_text_sample("hello world test document content here", 1);
        
        let is_unique1 = transformer.process_sample(&sample1).unwrap();
        let is_unique2 = transformer.process_sample(&sample2).unwrap();
        
        assert!(is_unique1);
        assert!(!is_unique2); // Second is duplicate
    }

    #[test]
    fn batch_processing() {
        let config = Config::default();
        let mut transformer = DedupTransformer::new(config).unwrap();
        
        // Add some samples
        transformer.push(create_text_sample("document one content here", 0));
        transformer.push(create_text_sample("document two different content", 1));
        transformer.push(create_text_sample("document one content here", 2)); // Duplicate
        
        let result = transformer.finish_batch();
        
        assert_eq!(result.len(), 2); // Two unique documents
    }

    #[test]
    fn extract_text_from_sample() {
        let config = Config::default();
        let transformer = DedupTransformer::new(config).unwrap();
        
        let sample = create_text_sample("hello world", 0);
        let text = transformer.extract_text(&sample).unwrap();
        
        assert_eq!(text, "hello world");
    }

    #[test]
    fn extract_text_missing_field() {
        let config = Config::default();
        let transformer = DedupTransformer::new(config).unwrap();
        
        let sample = Sample::new()
            .with("other_field", Tensor::bytes(vec![1, 2, 3]));
        
        let result = transformer.extract_text(&sample);
        assert!(result.is_err());
    }

    #[test]
    fn stateful_transform_finish_outputs_all() {
        let config = Config::default();
        let mut transform = StatefulDedupTransform::new(config).unwrap();
        
        // Push some samples
        transform.push(create_text_sample("doc a content", 0));
        transform.push(create_text_sample("doc b different", 1));
        transform.push(create_text_sample("doc a content", 2)); // Duplicate
        
        // Nothing output until finish
        let output = transform.push(create_text_sample("doc c unique", 3));
        assert!(output.is_empty());
        
        // Finish outputs deduplicated samples
        let result = transform.finish();
        assert_eq!(result.len(), 3); // 3 unique docs
    }

    #[test]
    fn stateful_transform_empty_finish() {
        let config = Config::default();
        let mut transform = StatefulDedupTransform::new(config).unwrap();
        
        let result = transform.finish();
        assert!(result.is_empty());
    }


    #[test]
    fn stats_available_after_processing() {
        let config = Config::default();
        let mut transformer = DedupTransformer::new(config).unwrap();
        
        transformer.push(create_text_sample("content here", 0));
        transformer.push(create_text_sample("content here", 1)); // Duplicate
        transformer.finish_batch();
        
        let stats = transformer.stats();
        assert_eq!(stats.doc_count, 2);
        assert_eq!(stats.duplicate_count, 1);
    }

    #[test]
    fn reset_clears_state() {
        let config = Config::default();
        let mut transformer = DedupTransformer::new(config).unwrap();
        
        transformer.push(create_text_sample("content", 0));
        transformer.push(create_text_sample("content", 1)); // duplicate of doc 0
        transformer.finish_batch();

        assert_eq!(transformer.stats().doc_count, 2);
        assert_eq!(
            transformer.duplicate_count(),
            1,
            "the two identical docs collapse to one duplicate"
        );

        transformer.reset();

        // The index must be fully cleared - not a stale populated index kept
        // behind a swallowed `LshIndex::new` error (the fixed silent fallback):
        // doc_count and duplicate_count return to zero.
        assert_eq!(transformer.stats().doc_count, 0, "reset must empty the index");
        assert_eq!(transformer.duplicate_count(), 0, "no duplicates survive reset");

        // Reuse after reset: pushing content identical to a PRE-reset document
        // must be treated as brand-new (unique), proving no ghost entry lingers
        // to cause a false duplicate or a doc_id collision against stale state.
        transformer.push(create_text_sample("content", 0));
        transformer.finish_batch();
        assert_eq!(transformer.stats().doc_count, 1, "one fresh document after reset");
        assert_eq!(
            transformer.duplicate_count(),
            0,
            "the re-pushed document is unique, not a ghost duplicate of a cleared entry"
        );
    }

    #[test]
    fn builder_pattern() {
        let config = Config::default();
        let transformer = DedupTransformer::new(config)
            .unwrap()
            .with_text_field("content")
            .with_streaming(true)
            .with_mark_duplicates(true);
        
        assert_eq!(transformer.text_field, "content");
        assert!(transformer.streaming);
        assert!(transformer.mark_duplicates);
    }

    #[test]
    fn unique_and_duplicate_counts() {
        let config = Config::default();
        let mut transformer = DedupTransformer::new(config).unwrap();
        
        // 4 documents: 2 unique, 2 duplicates
        transformer.push(create_text_sample("doc a", 0));
        transformer.push(create_text_sample("doc b", 1));
        transformer.push(create_text_sample("doc a", 2)); // Dup of 0
        transformer.push(create_text_sample("doc b", 3)); // Dup of 1
        
        transformer.finish_batch();
        
        assert_eq!(transformer.duplicate_count(), 2);
        assert_eq!(transformer.unique_count(), 2);
    }

    #[test]
    fn samples_without_text_field_are_each_kept_not_collapsed() {
        // Regression: samples lacking the dedup text field all hashed to the same
        // empty key (unwrap_or_default), so only the first survived finish_batch
        // and every later field-less sample was silently dropped as a false
        // "duplicate". Each is a distinct document and must be kept.
        let config = Config::default();
        let mut transformer = DedupTransformer::new(config).unwrap();

        for idx in 0..3u64 {
            transformer.push(Sample::new().with_metadata("test", idx));
        }

        let result = transformer.finish_batch();
        assert_eq!(
            result.len(),
            3,
            "each field-less sample must be kept, not collapsed under one empty key"
        );
    }

    #[test]
    fn samples_with_present_but_empty_text_field_still_dedup_by_content() {
        // The fix must not regress intended behaviour: two samples whose text
        // field is present but empty are genuinely identical content and should
        // still collapse to one unique document.
        let config = Config::default();
        let mut transformer = DedupTransformer::new(config).unwrap();

        transformer.push(create_text_sample("", 0));
        transformer.push(create_text_sample("", 1));

        let result = transformer.finish_batch();
        assert_eq!(
            result.len(),
            1,
            "two present-but-empty text fields are identical content and dedup to one"
        );
    }
    #[test]
    fn mark_duplicates_tags_all_samples_with_is_duplicate_tensor() {
        let config = Config::default();
        let mut transformer = DedupTransformer::new(config)
            .unwrap()
            .with_mark_duplicates(true);

        transformer.push(create_text_sample("unique document a", 0));
        transformer.push(create_text_sample("unique document b", 1));
        transformer.push(create_text_sample("unique document a", 2)); // Duplicate of 0

        let result = transformer.finish_batch();
        assert_eq!(result.len(), 3, "mark_duplicates preserves all 3 samples in batch");

        // Verify doc 0 (unique) is tagged 0
        let s0 = &result[0];
        let tag0 = s0.get("is_duplicate").expect("is_duplicate tensor present");
        assert_eq!(tag0.as_bytes(), &[0]);

        // Verify doc 1 (unique) is tagged 0
        let s1 = &result[1];
        let tag1 = s1.get("is_duplicate").expect("is_duplicate tensor present");
        assert_eq!(tag1.as_bytes(), &[0]);

        // Verify doc 2 (dup) is tagged 1
        let s2 = &result[2];
        let tag2 = s2.get("is_duplicate").expect("is_duplicate tensor present");
        assert_eq!(tag2.as_bytes(), &[1]);
    }

    #[test]
    fn streaming_mode_buffers_immediate_output() {
        let config = Config::default();
        let mut transformer = DedupTransformer::new(config)
            .unwrap()
            .with_streaming(true);

        transformer.push(create_text_sample("streaming doc one", 0));
        transformer.push(create_text_sample("streaming doc two", 1));
        transformer.push(create_text_sample("streaming doc one", 2)); // Duplicate

        let drained = transformer.drain_streaming();
        assert_eq!(drained.len(), 2, "streaming mode output queue has 2 unique samples");

        let finish_res = transformer.finish_batch();
        assert!(finish_res.is_empty(), "finish_batch is empty after drain_streaming");
    }
