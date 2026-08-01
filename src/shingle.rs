//! Shingling (k-gram generation) for text documents.
//!
//! Converts documents into sets of overlapping subsequences (shingles/k-grams)
//! which are then hashed for MinHash computation.

use crate::fast_hash::hash_bytes;

/// Iterator over shingles (k-grams) of a byte sequence.
///
/// Produces overlapping windows of size `k` from the input data.
#[derive(Debug, Clone)]
pub struct ShingleIterator<'a> {
    /// Input data.
    data: &'a [u8],
    /// Shingle size (window length).
    k: usize,
    /// Current position.
    pos: usize,
}

impl<'a> ShingleIterator<'a> {
    /// Create a new shingle iterator.
    ///
    /// # Panics
    ///
    /// Returns an empty iterator if `k` is 0.
    pub fn new(data: &'a [u8], k: usize) -> Self {
        Self { data, k: k.max(1), pos: 0 }
    }

    /// Create a new shingle iterator from a string.
    ///
    /// Shingles are computed over UTF-8 bytes.
    pub fn from_str(s: &'a str, k: usize) -> Self {
        Self::new(s.as_bytes(), k)
    }

    /// Get the number of shingles this iterator will produce.
    #[must_use]
    pub fn count_shingles(&self) -> usize {
        if self.data.len() < self.k {
            return 0;
        }
        self.data.len().saturating_sub(self.k).saturating_add(1)
    }

    /// Check if the document has any shingles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count_shingles() == 0
    }
}

impl<'a> Iterator for ShingleIterator<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.pos.checked_add(self.k)?;
        if end > self.data.len() {
            return None;
        }

        let shingle = &self.data[self.pos..end];
        self.pos += 1;
        Some(shingle)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count_shingles().saturating_sub(self.pos);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ShingleIterator<'_> {}

/// Iterator that yields hashed shingles directly.
///
/// More efficient than materializing all shingles first.
#[derive(Debug, Clone)]
pub struct HashedShingleIterator<'a> {
    inner: ShingleIterator<'a>,
}

impl<'a> HashedShingleIterator<'a> {
    /// Create a new hashed shingle iterator.
    pub fn new(data: &'a [u8], k: usize) -> Self {
        Self {
            inner: ShingleIterator::new(data, k),
        }
    }

    /// Create from a string.
    #[allow(dead_code)]
    pub fn from_str(s: &'a str, k: usize) -> Self {
        Self::new(s.as_bytes(), k)
    }
}

impl Iterator for HashedShingleIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(hash_bytes)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for HashedShingleIterator<'_> {}

/// Compute all shingles for a document.
#[must_use]
#[allow(dead_code)]
pub fn get_shingles(data: &[u8], k: usize) -> Vec<&[u8]> {
    ShingleIterator::new(data, k).collect()
}

/// Compute all hashed shingles for a document.
#[must_use]
#[allow(dead_code)]
pub fn get_hashed_shingles(data: &[u8], k: usize) -> Vec<u64> {
    HashedShingleIterator::new(data, k).collect()
}

/// Normalize text by lowercasing and removing excess whitespace.
///
/// This is useful for text deduplication where case differences
/// shouldn't affect similarity.
#[must_use]
#[allow(dead_code)]
pub fn normalize_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_space = true; // Skip leading whitespace

    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_was_space {
                result.push(' ');
                prev_was_space = true;
            }
        } else {
            // Use lowercase for case-insensitive comparison
            for lc in c.to_lowercase() {
                result.push(lc);
            }
            prev_was_space = false;
        }
    }

    // Trim trailing whitespace
    if result.ends_with(' ') {
        result.pop();
    }

    result
}

/// Create word-level shingles (n-grams of words).
///
/// Useful for document deduplication where word order matters
/// more than character-level variations.
#[allow(dead_code)]
pub struct WordShingleIterator<'a> {
    /// Words in the document.
    words: Vec<&'a str>,
    /// Shingle size in words.
    k: usize,
    /// Current position.
    pos: usize,
}

impl<'a> WordShingleIterator<'a> {
    /// Create a new word shingle iterator.
    ///
    /// # Panics
    ///
    /// Clamps `k` to at least 1.
    #[allow(dead_code)]
    pub fn new(text: &'a str, k: usize) -> Self {
        let k = k.max(1);
        let words: Vec<&str> = text.split_whitespace().collect();
        Self { words, k, pos: 0 }
    }

    /// Get the number of word shingles.
    #[must_use]
    #[allow(dead_code)]
    pub fn count_shingles(&self) -> usize {
        if self.words.len() < self.k {
            return 0;
        }
        self.words.len().saturating_sub(self.k).saturating_add(1)
    }
}

impl<'a> Iterator for WordShingleIterator<'a> {
    type Item = Vec<&'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.pos.checked_add(self.k)?;
        if end > self.words.len() {
            return None;
        }

        let shingle: Vec<&str> = self.words[self.pos..end].to_vec();
        self.pos += 1;
        Some(shingle)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count_shingles().saturating_sub(self.pos);
        (remaining, Some(remaining))
    }
}

// size_hint returns an exact (remaining, Some(remaining)) bound, so the default
// ExactSizeIterator::len() (the lower bound) is exact. Matches ShingleIterator
// and HashedShingleIterator above (ONE-PLACE consistency).
impl ExactSizeIterator for WordShingleIterator<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shingle_iterator_basic() {
        let data = b"hello";
        let shingles: Vec<_> = ShingleIterator::new(data, 2).collect();
        
        assert_eq!(shingles.len(), 4);
        assert_eq!(shingles[0], b"he");
        assert_eq!(shingles[1], b"el");
        assert_eq!(shingles[2], b"ll");
        assert_eq!(shingles[3], b"lo");
    }

    #[test]
    fn shingle_iterator_single_byte() {
        let data = b"abc";
        let shingles: Vec<_> = ShingleIterator::new(data, 1).collect();
        
        assert_eq!(shingles.len(), 3);
        assert_eq!(shingles[0], b"a");
        assert_eq!(shingles[1], b"b");
        assert_eq!(shingles[2], b"c");
    }

    #[test]
    fn shingle_iterator_short_input() {
        let data = b"hi";
        let shingles: Vec<_> = ShingleIterator::new(data, 5).collect();
        assert!(shingles.is_empty());
    }

    #[test]
    fn shingle_count_correct() {
        let iter = ShingleIterator::new(b"hello world", 3);
        assert_eq!(iter.count_shingles(), 9); // "hel", "ell", "llo", "lo ", "o w", " wo", "wor", "orl", "rld"
    }

    #[test]
    fn exact_size_iterator() {
        let iter = ShingleIterator::new(b"hello", 2);
        let (low, high) = iter.size_hint();
        assert_eq!(low, 4);
        assert_eq!(high, Some(4));
    }

    #[test]
    fn hashed_shingle_iterator() {
        let data = b"test";
        let hashes: Vec<_> = HashedShingleIterator::new(data, 2).collect();
        
        assert_eq!(hashes.len(), 3);
        // Hashes should be deterministic
        let hashes2: Vec<_> = HashedShingleIterator::new(data, 2).collect();
        assert_eq!(hashes, hashes2);
    }

    #[test]
    fn from_str_works() {
        let s = "hello";
        let shingles: Vec<_> = ShingleIterator::from_str(s, 2).collect();
        assert_eq!(shingles[0], b"he");
    }

    #[test]
    fn get_shingles_helper() {
        let shingles = get_shingles(b"abcd", 2);
        assert_eq!(shingles.len(), 3);
    }

    #[test]
    fn normalize_text_basic() {
        assert_eq!(normalize_text("Hello World"), "hello world");
        assert_eq!(normalize_text("  multiple   spaces  "), "multiple spaces");
    }

    #[test]
    fn normalize_text_unicode() {
        assert_eq!(normalize_text("CAFÉ"), "café");
        assert_eq!(normalize_text("Hello 世界"), "hello 世界");
    }

    #[test]
    fn word_shingle_iterator() {
        let text = "the quick brown fox";
        let shingles: Vec<_> = WordShingleIterator::new(text, 2).collect();
        
        assert_eq!(shingles.len(), 3);
        assert_eq!(shingles[0], vec!["the", "quick"]);
        assert_eq!(shingles[1], vec!["quick", "brown"]);
        assert_eq!(shingles[2], vec!["brown", "fox"]);
    }

    #[test]
    fn word_shingle_empty() {
        let shingles: Vec<_> = WordShingleIterator::new("", 2).collect();
        assert!(shingles.is_empty());
    }

    #[test]
    fn shingle_iterator_is_empty() {
        let iter = ShingleIterator::new(b"hi", 5);
        assert!(iter.is_empty());
        
        let iter = ShingleIterator::new(b"hello", 2);
        assert!(!iter.is_empty());
    }

    #[test]
    fn large_k_works() {
        let data = vec![b'a'; 1000];
        let shingles: Vec<_> = ShingleIterator::new(&data, 100).collect();
        assert_eq!(shingles.len(), 901);
    }
}
