//! Fast hash computation for MinHash signatures.
//!
//! Provides efficient hash functions for computing MinHash signatures.
//! Uses 128-bit arithmetic for modular reduction to avoid overflow
//! and provides good performance for shingle hashing.

/// A fast hasher for computing multiple hash values efficiently.
///
/// This struct uses a family of permutation hash functions suitable for MinHash.
/// The hash functions have the form: `h_i(x) = (a_i * x + b_i) mod p`
/// where `p` is a large prime and `a_i`, `b_i` are random odd coefficients.
#[derive(Debug, Clone)]
pub struct FastHasher {
    /// Coefficients 'a' for linear hash functions.
    coeffs_a: Vec<u64>,
    /// Coefficients 'b' for linear hash functions.
    coeffs_b: Vec<u64>,
    /// Number of hash functions.
    num_hashes: usize,
}

// Large Mersenne prime: 2^61 - 1
const MERSENNE_PRIME: u64 = (1_u64 << 61) - 1;

// Golden ratio constant for hash mixing
const PHI: u64 = 0x9e37_79b9_7f4a_7c15;

impl FastHasher {
    /// Create a new hasher with the given number of hash functions.
    ///
    /// A count of 0 returns a no-op hasher with empty coefficients. Counts
    /// above [`crate::config::MAX_SIGNATURE_SIZE`] are clamped to it: the
    /// coefficients are two `u64` vectors of this length, so an unbounded
    /// count (for example `usize::MAX` from a hostile or mistaken caller)
    /// would abort the process on allocation failure.
    pub fn new(num_hashes: usize, seed: u64) -> Self {
        let num_hashes = num_hashes.min(crate::config::MAX_SIGNATURE_SIZE);
        if num_hashes == 0 {
            return Self {
                coeffs_a: Vec::new(),
                coeffs_b: Vec::new(),
                num_hashes: 0,
            };
        }

        let mut coeffs_a = Vec::with_capacity(num_hashes);
        let mut coeffs_b = Vec::with_capacity(num_hashes);

        // Generate pseudo-random coefficients using splitmix64
        let mut state = seed.wrapping_add(PHI);
        for _ in 0..num_hashes {
            state = splitmix64(state);
            // Ensure 'a' is odd and non-zero for good mixing
            coeffs_a.push(state | 1);
            state = splitmix64(state);
            coeffs_b.push(state);
        }

        Self {
            coeffs_a,
            coeffs_b,
            num_hashes,
        }
    }

    /// Compute MinHash signature for a single shingle value.
    ///
    /// Returns a vector of hash values, one per hash function.
    /// For MinHash, we take the minimum across all shingles.
    #[allow(dead_code)]
    pub fn hash_shingle(&self, shingle: u64) -> Vec<u32> {
        let mut result = Vec::with_capacity(self.num_hashes);

        for i in 0..self.num_hashes {
            let hash = self.hash_single(shingle, i);
            result.push(hash);
        }

        result
    }

    /// Update a signature in-place with a new shingle using MINimum update.
    ///
    /// For each hash function, updates `signature[i] = min(signature[i], hash_i(shingle))`.
    pub fn update_signature(&self, signature: &mut [u32], shingle: u64) {
        const CHUNK_SIZE: usize = 8;

        // Bound the iteration to the shorter of signature and num_hashes
        // to prevent OOB if caller passes a shorter slice.
        let limit = signature.len().min(self.num_hashes);

        // Process in chunks for better cache locality
        for chunk_start in (0..limit).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(limit);

            for i in chunk_start..chunk_end {
                let hash = self.hash_single(shingle, i);
                if hash < signature[i] {
                    signature[i] = hash;
                }
            }
        }
    }

    /// Compute hash for a single function index.
    #[inline]
    fn hash_single(&self, shingle: u64, idx: usize) -> u32 {
        // h(x) = (a * x + b) mod p
        // Use 128-bit arithmetic to avoid overflow, then reduce mod p
        let a = self.coeffs_a[idx];
        let b = self.coeffs_b[idx];

        let product = u128::from(a).wrapping_mul(u128::from(shingle));
        let sum = product.wrapping_add(u128::from(b));

        // Reduce modulo Mersenne prime: x mod (2^61 - 1)
        let reduced = mod_mersenne(sum);

        // Convert to u32 for the signature
        reduced as u32
    }

    /// Batch hash multiple shingles and update signature.
    ///
    /// This is more cache-efficient than calling `update_signature` repeatedly.
    #[allow(dead_code)]
    pub fn update_signature_batch(&self, signature: &mut [u32], shingles: &[u64]) {
        for shingle in shingles {
            self.update_signature(signature, *shingle);
        }
    }

    /// Get the number of hash functions.
    #[must_use]
    pub const fn num_hashes(&self) -> usize {
        self.num_hashes
    }
}

/// Reduce a 128-bit value modulo the Mersenne prime 2^61 - 1.
/// Uses iterative reduction to ensure correctness for arbitrary 128-bit inputs.
#[inline]
fn mod_mersenne(mut x: u128) -> u64 {
    // For p = 2^61 - 1 we can reduce by folding the high bits into the low
    // bits: x -> (x_low + (x >> 61)). For large 128-bit values this may
    // need to be repeated until the value fits in 61 bits.
    const MASK: u128 = (1_u128 << 61) - 1;
    let p = u128::from(MERSENNE_PRIME);

    // Iteratively fold high bits until no bits remain above 61
    while (x >> 61) != 0 {
        let low = x & MASK;
        let high = x >> 61;
        x = low + high;
    }

    // One final correction
    if x >= p {
        (x - p) as u64
    } else {
        x as u64
    }
}

/// SplitMix64 pseudo-random number generator.
#[inline]
const fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Compute a fast, non-cryptographic hash of a byte slice.
///
/// Uses the 64-bit variant of MurmurHash3 for good distribution.
#[must_use]
pub fn hash_bytes(data: &[u8]) -> u64 {
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;
    const SEED: u64 = 0x9e37_79b9_7f4a_7c15;

    let mut h = SEED;

    // Process 8 bytes at a time
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let mut k = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        k = k.wrapping_mul(C1);
        k = k.rotate_left(31);
        k = k.wrapping_mul(C2);

        h ^= k;
        h = h.rotate_left(27);
        h = h.wrapping_mul(5).wrapping_add(0x52ce_adbe_e7ef_7e45);
    }

    // Process remaining bytes
    if !remainder.is_empty() {
        let mut k = 0_u64;
        for (i, &b) in remainder.iter().enumerate() {
            k ^= u64::from(b) << (i * 8);
        }
        k = k.wrapping_mul(C1);
        k = k.rotate_left(31);
        k = k.wrapping_mul(C2);
        h ^= k;
    }

    // Finalization
    h ^= data.len() as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;

    h
}

/// Compute hash of a string slice.
#[must_use]
#[allow(dead_code)]
pub fn hash_str(s: &str) -> u64 {
    hash_bytes(s.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hasher_creates_correct_size() {
        let hasher = FastHasher::new(128, 42);
        assert_eq!(hasher.num_hashes(), 128);
        assert_eq!(hasher.coeffs_a.len(), 128);
        assert_eq!(hasher.coeffs_b.len(), 128);
    }

    /// Regression: `FastHasher::new` allocated its coefficient vectors with
    /// the caller-supplied count, so `FastHasher::new(usize::MAX, seed)` died
    /// on a capacity-overflow panic / allocation abort. The count is now
    /// clamped to `MAX_SIGNATURE_SIZE`, keeping construction total for any
    /// input while leaving realistic counts untouched.
    #[test]
    fn new_clamps_hostile_num_hashes() {
        let hasher = FastHasher::new(usize::MAX, 42);
        assert_eq!(hasher.num_hashes(), crate::config::MAX_SIGNATURE_SIZE);
        assert_eq!(hasher.coeffs_a.len(), crate::config::MAX_SIGNATURE_SIZE);
    }

    #[test]
    fn hash_single_deterministic() {
        let hasher = FastHasher::new(64, 12345);
        let h1 = hasher.hash_single(0xdead_beef, 0);
        let h2 = hasher.hash_single(0xdead_beef, 0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_shingles_different_hashes() {
        let hasher = FastHasher::new(64, 42);
        let sig1 = hasher.hash_shingle(1);
        let sig2 = hasher.hash_shingle(2);
        
        // Very unlikely to be identical across all hash functions
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn update_signature_works() {
        let hasher = FastHasher::new(64, 42);
        let mut sig = vec![u32::MAX; 64];
        
        hasher.update_signature(&mut sig, 12345);
        
        // Signature should have been updated (minimized)
        assert!(sig.iter().any(|&x| x < u32::MAX));
    }

    #[test]
    fn signature_minimum_property() {
        let hasher = FastHasher::new(32, 42);
        let mut sig = vec![u32::MAX; 32];
        
        // First shingle sets initial values
        hasher.update_signature(&mut sig, 100);
        let first_sig = sig.clone();
        
        // Second shingle can only decrease values
        hasher.update_signature(&mut sig, 200);
        
        for i in 0..32 {
            assert!(sig[i] <= first_sig[i]);
        }
    }

    #[test]
    fn hash_bytes_deterministic() {
        let data = b"hello world";
        let h1 = hash_bytes(data);
        let h2 = hash_bytes(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_bytes_different_input_different_output() {
        let h1 = hash_bytes(b"hello");
        let h2 = hash_bytes(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_str_same_as_bytes() {
        let s = "hello world";
        assert_eq!(hash_str(s), hash_bytes(s.as_bytes()));
    }

    #[test]
    fn splitmix64_produces_varied_output() {
        let s1 = splitmix64(1);
        let s2 = splitmix64(2);
        assert_ne!(s1, s2);
    }

    #[test]
    fn mod_mersenne_reduction() {
        // Test that mod_mersenne works correctly
        let p = u128::from(MERSENNE_PRIME);
        
        // x mod p should be x for x < p
        assert_eq!(mod_mersenne(12345), 12345);
        
        // p mod p should be 0
        assert_eq!(mod_mersenne(p), 0);
        
        // (p + 1) mod p should be 1
        assert_eq!(mod_mersenne(p + 1), 1);
    }

    #[test]
    fn batch_update_matches_individual() {
        let hasher = FastHasher::new(32, 42);
        let shingles = vec![1_u64, 2, 3, 4, 5];
        
        let mut sig_batch = vec![u32::MAX; 32];
        hasher.update_signature_batch(&mut sig_batch, &shingles);
        
        let mut sig_individual = vec![u32::MAX; 32];
        for shingle in &shingles {
            hasher.update_signature(&mut sig_individual, *shingle);
        }
        
        assert_eq!(sig_batch, sig_individual);
    }

    #[test]
    fn hash_distribution_uniform() {
        // Test that hash values are reasonably distributed
        let hasher = FastHasher::new(64, 42);
        let mut bins = [0_u32; 16];
        
        for i in 0..10000 {
            let hash = hasher.hash_single(i, 0);
            let bin = (hash >> 28) as usize % 16; // Use high bits
            bins[bin] += 1;
        }
        
        // Each bin should have roughly 10000/16 = 625 items
        // Allow 50% variance for statistical fluctuation
        let expected = 10000 / 16;
        for count in &bins {
            assert!(
                *count >= expected / 2 && *count <= expected * 3 / 2,
                "bin count {} is outside expected range around {}",
                count,
                expected
            );
        }
    }
}
