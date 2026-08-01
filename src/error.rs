//! Error types for the dedup crate.
//!
//! Every error is actionable and tells you how to fix the problem.

/// All errors that can occur during deduplication.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Configuration was invalid.
    #[error("invalid configuration: {reason}. Fix: {fix}")]
    InvalidConfig {
        /// What was wrong.
        reason: String,
        /// How to fix it.
        fix: String,
    },

    /// A document was too large to process.
    #[error("document too large: size={size}, max={max}. Fix: increase max_document_size in Config or filter large documents before deduplication.")]
    DocumentTooLarge {
        /// Actual document size.
        size: usize,
        /// Maximum allowed size.
        max: usize,
    },

    /// A document was empty and cannot be processed.
    #[error("empty document at index {index}. Fix: filter empty documents before deduplication.")]
    EmptyDocument {
        /// Index of the empty document.
        index: usize,
    },

    /// Internal error during hashing.
    #[error("hashing failed: {reason}. Fix: check for integer overflow or memory exhaustion.")]
    HashingFailed {
        /// What went wrong.
        reason: String,
    },

    /// Memory limit exceeded.
    #[error("memory limit exceeded: {usage_bytes} bytes. Fix: increase memory_limit_in_mb in Config or reduce num_bands.")]
    MemoryLimitExceeded {
        /// Current memory usage in bytes.
        usage_bytes: u64,
    },

    /// IO error during processing.
    #[error("io error: {0}. Fix: check file permissions and disk space.")]
    Io(#[from] std::io::Error),
}

/// Convenience result type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_includes_fix() {
        let err = Error::InvalidConfig {
            reason: "num_bands must divide signature_size".to_string(),
            fix: "use a signature_size divisible by num_bands".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Fix:"));
        assert!(msg.contains("signature_size"));
    }

    #[test]
    fn document_too_large_error_includes_size() {
        let err = Error::DocumentTooLarge {
            size: 1000000,
            max: 100000,
        };
        let msg = err.to_string();
        assert!(msg.contains("1000000"));
        assert!(msg.contains("100000"));
    }
}
