use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Out of memory on domain {domain_id:?}: requested {requested_bytes} bytes, budget remaining {available_bytes} bytes")]
    OutOfMemory {
        domain_id: String,
        requested_bytes: u64,
        available_bytes: u64,
    },

    #[error("Backend initialization failed: {0}")]
    BackendInitFailed(String),

    #[error("Unsupported capability: {0}")]
    UnsupportedCapability(String),

    #[error("Invalid resource state transition: {0}")]
    InvalidStateTransition(String),

    #[error("Transfer synchronization error: {0}")]
    SyncError(String),

    #[error("I/O or driver error: {0}")]
    DriverError(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}
