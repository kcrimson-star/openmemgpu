use openmemgpu_core::error::CoreError;
use parking_lot::Mutex;

pub struct StagingBufferManager {
    pool_size_bytes: usize,
    in_flight_bytes: Mutex<usize>,
}

impl StagingBufferManager {
    pub fn new(pool_size_bytes: usize) -> Self {
        Self {
            pool_size_bytes,
            in_flight_bytes: Mutex::new(0),
        }
    }

    pub fn acquire_staging_slot(&self, bytes: usize) -> Result<(), CoreError> {
        let mut in_flight = self.in_flight_bytes.lock();
        if *in_flight + bytes > self.pool_size_bytes {
            return Err(CoreError::OutOfMemory {
                domain_id: "Host Staging Pool".into(),
                requested_bytes: bytes as u64,
                available_bytes: (self.pool_size_bytes - *in_flight) as u64,
            });
        }
        *in_flight += bytes;
        Ok(())
    }

    pub fn release_staging_slot(&self, bytes: usize) {
        let mut in_flight = self.in_flight_bytes.lock();
        *in_flight = in_flight.saturating_sub(bytes);
    }
}
