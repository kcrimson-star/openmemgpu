use openmemgpu_core::memory::DomainId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBlock {
    pub id: BlockId,
    pub domain_id: DomainId,
    pub offset_bytes: u64,
    pub size_bytes: u64,
    pub is_bound: bool,
}

#[derive(Debug, Default)]
pub struct MemoryPool {
    domain_id: DomainId,
    capacity_bytes: u64,
    allocated_bytes: u64,
    next_block_id: u64,
}

impl MemoryPool {
    pub fn new(domain_id: DomainId, capacity_bytes: u64) -> Self {
        Self {
            domain_id,
            capacity_bytes,
            allocated_bytes: 0,
            next_block_id: 1,
        }
    }

    pub fn allocate(&mut self, size_bytes: u64) -> Option<MemoryBlock> {
        if self.allocated_bytes + size_bytes > self.capacity_bytes {
            return None;
        }

        let block = MemoryBlock {
            id: BlockId(self.next_block_id),
            domain_id: self.domain_id,
            offset_bytes: self.allocated_bytes,
            size_bytes,
            is_bound: false,
        };

        self.next_block_id += 1;
        self.allocated_bytes += size_bytes;
        Some(block)
    }

    pub fn release(&mut self, block: &MemoryBlock) {
        if block.domain_id == self.domain_id {
            self.allocated_bytes = self.allocated_bytes.saturating_sub(block.size_bytes);
        }
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }
}
