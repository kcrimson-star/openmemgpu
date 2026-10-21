pub mod capacity;
pub mod expansion;
pub mod manager;
pub mod policy;
pub mod residency;

pub use capacity::VvramCapacityReport;
pub use expansion::{DynamicVramExpander, OversizedWorkloadSlices, VramExpansionStatus};
pub use manager::{VvramBuffer, VvramBufferDescriptor, VvramManager};
pub use policy::PlacementPolicy;
pub use residency::ResidencyState;
