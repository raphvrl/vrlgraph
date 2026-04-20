use slotmap::new_key_type;
use smallvec::SmallVec;

use super::BufferHandle;

new_key_type! {
    pub struct StreamingBuffer;
}

pub(super) struct StreamingSlots {
    pub(super) slots: SmallVec<[BufferHandle; 3]>,
}

impl StreamingSlots {
    pub(super) fn new(slots: SmallVec<[BufferHandle; 3]>) -> Self {
        Self { slots }
    }

    #[inline]
    pub(super) fn slot(&self, frame_index: usize) -> BufferHandle {
        debug_assert!(!self.slots.is_empty(), "StreamingSlots has no slots");
        self.slots[frame_index % self.slots.len()]
    }
}
