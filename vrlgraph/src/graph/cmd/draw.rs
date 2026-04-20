use ash::vk;

use super::Cmd;
use crate::resource::Buffer;

impl<'a> Cmd<'a> {
    /// Binds a vertex buffer to slot 0.
    pub fn bind_vertex_buffer(&self, handle: Buffer, offset: vk::DeviceSize) {
        let raw_buf = self.buffer(handle).raw;
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(self.raw, 0, &[raw_buf], &[offset])
        };
    }

    /// Binds an index buffer. Indices are expected to be `u32`.
    pub fn bind_index_buffer(&self, handle: Buffer, offset: vk::DeviceSize) {
        let raw_buf = self.buffer(handle).raw;
        unsafe {
            self.device
                .cmd_bind_index_buffer(self.raw, raw_buf, offset, vk::IndexType::UINT32)
        };
    }

    /// Draws `vertices` vertices and `instances` instances, starting from index 0.
    pub fn draw(&self, vertices: u32, instances: u32) {
        unsafe { self.device.cmd_draw(self.raw, vertices, instances, 0, 0) };
    }

    /// Draws using an index buffer. `first_index` is the byte offset into the
    /// index buffer divided by the index size. `vertex_offset` is added to each
    /// index value before fetching a vertex.
    pub fn draw_indexed(&self, indices: u32, instances: u32, first_index: u32, vertex_offset: i32) {
        unsafe {
            self.device.cmd_draw_indexed(
                self.raw,
                indices,
                instances,
                first_index,
                vertex_offset,
                0,
            )
        };
    }

    pub fn draw_indirect(
        &self,
        handle: Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        let raw_buf = self.buffer(handle).raw;
        unsafe {
            self.device
                .cmd_draw_indirect(self.raw, raw_buf, offset, draw_count, stride)
        };
    }

    pub fn draw_indexed_indirect(
        &self,
        handle: Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        let raw_buf = self.buffer(handle).raw;
        unsafe {
            self.device
                .cmd_draw_indexed_indirect(self.raw, raw_buf, offset, draw_count, stride)
        };
    }

    /// Dispatches a compute workload of `x * y * z` workgroups.
    /// A compute pipeline must be bound before calling this.
    pub fn dispatch(&self, x: u32, y: u32, z: u32) {
        debug_assert!(
            self.bound_bind_point == vk::PipelineBindPoint::COMPUTE,
            "Cmd: bind_compute_pipeline() must be called before dispatch()"
        );
        unsafe { self.device.cmd_dispatch(self.raw, x, y, z) };
    }

    /// Dispatches a compute workload using arguments read from a buffer at `offset`.
    /// The buffer must contain a `VkDispatchIndirectCommand`.
    pub fn dispatch_indirect(&self, handle: Buffer, offset: vk::DeviceSize) {
        debug_assert!(
            self.bound_bind_point == vk::PipelineBindPoint::COMPUTE,
            "Cmd: bind_compute_pipeline() must be called before dispatch_indirect()"
        );
        let raw_buf = self.buffer(handle).raw;
        unsafe { self.device.cmd_dispatch_indirect(self.raw, raw_buf, offset) };
    }
}
