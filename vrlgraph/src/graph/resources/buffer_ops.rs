use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::graph::transfer::TransferId;
use crate::graph::{Graph, GraphError};
use crate::resource::{AsyncBuffer, Buffer, BufferDesc, GpuBuffer, ResourceError, StreamingBuffer};

impl Graph {
    pub fn create_buffer(&mut self, desc: &BufferDesc) -> Result<Buffer, ResourceError> {
        let device = self.device.ash_device().clone();
        let handle = self
            .resources
            .create_buffer(&device, self.device.allocator_mut(), desc)?;
        let address = self
            .resources
            .get_buffer(handle)
            .expect("buffer just created")
            .device_address;
        Ok(Buffer(handle, address))
    }

    pub fn destroy_buffer(&mut self, handle: Buffer) {
        let device = self.device.ash_device().clone();
        self.resources
            .destroy_buffer(&device, self.device.allocator_mut(), handle.0);
        self.buffer_states.remove(&handle.0);
    }

    pub fn destroy_async_buffer(&mut self, handle: AsyncBuffer) {
        self.transfer.wait_for_buffer(handle.0);
        let device = self.device.ash_device().clone();
        self.resources
            .destroy_buffer(&device, self.device.allocator_mut(), handle.0);
        self.buffer_states.remove(&handle.0);
    }

    pub fn get_buffer(&self, handle: Buffer) -> Option<&GpuBuffer> {
        self.resources.get_buffer(handle.0)
    }

    pub fn create_streaming_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        label: &str,
    ) -> Result<StreamingBuffer, ResourceError> {
        let frames = self.frames.len();
        let device = self.device.ash_device().clone();
        self.resources.create_streaming_buffer(
            &device,
            self.device.allocator_mut(),
            size,
            usage,
            location,
            label,
            frames,
        )
    }

    pub fn destroy_streaming_buffer(&mut self, handle: StreamingBuffer) {
        let device = self.device.ash_device().clone();
        let frames = self.frames.len();
        for i in 0..frames {
            if let Some(slot) = self.resources.streaming_slot(handle, i) {
                self.buffer_states.remove(&slot);
            }
        }
        self.resources
            .destroy_streaming_buffer(&device, self.device.allocator_mut(), handle);
    }

    /// Writes a [`ShaderType`](crate::ShaderType) or
    /// [`DynShaderType`](crate::DynShaderType) value (e.g. a struct with a
    /// `Vec<T>` tail field) into a buffer with automatic scalar-layout padding.
    pub fn write_buffer<T: crate::DynShaderType>(&self, handle: Buffer, value: &T) {
        self.resources
            .get_buffer(handle.0)
            .expect("write_buffer: invalid buffer handle")
            .write(value);
    }

    /// Writes a slice of [`Pod`](bytemuck::Pod) values into a buffer.
    pub fn write_buffer_slice<T: bytemuck::Pod>(&self, handle: Buffer, data: &[T]) {
        self.resources
            .get_buffer(handle.0)
            .expect("write_buffer_slice: invalid buffer handle")
            .write_slice(data);
    }

    pub(in crate::graph) fn host_buffer(
        &mut self,
        label: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer, GraphError> {
        Ok(self.create_buffer(&BufferDesc {
            size,
            usage: usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            location: MemoryLocation::CpuToGpu,
            label: label.to_string(),
        })?)
    }

    /// Uploads data to a GPU-only buffer asynchronously via the transfer queue.
    /// The buffer is not usable until the returned [`TransferId`] completes.
    pub fn upload_buffer_async(
        &mut self,
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
        label: &str,
    ) -> Result<(Buffer, TransferId), GraphError> {
        let size = bytes.len() as vk::DeviceSize;
        let device = self.device.ash_device().clone();

        let staging = self.create_staging(bytes, &format!("{label}_staging"))?;

        let dst = self.resources.create_buffer(
            &device,
            self.device.allocator_mut(),
            &BufferDesc {
                size,
                usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
                location: MemoryLocation::GpuOnly,
                label: label.to_string(),
            },
        )?;

        let src_raw = self
            .resources
            .get_buffer(staging)
            .expect("buffer just created")
            .raw;
        let dst_buf = self.resources.get_buffer(dst).expect("buffer just created");
        let dst_raw = dst_buf.raw;
        let dst_addr = dst_buf.device_address;

        let id = self
            .transfer
            .enqueue_buffer(src_raw, dst_raw, size, staging);
        self.transfer.submit_pending()?;

        Ok((Buffer(dst, dst_addr), id))
    }
}
