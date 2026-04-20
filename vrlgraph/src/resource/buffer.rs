use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use super::ResourceError;

/// Description of a GPU buffer.
///
/// Passed to [`Graph::create_buffer`](crate::graph::Graph::create_buffer).
pub struct BufferDesc {
    /// Size in bytes.
    pub size: vk::DeviceSize,
    /// Vulkan buffer usage flags (e.g. `VERTEX_BUFFER`, `UNIFORM_BUFFER`).
    pub usage: vk::BufferUsageFlags,
    /// Memory location. Use `CpuToGpu` for buffers written from the CPU each
    /// frame (uniforms, staging). Use `GpuOnly` for static geometry that is
    /// uploaded once.
    pub location: MemoryLocation,
    /// Debug label.
    pub label: String,
}

/// A GPU-resident buffer.
///
/// Returned by [`Cmd::buffer`](crate::graph::Cmd::buffer) and
/// [`Cmd::streaming_buffer`](crate::graph::Cmd::streaming_buffer). Pass [`gpu::Buffer`](crate::gpu::Buffer) handles
/// directly to [`Cmd::bind_vertex_buffer`](crate::graph::Cmd::bind_vertex_buffer),
/// [`Cmd::bind_index_buffer`](crate::graph::Cmd::bind_index_buffer), and the indirect draw/dispatch
/// methods. Use `device_address` to pass the buffer as a raw 64-bit GPU pointer via push constants
/// (requires [`ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS`] at creation time).
pub struct GpuBuffer {
    /// The underlying `VkBuffer`.
    pub raw: vk::Buffer,
    /// Allocated size in bytes.
    pub size: vk::DeviceSize,
    /// Vulkan buffer usage flags.
    pub usage: vk::BufferUsageFlags,
    /// GPU virtual address of this buffer for bindless access via push constants.
    pub device_address: vk::DeviceAddress,
    allocation: Allocation,
}

impl GpuBuffer {
    pub(super) fn create(
        device: &ash::Device,
        allocator: &mut Allocator,
        desc: &BufferDesc,
    ) -> Result<Self, ResourceError> {
        let create_info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let raw = unsafe { device.create_buffer(&create_info, None)? };

        let requirements = unsafe { device.get_buffer_memory_requirements(raw) };

        let allocation = match allocator.allocate(&AllocationCreateDesc {
            name: &desc.label,
            requirements,
            location: desc.location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => {
                unsafe { device.destroy_buffer(raw, None) };
                return Err(ResourceError::Allocation(e));
            }
        };

        if let Err(e) =
            unsafe { device.bind_buffer_memory(raw, allocation.memory(), allocation.offset()) }
        {
            allocator.free(allocation).ok();
            unsafe { device.destroy_buffer(raw, None) };
            return Err(ResourceError::Vulkan(e));
        }

        let device_address = if desc
            .usage
            .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        {
            unsafe {
                device
                    .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(raw))
            }
        } else {
            0
        };

        Ok(Self {
            raw,
            size: desc.size,
            usage: desc.usage,
            device_address,
            allocation,
        })
    }

    pub(super) fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe { device.destroy_buffer(self.raw, None) };
        if let Err(e) = allocator.free(self.allocation) {
            tracing::error!("failed to free buffer allocation: {e}");
        }
    }

    /// Returns a pointer to the mapped CPU memory, or `None` if the buffer is
    /// not host-visible.
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        self.allocation.mapped_ptr().map(|p| p.as_ptr() as *mut u8)
    }

    /// Writes a [`ShaderType`](crate::ShaderType) or
    /// [`DynShaderType`](crate::DynShaderType) value (e.g. a struct with a
    /// `Vec<T>` tail field) with automatic scalar-layout padding.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not host-visible.
    pub fn write<T: crate::DynShaderType>(&self, value: &T) {
        let size = value.padded_size();
        if size <= 256 {
            let mut buf = [0u8; 256];
            value.write_padded_dyn(&mut buf[..size]);
            self.write_bytes(&buf[..size]);
        } else {
            let mut buf = vec![0u8; size];
            value.write_padded_dyn(&mut buf);
            self.write_bytes(&buf);
        }
    }

    /// Copies a slice of [`Pod`](bytemuck::Pod) values into the buffer via the
    /// CPU-mapped pointer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not host-visible or if `data` is larger than
    /// the allocated size.
    pub fn write_slice<T: bytemuck::Pod>(&self, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);
        assert!(
            bytes.len() <= self.size as usize,
            "GpuBuffer::write_slice: data ({} B) exceeds buffer size ({} B)",
            bytes.len(),
            self.size,
        );
        let ptr = self
            .mapped_ptr()
            .expect("GpuBuffer::write_slice: buffer is not host-visible");
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    }

    /// Writes raw bytes into the buffer via the CPU-mapped pointer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not host-visible or if `bytes` is larger than
    /// the allocated size.
    pub fn write_bytes(&self, bytes: &[u8]) {
        assert!(
            bytes.len() <= self.size as usize,
            "GpuBuffer::write_bytes: data ({} B) exceeds buffer size ({} B)",
            bytes.len(),
            self.size,
        );
        let ptr = self
            .mapped_ptr()
            .expect("GpuBuffer::write_bytes: buffer is not host-visible");
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    }
}
