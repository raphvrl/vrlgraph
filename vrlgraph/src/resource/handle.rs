use ash::vk;
use slotmap::new_key_type;

new_key_type! {
    pub(crate) struct BufferHandle;

    pub struct ImageHandle;

    pub(crate) struct PipelineHandle;

    pub struct SamplerHandle;

    pub(crate) struct ShaderModuleHandle;
}

/// An opaque handle to a GPU buffer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Buffer(pub(crate) BufferHandle, pub(crate) vk::DeviceAddress);

impl Buffer {
    /// GPU virtual address of this buffer for bindless access via push constants.
    /// Returns `0` if the buffer was not created with
    /// [`vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS`].
    pub fn address(&self) -> vk::DeviceAddress {
        self.1
    }
}

/// An opaque handle to a GPU buffer whose data is being uploaded asynchronously.
/// Use [`Cmd::try_buffer`](crate::graph::Cmd::try_buffer)
/// to access the underlying buffer — it returns `None` while the transfer is
/// still in progress.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AsyncBuffer(pub(crate) BufferHandle, pub(crate) vk::DeviceAddress);

impl AsyncBuffer {
    /// GPU virtual address of this buffer for bindless access via push constants.
    /// The address is valid immediately — independently of the in-flight transfer.
    pub fn address(&self) -> vk::DeviceAddress {
        self.1
    }
}

/// An opaque handle to a GPU pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Pipeline(pub(crate) PipelineHandle);

/// An opaque handle to a GPU shader module.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ShaderModule(pub(crate) ShaderModuleHandle);
