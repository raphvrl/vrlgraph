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
pub struct Buffer(pub(crate) BufferHandle);

/// An opaque handle to a GPU buffer whose data is being uploaded asynchronously.
/// Use [`FrameResources::try_buffer`](crate::graph::FrameResources::try_buffer)
/// to access the underlying buffer — it returns `None` while the transfer is
/// still in progress.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AsyncBuffer(pub(crate) BufferHandle);

/// An opaque handle to a GPU pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Pipeline(pub(crate) PipelineHandle);

/// An opaque handle to a GPU shader module.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ShaderModule(pub(crate) ShaderModuleHandle);
