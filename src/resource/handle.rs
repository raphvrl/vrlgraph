use slotmap::new_key_type;

new_key_type! {
    pub(crate) struct BufferHandle;

    pub struct ImageHandle;

    pub(crate) struct PipelineHandle;

    pub struct SamplerHandle;
}

/// An opaque handle to a GPU buffer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Buffer(pub(crate) BufferHandle);

/// An opaque handle to a GPU pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Pipeline(pub(crate) PipelineHandle);
