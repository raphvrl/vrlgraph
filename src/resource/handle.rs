use slotmap::new_key_type;

new_key_type! {
    pub struct BufferHandle;

    pub struct ImageHandle;

    pub struct PipelineHandle;

    pub struct SamplerHandle;
}
