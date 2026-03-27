#![doc = include_str!("../README.md")]

pub mod device;
pub mod graph;
pub mod resource;

#[doc(hidden)]
pub use ash::vk;

pub mod prelude {

    pub use crate::graph::{
        Access, BufferUsage, Cmd, ComputePipelineBuilder, DescriptorSetBuilder, DescriptorWrite,
        DynamicDescriptorSet, Frame, FrameResources, GpuPreference, Graph, GraphBuilder,
        GraphError, GraphImage, LoadOp, PassSetup, PassTiming, PipelineBuilder, PresentMode,
        PushDescriptor, ReadParam, StreamingBufferHandle, WithLayer, WithLayerLoadOp, WithLoadOp,
        WriteParam,
    };

    pub use crate::resource::{
        BufferDesc, BufferHandle, ImageDesc, ImageHandle, ImageKind, PipelineHandle, ResourceError,
        SamplerHandle,
    };

    pub use crate::device::DeviceError;

    pub use ash::vk;
}
