#![doc = include_str!("../README.md")]

pub mod device;
pub mod graph;
pub mod resource;

#[doc(hidden)]
pub use ash::vk;

pub mod prelude {

    pub use crate::graph::{
        Access, Array2D, BindlessIndex, BufferUsage, Cmd, ComputePipelineBuilder,
        Cubemap, Frame, FrameResources, GpuPreference, Graph, GraphBuilder, GraphError, Image,
        LoadOp, PassSetup, PassTiming, PipelineBuilder, PresentMode, ReadParam, Sampled, Sampler,
        Storage, StreamingBufferHandle, WithLayer, WithLayerLoadOp, WithLoadOp, WriteParam,
    };

    pub use crate::resource::{
        Buffer, BufferDesc, ImageDesc, ImageHandle, ImageKind, Pipeline, ResourceError,
    };

    pub use crate::device::DeviceError;

    pub use ash::vk;
}
