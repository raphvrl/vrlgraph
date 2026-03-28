#![doc = include_str!("../README.md")]

pub mod device;
pub mod graph;
pub mod resource;
pub mod vertex;

#[doc(hidden)]
pub use ash::vk;

pub use vertex::{VertexAttribute, VertexInput};
pub use vrlgraph_derive::VertexInput;

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

    pub use crate::vertex::{VertexAttribute, VertexInput};
    pub use vrlgraph_derive::VertexInput;

    pub use ash::vk;
}
