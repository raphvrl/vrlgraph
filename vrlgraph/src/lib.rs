#![doc = include_str!("../README.md")]

pub mod device;
pub mod graph;
pub mod resource;
pub mod shader;
pub mod types;
pub mod vertex;

pub use ash;

pub use shader::{ShaderType, round_up};
pub use vertex::{VertexAttribute, VertexInput};
pub use vrlgraph_derive::{ShaderType, VertexInput};

pub mod prelude {

    pub use crate::graph::{
        Access, Array2D, BindlessIndex, BufferUsage, Cmd, ComputePipelineBuilder, Cubemap, Frame,
        FrameResources, GpuPreference, Graph, GraphError, Image, LoadOp, PassTiming, PresentMode,
        Sampled, Sampler, SamplerBuilder, Storage, StreamingBufferHandle, WithClearColor,
        WithLayer, WithLayerClearColor, WithLayerLoadOp, WithLoadOp,
    };

    pub use crate::resource::{Buffer, BufferDesc, ImageKind, Pipeline, ShaderModule};

    pub use crate::types::*;

    pub use ash::vk;
    pub use vrlgraph_derive::{ShaderType, VertexInput};
}
