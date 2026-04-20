#![doc = include_str!("../README.md")]

pub mod device;
pub mod graph;
pub mod resource;
pub mod shader;
pub mod types;

pub use ash;
pub use bytemuck;

pub use shader::{DynShaderType, ShaderType, VertexAttribute, VertexInput, round_up};
pub use vrlgraph_derive::{ShaderType, VertexInput};

pub mod gpu {
    pub use crate::graph::{
        Access, Array2D, BindlessIndex, BufferUsage, Cmd, ComputePipelineBuilder, Cubemap,
        FrameBuilder, GpuBufferBuilder, GpuPreference, Graph, GraphError, HostBufferBuilder, Image,
        LoadOp, PassTiming, PresentMode, Sampled, Sampler, SamplerBuilder, Storage,
        StreamingBuffer, TextureBuilder, WithClearColor, WithLayer, WithLayerClearColor,
        WithLayerLoadOp, WithLoadOp,
    };
    pub use crate::resource::{AsyncBuffer, Buffer, BufferDesc, ImageKind, Pipeline, ShaderModule};
    pub use crate::shader::DynShaderType;
    pub use crate::types::*;
}

pub mod prelude {
    pub use crate::gpu;
    pub use ash::vk;
    pub use vrlgraph_derive::{ShaderType, VertexInput};
}
