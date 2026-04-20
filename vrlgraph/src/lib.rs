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
    pub use crate::graph::{Image, Sampler, StreamingBuffer};
    pub use crate::resource::{AsyncBuffer, Buffer, BufferDesc, ImageKind, Pipeline, ShaderModule};
}

pub mod prelude {

    pub use crate::graph::{
        Access, Array2D, BindlessIndex, BufferUsage, Cmd, ComputePipelineBuilder, Cubemap,
        FrameBuilder, GpuBufferBuilder, GpuPreference, Graph, GraphError, HostBufferBuilder,
        LoadOp, PassTiming, PresentMode, Sampled, SamplerBuilder, Storage, TextureBuilder,
        WithClearColor, WithLayer, WithLayerClearColor, WithLayerLoadOp, WithLoadOp,
    };

    pub use crate::gpu::*;

    pub use crate::types::*;

    pub use crate::shader::DynShaderType;

    pub use ash::vk;
    pub use vrlgraph_derive::{ShaderType, VertexInput};
}
