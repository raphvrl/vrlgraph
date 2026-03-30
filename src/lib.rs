#![doc = include_str!("../README.md")]

pub mod device;
pub mod graph;
pub mod resource;
pub mod types;
pub mod vertex;

pub use ash;

pub use vertex::{VertexAttribute, VertexInput};
pub use vrlgraph_derive::VertexInput;

pub mod prelude {

    pub use crate::graph::{
        Access, Array2D, BindlessIndex, BufferUsage, Cubemap, Frame, GpuPreference, Graph,
        GraphError, Image, LoadOp, PassTiming, PresentMode, Sampled, Sampler, SamplerBuilder,
        Storage, StreamingBufferHandle, WithLayer, WithLayerLoadOp, WithLoadOp,
    };

    pub use crate::resource::{Buffer, BufferDesc, ImageKind, Pipeline, ShaderModule};

    pub use crate::types::*;

    pub use vrlgraph_derive::VertexInput;
}
