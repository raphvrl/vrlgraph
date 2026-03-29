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
        Access, BufferUsage, Frame, Graph, GraphError, Image, LoadOp, PresentMode, Sampler,
        StreamingBufferHandle,
    };

    pub use crate::resource::{Buffer, BufferDesc, ImageDesc, ImageKind, Pipeline};

    pub use vrlgraph_derive::VertexInput;

    pub use ash::vk;
}
