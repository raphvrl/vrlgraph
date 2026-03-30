//! GPU resource management — buffers, images, pipelines, and samplers.
//!
//! All resources are referenced through opaque handles ([`Buffer`],
//! [`ImageHandle`], [`Pipeline`], [`SamplerHandle`], [`ShaderModule`]).
//! The underlying objects are owned by the [`Graph`](crate::graph::Graph)
//! and freed when you call the corresponding `destroy_*` method or when
//! the graph is dropped.

mod buffer;
mod handle;
mod image;
mod pipeline;
mod pool;
mod shader;
mod streaming;

use ash::vk;
use thiserror::Error;

/// Errors returned by resource allocation operations.
#[derive(Debug, Error)]
pub enum ResourceError {
    /// A Vulkan call failed during buffer or image creation.
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
    /// GPU memory allocation failed.
    #[error("Allocation error: {0}")]
    Allocation(#[from] gpu_allocator::AllocationError),
}

pub use buffer::{BufferDesc, GpuBuffer};
#[cfg(debug_assertions)]
pub(crate) use handle::ShaderModuleHandle;
pub use handle::{Buffer, ImageHandle, Pipeline, SamplerHandle, ShaderModule};
pub(crate) use handle::{BufferHandle, PipelineHandle};
pub use image::{GpuImage, ImageKind};
pub(crate) use image::ImageDesc;
pub use pipeline::GpuPipeline;
pub(crate) use pool::ResourcePool;
pub(crate) use shader::GpuShaderModule;
pub use streaming::StreamingBufferHandle;
