use ash::vk;
use thiserror::Error;

use crate::device::DeviceError;
use crate::resource::ResourceError;

use super::cmd::CommandError;
use super::sync::SyncError;

/// Errors returned by graph operations.
#[derive(Debug, Error)]
pub enum GraphError {
    /// A Vulkan device-level error during initialization or swapchain setup.
    #[error("Device error: {0}")]
    Device(#[from] DeviceError),
    /// Frame synchronization failed (semaphore or fence error).
    #[error("Sync error: {0}")]
    Sync(#[from] SyncError),
    /// Command buffer recording or submission failed.
    #[error("Command error: {0}")]
    Command(#[from] CommandError),
    /// GPU resource allocation failed.
    #[error("Resource error: {0}")]
    Resource(#[from] ResourceError),
    /// A raw Vulkan call returned an error code.
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
    /// A SPIR-V file could not be read or is malformed.
    #[error("Shader load error: {0}")]
    ShaderLoad(String),
    /// The window handle provided to the builder is no longer valid.
    #[error("Window handle unavailable")]
    WindowHandle,
    /// The swapchain is out of date and must be recreated. Call [`crate::graph::Graph::resize`]
    /// and skip the current frame. This is expected after a window resize.
    #[error("Swapchain out of date")]
    SwapchainOutOfDate,
    /// A dependency cycle was detected in the declared passes. The named pass
    /// is part of the cycle.
    #[error("Render pass cycle detected involving pass '{0}'")]
    PassCycle(&'static str),
    /// Image readback was requested for a format that does not have a fixed,
    /// uniform bytes-per-pixel mapping. Depth, stencil, planar, and
    /// block-compressed formats fall in this category.
    #[error("Image readback is not supported for format {0:?}")]
    UnsupportedReadbackFormat(vk::Format),
}
