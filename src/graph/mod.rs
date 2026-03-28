//! Render graph — pass declaration, frame execution, and resource management.
//!
//! The main entry point is [`Graph`]. Build one with [`Graph::builder`], then
//! call [`Graph::begin_frame`] / [`Graph::end_frame`] each frame with passes
//! declared in between.

mod access;
mod barrier;
mod builder;
mod command;
mod dag;
mod descriptor;
mod frame;
mod image;
mod pass;
mod pipeline;
mod pipelines;
mod query;
#[cfg(debug_assertions)]
mod reload;
mod resources;
mod sync;
mod transient;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use ash::vk;
use thiserror::Error;

use crate::device::{DeviceError, GpuDevice};
use crate::resource::{BufferHandle, ImageDesc, PipelineHandle, ResourceError, ResourcePool};
use barrier::BufferBarrierState;
use command::{CommandError, CommandPool};
use descriptor::OwnedDescriptorResources;
use image::ImageEntry;
use pass::RecordedPass;
use query::TimestampQueryPool;
#[cfg(debug_assertions)]
use reload::{PipelineDesc, ShaderWatcher};
use sync::{FrameSync, SyncError};
use transient::TransientCache;

pub use crate::resource::StreamingBufferHandle;
pub use access::{Access, BufferUsage, LoadOp};
pub use builder::{GpuPreference, GraphBuilder, PresentMode};
pub use command::Cmd;
pub use descriptor::{DescriptorSetBuilder, DescriptorWrite, DynamicDescriptorSet, PushDescriptor};
pub use frame::PassSetup;
pub use image::GraphImage;
pub use pass::{FrameResources, ReadParam, WithLayer, WithLayerLoadOp, WithLoadOp, WriteParam};
pub use pipeline::{ComputePipelineBuilder, PipelineBuilder};
pub use query::PassTiming;

type ResizableFn = Box<dyn Fn(vk::Extent2D) -> ImageDesc>;

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
    /// A texture file could not be read or decoded.
    #[error("Image load error: {0}")]
    ImageLoad(String),
    /// A SPIR-V file could not be read or is malformed.
    #[error("Shader load error: {0}")]
    ShaderLoad(String),
    /// The window handle provided to the builder is no longer valid.
    #[error("Window handle unavailable")]
    WindowHandle,
    /// The swapchain is out of date and must be recreated. Call [`Graph::resize`]
    /// and skip the current frame. This is expected after a window resize.
    #[error("Swapchain out of date")]
    SwapchainOutOfDate,
    /// A dependency cycle was detected in the declared passes. The named pass
    /// is part of the cycle.
    #[error("Render pass cycle detected involving pass '{0}'")]
    PassCycle(&'static str),
}

/// Per-frame data returned by [`Graph::begin_frame`].
///
/// `backbuffer` is the swapchain image for this frame — write to it as a
/// color attachment to put pixels on screen. `extent` reflects the current
/// surface size and should be used for viewport/scissor setup. `resized` is
/// `true` only on the first frame after a window resize, which is the right
/// moment to call [`DynamicDescriptorSet::update`] on any sets that reference
/// resizable images.
pub struct Frame {
    /// The swapchain image for this frame. Declare it as a write target with
    /// [`Access::ColorAttachment`] in the final render pass.
    pub backbuffer: GraphImage,
    /// Current surface dimensions. Use this for viewport and scissor setup.
    pub extent: vk::Extent2D,
    /// Swapchain image index for this frame.
    pub index: u32,
    /// `true` on the first frame after the window was resized.
    pub resized: bool,
}

pub(crate) struct FrameData {
    pool: CommandPool,
}

/// The render graph.
///
/// `Graph` owns the Vulkan device, the swapchain, all GPU resources, and the
/// frame timeline. You interact with it by declaring passes between
/// [`begin_frame`](Graph::begin_frame) and [`end_frame`](Graph::end_frame).
/// The graph resolves pass dependencies, inserts pipeline barriers, and
/// submits all work at `end_frame`.
///
/// # Example
///
/// ```rust,no_run
/// use vrlgraph::prelude::*;
///
/// # fn example(window: &impl raw_window_handle::HasWindowHandle) -> Result<(), GraphError> {
/// let mut graph = Graph::builder()
///     .window(window)
///     .size(1280, 720)
///     .build()?;
///
/// let frame = graph.begin_frame()?;
///
/// graph.render_pass("main")
///     .write((frame.backbuffer, Access::ColorAttachment))
///     .execute(move |cmd, res| {
///         // record commands
///     });
///
/// graph.end_frame()?;
/// # Ok(())
/// # }
/// ```
pub struct Graph {
    pub(crate) resources: ResourcePool,
    pub(crate) owned_descs: Vec<OwnedDescriptorResources>,
    pub(crate) frames: Vec<FrameData>,
    pub(crate) sync: FrameSync,
    pub(crate) current: usize,
    pub(crate) present_mode: vk::PresentModeKHR,
    pub(crate) images: Vec<ImageEntry>,
    pub(crate) persistent_count: usize,
    pub(crate) resizable_images: Vec<(usize, ResizableFn)>,
    pub(crate) buffer_states: HashMap<BufferHandle, BufferBarrierState>,
    pub(crate) pipeline_cache: vk::PipelineCache,
    pub(crate) pipeline_cache_path: Option<PathBuf>,
    pub(crate) transient_cache: TransientCache,
    pub(crate) timestamp_pools: Vec<TimestampQueryPool>,
    pub(crate) timestamp_names: Vec<Vec<&'static str>>,
    pub(crate) timestamp_written: Vec<bool>,
    pub(crate) timestamp_period: f64,
    pub(crate) last_timings: Vec<PassTiming>,
    pub(crate) pending_passes: Vec<RecordedPass>,
    pub(crate) frame_active: bool,
    pub(crate) image_index: u32,
    pub(crate) frame_index: usize,
    pub(crate) sc_graph_image: Option<GraphImage>,
    pub(crate) pending_resize: Option<(u32, u32)>,
    #[cfg(debug_assertions)]
    pub(crate) pipeline_descs: HashMap<PipelineHandle, PipelineDesc>,
    #[cfg(debug_assertions)]
    pub(crate) shader_watcher: ShaderWatcher,
    pub(crate) device: GpuDevice,
}

impl Graph {
    /// Returns a [`GraphBuilder`] to configure and initialize the graph.
    pub fn builder() -> GraphBuilder {
        GraphBuilder::new()
    }

    pub(super) fn from_device(
        device: GpuDevice,
        frames_count: usize,
        present_mode: vk::PresentModeKHR,
        pipeline_cache_path: Option<PathBuf>,
    ) -> Result<Self, GraphError> {
        let sync = FrameSync::new(
            device.ash_device(),
            frames_count,
            device.swapchain().image_count(),
        )?;

        let frames = (0..frames_count)
            .map(|_| {
                Ok(FrameData {
                    pool: CommandPool::new(device.ash_device(), device.graphics_family())?,
                })
            })
            .collect::<Result<Vec<_>, CommandError>>()?;

        let cache_data = pipeline_cache_path
            .as_ref()
            .and_then(|p| std::fs::read(p).ok())
            .unwrap_or_default();

        let pipeline_cache = unsafe {
            device.ash_device().create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default().initial_data(&cache_data),
                None,
            )?
        };

        let timestamp_period = device.properties().limits.timestamp_period as f64;

        let timestamp_pools = if timestamp_period > 0.0 {
            (0..frames_count)
                .map(|_| TimestampQueryPool::new(device.ash_device()))
                .collect::<Result<Vec<_>, vk::Result>>()?
        } else {
            tracing::warn!(
                "GPU timestamp queries not supported on this device — profiling disabled"
            );
            Vec::new()
        };

        let timestamp_names = vec![Vec::new(); frames_count];
        let timestamp_written = vec![false; frames_count];

        Ok(Self {
            device,
            resources: ResourcePool::new(),
            owned_descs: Vec::new(),
            frames,
            sync,
            current: 0,
            present_mode,
            images: Vec::new(),
            persistent_count: 0,
            resizable_images: Vec::new(),
            buffer_states: HashMap::new(),
            pipeline_cache,
            pipeline_cache_path,
            transient_cache: TransientCache::new(),
            timestamp_pools,
            timestamp_names,
            timestamp_written,
            timestamp_period,
            last_timings: Vec::new(),
            pending_passes: Vec::new(),
            frame_active: false,
            image_index: 0,
            frame_index: 0,
            sc_graph_image: None,
            pending_resize: None,
            #[cfg(debug_assertions)]
            pipeline_descs: HashMap::new(),
            #[cfg(debug_assertions)]
            shader_watcher: ShaderWatcher::default(),
        })
    }

    /// Blocks until all GPU work submitted by this graph has completed.
    pub fn wait_idle(&self) -> Result<(), GraphError> {
        unsafe { self.device.ash_device().device_wait_idle()? };
        Ok(())
    }

    /// Returns a reference to the underlying [`GpuDevice`].
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// Returns a reference to the underlying [`ash::Device`].
    pub fn ash_device(&self) -> &ash::Device {
        self.device.ash_device()
    }

    /// Returns a mutable reference to the underlying [`GpuDevice`].
    pub fn device_mut(&mut self) -> &mut GpuDevice {
        &mut self.device
    }

    /// Returns the GPU execution times for each pass in the previous frame.
    ///
    /// Timings are available after [`end_frame`](Graph::end_frame) returns.
    /// Each entry contains the pass name and its duration in nanoseconds.
    /// Returns an empty slice if the GPU does not support timestamp queries.
    pub fn pass_timings(&self) -> &[PassTiming] {
        &self.last_timings
    }

    /// Returns the `VkImageView` for a graph image, if it is currently allocated.
    ///
    /// Returns `None` for transient images outside of a frame or for images
    /// that have not been used in any pass yet.
    pub fn image_view(&self, handle: GraphImage) -> Option<vk::ImageView> {
        let entry = &self.images[handle.0 as usize];
        if entry.handle.is_some() || entry.external.is_some() {
            Some(entry.view(&self.resources))
        } else {
            None
        }
    }

    /// Queues a swapchain resize. The swapchain is recreated at the start of
    /// the next frame. Resizable images are updated automatically. Call this
    /// when the window size changes or after receiving [`GraphError::SwapchainOutOfDate`].
    pub fn resize(&mut self, width: u32, height: u32) {
        self.pending_resize = Some((width, height));
    }

    fn apply_resize(&mut self, width: u32, height: u32) -> Result<bool, GraphError> {
        self.device
            .recreate_swapchain((width, height), self.present_mode)
            .map_err(GraphError::from)?;

        let new_extent = self.device.swapchain().extent();
        let device = self.device.ash_device().clone();

        let updates: Vec<(usize, ImageDesc)> = self
            .resizable_images
            .iter()
            .map(|(idx, f)| (*idx, f(new_extent)))
            .collect();

        let mut any_recreated = false;

        for (idx, new_desc) in updates {
            let entry = &mut self.images[idx];

            if entry.desc == new_desc {
                continue;
            }

            any_recreated = true;
            let saved_usage = entry.usage;
            let aspect = entry.aspect;

            if let Some(h) = entry.handle.take() {
                self.resources
                    .destroy_image(&device, self.device.allocator_mut(), h);
            }

            entry.desc = new_desc;
            entry.layout = vk::ImageLayout::UNDEFINED;
            entry.stage = vk::PipelineStageFlags2::NONE;
            entry.access = vk::AccessFlags2::NONE;

            if !saved_usage.is_empty() {
                let usage = saved_usage | vk::ImageUsageFlags::TRANSFER_DST;
                let handle = self.resources.create_image(
                    &device,
                    self.device.allocator_mut(),
                    &entry.desc,
                    usage,
                    aspect,
                )?;
                self.images[idx].handle = Some(handle);
            }
        }

        Ok(any_recreated)
    }

    pub(crate) fn collect_live_images(&self, passes: &[RecordedPass]) -> HashSet<u32> {
        let mut live = HashSet::new();

        if let Some(sc) = self.sc_graph_image {
            live.insert(sc.0);
        }

        for i in 0..self.persistent_count {
            if passes
                .iter()
                .any(|p| p.writes.iter().any(|w| w.image.0 == i as u32))
            {
                live.insert(i as u32);
            }
        }

        live
    }

    pub(in crate::graph) fn cleanup_frame(&mut self) {
        self.images.truncate(self.persistent_count);
        self.frame_active = false;
        self.sc_graph_image = None;
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            let device = self.device.ash_device();

            _ = device.device_wait_idle();

            if let Some(path) = &self.pipeline_cache_path
                && let Ok(data) = self
                    .device
                    .ash_device()
                    .get_pipeline_cache_data(self.pipeline_cache)
            {
                let _ = std::fs::write(path, &data);
            }

            self.device
                .ash_device()
                .destroy_pipeline_cache(self.pipeline_cache, None);
        }
        let device = self.device.ash_device().clone();
        let alloc = self.device.allocator_mut();
        self.resources.drain_buffers(&device, alloc);
        self.resources.drain_pipelines(&device);
        self.resources.drain_samplers(&device);
        for owned in self.owned_descs.drain(..) {
            unsafe {
                device.destroy_descriptor_pool(owned.pool, None);
                device.destroy_descriptor_set_layout(owned.layout, None);
            }
        }
        let alloc = self.device.allocator_mut();
        self.transient_cache
            .clear(&mut self.resources, &device, alloc);
        for entry in self.images.drain(..) {
            if let Some(h) = entry.handle {
                self.resources.destroy_image(&device, alloc, h);
            }
        }
    }
}
