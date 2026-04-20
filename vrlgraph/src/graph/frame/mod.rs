mod begin;
mod execute;
mod pass_setup;

pub use pass_setup::PassSetup;

use ash::vk;

use super::image::{Image, ImageBuilder, ImageOrigin};
use super::schedule::pass::RecordedPass;
use super::{Graph, GraphError};

/// In-flight frame returned by [`Graph::begin_frame`].
///
/// Holds the per-frame state and accumulates pass declarations. Submit GPU
/// work with [`submit`](FrameBuilder::submit). Dropping the builder without
/// calling `submit` discards the frame (rolls back transient state).
#[must_use = "FrameBuilder must be submitted with .submit() — dropping it discards the frame"]
pub struct FrameBuilder<'frame> {
    pub(crate) graph: &'frame mut Graph,
    /// The swapchain image for this frame. Declare it as a write target with
    /// [`Access::ColorAttachment`] in the final render pass.
    pub backbuffer: Image,
    /// Current surface dimensions. Use this for viewport and scissor setup.
    pub extent: vk::Extent2D,
    /// Swapchain image index for this frame.
    pub index: u32,
    /// `true` on the first frame after the window was resized.
    pub resized: bool,
    pub(crate) pending_passes: Vec<RecordedPass<'frame>>,
}

impl<'frame> FrameBuilder<'frame> {
    pub fn graph(&mut self) -> &mut Graph {
        self.graph
    }

    /// Declares a transient image local to this frame.
    ///
    /// Transient images are allocated (via the transient cache) when the
    /// frame is submitted and their graph-side bookkeeping is freed at
    /// frame end (on [`submit`](Self::submit) success the cleanup happens
    /// at the start of the next frame; on drop it happens immediately).
    /// They cannot outlive the frame.
    ///
    /// When no extent is specified, the swapchain extent is used.
    ///
    /// ```no_run
    /// use vrlgraph::prelude::*;
    /// # fn demo(frame: &mut gpu::FrameBuilder<'_>) -> Result<(), gpu::GraphError> {
    /// let scratch = frame.transient_image("scratch")
    ///     .format(vk::Format::R8G8B8A8_UNORM)
    ///     .build()?;
    ///
    /// let bb = frame.backbuffer;
    /// frame.render_pass("use_scratch")
    ///     .read((scratch, gpu::Access::ShaderRead))
    ///     .write((bb, gpu::Access::ColorAttachment))
    ///     .execute(|_cmd| {});
    /// # Ok(()) }
    /// ```
    pub fn transient_image(&mut self, label: impl Into<String>) -> ImageBuilder<'_> {
        ImageBuilder::new(self.graph, ImageOrigin::Transient, label)
    }

    pub fn render_pass<'a>(&'a mut self, name: &'static str) -> PassSetup<'a, 'frame> {
        PassSetup::new(self, name)
    }

    pub fn compute_pass<'a>(&'a mut self, name: &'static str) -> PassSetup<'a, 'frame> {
        self.render_pass(name)
    }

    /// Submits all declared passes to the GPU and presents the frame.
    pub fn submit(mut self) -> Result<(), GraphError> {
        let pending = std::mem::take(&mut self.pending_passes);
        match self.graph.execute_frame(pending) {
            Ok(()) => {
                self.graph.frame_active = false;
                std::mem::forget(self);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

impl<'frame> Drop for FrameBuilder<'frame> {
    fn drop(&mut self) {
        self.graph.cleanup_frame();
    }
}
