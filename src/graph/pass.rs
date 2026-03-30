#![allow(private_interfaces)]

use ash::vk;

use crate::resource::{
    Buffer, BufferHandle, GpuBuffer, GpuImage, GpuPipeline, Pipeline, ResourcePool,
    StreamingBufferHandle,
};

use super::access::{Access, BufferUsage, LoadOp};
use super::bindless::Sampler;
use super::command::Cmd;
use super::image::{Image, ImageEntry};

#[derive(Clone)]
pub(crate) struct PassAccess {
    pub image: Image,
    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub is_color: bool,
    pub is_depth: bool,
    pub load_op: LoadOp,
    pub layer: Option<u32>,
    pub clear_color: Option<[f32; 4]>,
}

#[derive(Clone)]
pub(crate) struct BufferAccess {
    pub handle: BufferHandle,
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

impl BufferAccess {
    pub(crate) fn new(handle: BufferHandle, usage: BufferUsage) -> Self {
        Self {
            handle,
            stage: usage.stage(),
            access: usage.flags(),
        }
    }
}

pub(crate) struct RecordedPass {
    pub name: &'static str,
    pub reads: Vec<PassAccess>,
    pub writes: Vec<PassAccess>,
    pub buffer_reads: Vec<BufferAccess>,
    pub buffer_writes: Vec<BufferAccess>,

    pub view_mask: u32,
    pub execute: ExecuteFn,
}

type ExecuteFn = Box<dyn FnOnce(&mut Cmd, &FrameResources<'_>)>;

pub(crate) struct PassContext<'a> {
    pub reads: &'a mut Vec<PassAccess>,
    pub writes: &'a mut Vec<PassAccess>,
    pub buffer_reads: &'a mut Vec<BufferAccess>,
    pub buffer_writes: &'a mut Vec<BufferAccess>,
    pub images: &'a mut Vec<ImageEntry>,
    pub frame_index: usize,
    pub resources: &'a crate::resource::ResourcePool,
}

mod sealed {
    pub trait Sealed {}
}

pub trait ReadParam: sealed::Sealed {
    #[doc(hidden)]
    fn apply_read(self, ctx: &mut PassContext<'_>);
}

pub trait WriteParam: sealed::Sealed {
    #[doc(hidden)]
    fn apply_write(self, ctx: &mut PassContext<'_>);
}

/// An image write with an explicit [`LoadOp`].
///
/// Use instead of a plain `(image, access)` tuple when you need to control
/// whether the attachment is cleared, preserved, or discarded at the start of
/// the pass.
///
/// ```rust,no_run
/// # use vrlgraph::prelude::*;
/// # fn example(graph: &mut Graph, target: Image) {
/// graph.render_pass("accumulate")
///     .write(WithLoadOp(target, Access::ColorAttachment, LoadOp::Load))
///     .execute(|cmd, res| { /* ... */ });
/// # }
/// ```
pub struct WithLoadOp(pub Image, pub Access, pub LoadOp);

/// An image write targeting a single layer of an array image or cubemap.
///
/// The pass will only render into the specified layer. Useful for building
/// cubemaps face by face or updating individual slices of an array texture.
pub struct WithLayer(pub Image, pub Access, pub u32);

/// An image write targeting a single layer with an explicit [`LoadOp`].
pub struct WithLayerLoadOp(pub Image, pub Access, pub LoadOp, pub u32);

impl sealed::Sealed for (Image, Access) {}
impl sealed::Sealed for WithLoadOp {}
impl sealed::Sealed for WithLayer {}
impl sealed::Sealed for WithLayerLoadOp {}
impl sealed::Sealed for (Buffer, BufferUsage) {}
impl sealed::Sealed for (StreamingBufferHandle, BufferUsage) {}

fn make_write_access(
    image: Image,
    access: Access,
    load_op: LoadOp,
    layer: Option<u32>,
    clear_color: Option<[f32; 4]>,
) -> PassAccess {
    PassAccess {
        image,
        layout: access.layout(),
        stage: access.stage(),
        access: access.flags(),
        is_color: access.is_color_attachment(),
        is_depth: access.is_depth_attachment(),
        load_op,
        layer,
        clear_color,
    }
}

impl ReadParam for (Image, Access) {
    fn apply_read(self, ctx: &mut PassContext<'_>) {
        let (image, access) = self;
        ctx.images[image.0 as usize].usage |= access.usage_flags();
        ctx.reads.push(PassAccess {
            image,
            layout: access.layout(),
            stage: access.stage(),
            access: access.flags(),
            is_color: false,
            is_depth: false,
            load_op: LoadOp::Auto,
            layer: None,
            clear_color: None,
        });
    }
}

impl WriteParam for (Image, Access) {
    fn apply_write(self, ctx: &mut PassContext<'_>) {
        let (image, access) = self;
        ctx.images[image.0 as usize].usage |= access.usage_flags();
        ctx.writes
            .push(make_write_access(image, access, LoadOp::Auto, None, None));
    }
}

impl WriteParam for WithLoadOp {
    fn apply_write(self, ctx: &mut PassContext<'_>) {
        let WithLoadOp(image, access, load_op) = self;
        ctx.images[image.0 as usize].usage |= access.usage_flags();
        ctx.writes
            .push(make_write_access(image, access, load_op, None, None));
    }
}

impl WriteParam for WithLayer {
    fn apply_write(self, ctx: &mut PassContext<'_>) {
        let WithLayer(image, access, layer) = self;
        ctx.images[image.0 as usize].usage |= access.usage_flags();
        ctx.writes
            .push(make_write_access(image, access, LoadOp::Auto, Some(layer), None));
    }
}

impl WriteParam for WithLayerLoadOp {
    fn apply_write(self, ctx: &mut PassContext<'_>) {
        let WithLayerLoadOp(image, access, load_op, layer) = self;
        ctx.images[image.0 as usize].usage |= access.usage_flags();
        ctx.writes
            .push(make_write_access(image, access, load_op, Some(layer), None));
    }
}

/// An image write that clears the attachment to a specific color at the start
/// of the pass. Implies [`LoadOp::Clear`].
///
/// ```rust,no_run
/// # use vrlgraph::prelude::*;
/// # use vrlgraph::graph::WithClearColor;
/// # fn example(graph: &mut Graph, frame: &Frame) {
/// graph.render_pass("main")
///     .write(WithClearColor(frame.backbuffer, Access::ColorAttachment, [0.1, 0.2, 0.3, 1.0]))
///     .execute(|cmd, res| { /* ... */ });
/// # }
/// ```
pub struct WithClearColor(pub Image, pub Access, pub [f32; 4]);

/// An image write targeting a single layer with a specific clear color.
pub struct WithLayerClearColor(pub Image, pub Access, pub [f32; 4], pub u32);

impl sealed::Sealed for WithClearColor {}
impl sealed::Sealed for WithLayerClearColor {}

impl WriteParam for WithClearColor {
    fn apply_write(self, ctx: &mut PassContext<'_>) {
        let WithClearColor(image, access, color) = self;
        ctx.images[image.0 as usize].usage |= access.usage_flags();
        ctx.writes
            .push(make_write_access(image, access, LoadOp::Clear, None, Some(color)));
    }
}

impl WriteParam for WithLayerClearColor {
    fn apply_write(self, ctx: &mut PassContext<'_>) {
        let WithLayerClearColor(image, access, color, layer) = self;
        ctx.images[image.0 as usize].usage |= access.usage_flags();
        ctx.writes
            .push(make_write_access(image, access, LoadOp::Clear, Some(layer), Some(color)));
    }
}

impl ReadParam for (Buffer, BufferUsage) {
    fn apply_read(self, ctx: &mut PassContext<'_>) {
        let (handle, usage) = self;
        ctx.buffer_reads.push(BufferAccess::new(handle.0, usage));
    }
}

impl WriteParam for (Buffer, BufferUsage) {
    fn apply_write(self, ctx: &mut PassContext<'_>) {
        let (handle, usage) = self;
        ctx.buffer_writes.push(BufferAccess::new(handle.0, usage));
    }
}

impl ReadParam for (StreamingBufferHandle, BufferUsage) {
    fn apply_read(self, ctx: &mut PassContext<'_>) {
        let (handle, usage) = self;
        let slot = ctx
            .resources
            .streaming_slot(handle, ctx.frame_index)
            .expect("streaming buffer handle stale — destroyed before pass recording");
        ctx.buffer_reads.push(BufferAccess::new(slot, usage));
    }
}

impl WriteParam for (StreamingBufferHandle, BufferUsage) {
    fn apply_write(self, ctx: &mut PassContext<'_>) {
        let (handle, usage) = self;
        let slot = ctx
            .resources
            .streaming_slot(handle, ctx.frame_index)
            .expect("streaming buffer handle stale — destroyed before pass recording");
        ctx.buffer_writes.push(BufferAccess::new(slot, usage));
    }
}

/// Provides access to GPU resources inside a pass closure.
///
/// `FrameResources` is the second argument to the closure passed to
/// [`PassSetup::execute`](super::frame::PassSetup::execute). Use it to look
/// up the underlying GPU objects for handles declared in the pass.
pub struct FrameResources<'a> {
    pub(crate) images: &'a [ImageEntry],
    pub(crate) pool: &'a ResourcePool,
    pub(crate) frame_index: usize,
}

impl<'a> FrameResources<'a> {
    pub(crate) fn new(
        images: &'a [ImageEntry],
        pool: &'a ResourcePool,
        frame_index: usize,
    ) -> Self {
        Self {
            images,
            pool,
            frame_index,
        }
    }

    /// Returns the [`GpuImage`] for a graph image handle.
    ///
    /// # Panics
    ///
    /// Panics if the image is not allocated or if the handle is stale.
    pub fn image(&self, handle: Image) -> &GpuImage {
        let entry = &self.images[handle.0 as usize];
        let h = entry
            .handle
            .expect("image not allocated — declare it before recording the pass");
        self.pool
            .get_image(h)
            .expect("image handle stale — destroyed before frame end")
    }

    /// Returns the full `VkImageView` for a graph image (all layers, all mips).
    pub fn image_view(&self, handle: Image) -> vk::ImageView {
        self.images[handle.0 as usize].view(self.pool)
    }

    /// Returns a `VkImageView` for a single layer of an array image or cubemap.
    ///
    /// # Panics
    ///
    /// Panics if `layer` is out of range.
    pub fn layer_view(&self, handle: Image, layer: u32) -> vk::ImageView {
        let entry = &self.images[handle.0 as usize];
        let h = entry
            .handle
            .expect("image not allocated — declare it before recording the pass");
        let img = self
            .pool
            .get_image(h)
            .expect("image handle stale — destroyed before frame end");
        *img.layer_views
            .get(layer as usize)
            .unwrap_or_else(|| panic!("layer {layer} out of range (count: {})", img.layer_count))
    }

    /// Returns the [`GpuBuffer`] for a buffer handle.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale (buffer was destroyed before the frame ended).
    pub fn buffer(&self, handle: Buffer) -> &GpuBuffer {
        self.pool
            .get_buffer(handle.0)
            .expect("buffer handle stale — destroyed before frame end")
    }

    /// Returns the [`GpuBuffer`] for the current frame's slot of a streaming buffer.
    pub fn streaming_buffer(&self, handle: StreamingBufferHandle) -> &GpuBuffer {
        let slot = self
            .pool
            .streaming_slot(handle, self.frame_index)
            .expect("streaming buffer handle stale — destroyed before frame end");
        self.pool
            .get_buffer(slot)
            .expect("streaming buffer slot stale — internal error")
    }

    /// Returns the [`GpuPipeline`] for a pipeline handle.
    pub fn pipeline(&self, handle: Pipeline) -> &GpuPipeline {
        self.pool
            .get_pipeline(handle.0)
            .expect("pipeline handle stale — destroyed before frame end")
    }

    /// Returns the bindless sampled image index as a `u32` ready for push constants.
    ///
    /// The image must have been created with `SAMPLED` usage (e.g. via
    /// [`Graph::load_texture`](crate::graph::Graph::load_texture) or with
    /// `ash::vk::ImageUsageFlags::SAMPLED`).
    pub fn sampled_index(&self, handle: Image) -> u32 {
        self.images[handle.0 as usize]
            .sampled_index
            .expect("image has no bindless sampled index — was it created with SAMPLED usage?")
            .raw()
    }

    /// Returns the bindless storage image index as a `u32` ready for push constants.
    ///
    /// The image must have been created with `ash::vk::ImageUsageFlags::STORAGE`.
    pub fn storage_index(&self, handle: Image) -> u32 {
        self.images[handle.0 as usize]
            .storage_index
            .expect("image has no bindless storage index — was it created with STORAGE usage?")
            .raw()
    }

    /// Returns the bindless cubemap index as a `u32` ready for push constants.
    ///
    /// The image must have been created with [`ImageKind::Cubemap`](crate::resource::ImageKind::Cubemap)
    /// (or `CubemapArray`) and `ash::vk::ImageUsageFlags::SAMPLED`.
    pub fn cubemap_index(&self, handle: Image) -> u32 {
        self.images[handle.0 as usize]
            .cubemap_index
            .expect("image has no bindless cubemap index — was it created with Cubemap kind and SAMPLED usage?")
            .raw()
    }

    /// Returns the bindless 2D array index as a `u32` ready for push constants.
    ///
    /// The image must have been created with [`ImageKind::Image2DArray`](crate::resource::ImageKind::Image2DArray)
    /// and `ash::vk::ImageUsageFlags::SAMPLED`.
    pub fn array_index(&self, handle: Image) -> u32 {
        self.images[handle.0 as usize]
            .array_index
            .expect("image has no bindless array index — was it created with Image2DArray kind and SAMPLED usage?")
            .raw()
    }

    /// Returns the bindless sampler index as a `u32` ready for push constants.
    pub fn sampler_index(&self, sampler: Sampler) -> u32 {
        sampler.raw()
    }
}
