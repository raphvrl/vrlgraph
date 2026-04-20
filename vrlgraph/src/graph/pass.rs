#![allow(private_interfaces)]

use ash::vk;

use crate::resource::{AsyncBuffer, Buffer, BufferHandle, StreamingBufferHandle};

use super::access::{Access, BufferUsage, LoadOp};
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

type ExecuteFn = Box<dyn for<'a> FnOnce(&mut Cmd<'a>)>;

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
///     .execute(|cmd| { /* ... */ });
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
impl sealed::Sealed for (AsyncBuffer, BufferUsage) {}
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
        ctx.writes.push(make_write_access(
            image,
            access,
            LoadOp::Auto,
            Some(layer),
            None,
        ));
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
///     .execute(|cmd| { /* ... */ });
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
        ctx.writes.push(make_write_access(
            image,
            access,
            LoadOp::Clear,
            None,
            Some(color),
        ));
    }
}

impl WriteParam for WithLayerClearColor {
    fn apply_write(self, ctx: &mut PassContext<'_>) {
        let WithLayerClearColor(image, access, color, layer) = self;
        ctx.images[image.0 as usize].usage |= access.usage_flags();
        ctx.writes.push(make_write_access(
            image,
            access,
            LoadOp::Clear,
            Some(layer),
            Some(color),
        ));
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

impl ReadParam for (AsyncBuffer, BufferUsage) {
    fn apply_read(self, ctx: &mut PassContext<'_>) {
        let (handle, usage) = self;
        ctx.buffer_reads.push(BufferAccess::new(handle.0, usage));
    }
}

impl WriteParam for (AsyncBuffer, BufferUsage) {
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

