use super::FrameBuilder;
use crate::graph::cmd::Cmd;
use crate::graph::image::Image;
use crate::graph::schedule::access::{Access, LoadOp};
use crate::graph::schedule::pass::{
    BufferAccess, PassAccess, PassContext, ReadParam, RecordedPass, WithLayer, WithLayerLoadOp,
    WithLoadOp, WriteParam,
};

pub struct PassSetup<'a, 'frame> {
    builder: &'a mut FrameBuilder<'frame>,
    name: &'static str,
    reads: Vec<PassAccess>,
    writes: Vec<PassAccess>,
    buffer_reads: Vec<BufferAccess>,
    buffer_writes: Vec<BufferAccess>,
    view_mask: u32,
}

impl<'a, 'frame> PassSetup<'a, 'frame> {
    pub(super) fn new(builder: &'a mut FrameBuilder<'frame>, name: &'static str) -> Self {
        Self {
            builder,
            name,
            reads: Vec::new(),
            writes: Vec::new(),
            buffer_reads: Vec::new(),
            buffer_writes: Vec::new(),
            view_mask: 0,
        }
    }

    fn with_ctx(&mut self, f: impl FnOnce(&mut PassContext<'_>)) {
        let graph = &mut *self.builder.graph;
        let mut ctx = PassContext {
            reads: &mut self.reads,
            writes: &mut self.writes,
            buffer_reads: &mut self.buffer_reads,
            buffer_writes: &mut self.buffer_writes,
            images: &mut graph.images,
            frame_index: graph.frame_index,
            resources: &graph.resources,
        };
        f(&mut ctx);
    }

    pub fn read(mut self, param: impl ReadParam) -> Self {
        self.with_ctx(|ctx| param.apply_read(ctx));
        self
    }

    pub fn write(mut self, param: impl WriteParam) -> Self {
        self.with_ctx(|ctx| param.apply_write(ctx));
        self
    }

    pub fn write_with(self, image: Image, access: Access, load_op: LoadOp) -> Self {
        self.write(WithLoadOp(image, access, load_op))
    }

    pub fn write_layer(self, image: Image, access: Access, layer: u32) -> Self {
        self.write(WithLayer(image, access, layer))
    }

    pub fn write_layer_with(
        self,
        image: Image,
        access: Access,
        load_op: LoadOp,
        layer: u32,
    ) -> Self {
        self.write(WithLayerLoadOp(image, access, load_op, layer))
    }

    pub fn multiview(mut self, view_mask: u32) -> Self {
        self.view_mask = view_mask;
        self
    }

    pub fn execute<F>(self, f: F)
    where
        F: for<'b> FnOnce(&mut Cmd<'b>) + 'frame,
    {
        let PassSetup {
            builder,
            name,
            reads,
            writes,
            buffer_reads,
            buffer_writes,
            view_mask,
        } = self;
        builder.pending_passes.push(RecordedPass {
            name,
            reads,
            writes,
            buffer_reads,
            buffer_writes,
            view_mask,
            execute: Box::new(f),
        });
    }
}
