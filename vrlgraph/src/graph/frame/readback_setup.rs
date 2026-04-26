use std::sync::{Arc, OnceLock};

use ash::vk;
use gpu_allocator::MemoryLocation;

use super::FrameBuilder;
use crate::graph::GraphError;
use crate::graph::image::Image;
use crate::graph::readback::{
    BufferReadback, ImageMeta, ImageReadback, ReadbackInner, align_up, bytes_per_pixel,
};
use crate::graph::schedule::access::LoadOp;
use crate::graph::schedule::pass::{BufferAccess, PassAccess, RecordedPass};
use crate::resource::{Buffer, BufferDesc};

impl<'frame> FrameBuilder<'frame> {
    /// Schedules a copy of `image` to a CPU-visible buffer at frame end.
    ///
    /// Inserts an implicit transfer pass that runs after the image's last
    /// write in the dependency graph. The DAG ensures correct ordering;
    /// no manual barrier is required.
    ///
    /// Returns an [`ImageReadback`] handle that becomes ready once the
    /// frame's GPU work completes (typically after `frames_in_flight`
    /// frames). Poll with [`ImageReadback::is_ready`] / [`ImageReadback::try_get`]
    /// or block with [`ImageReadback::wait`].
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::UnsupportedReadbackFormat`] if the image
    /// format is not a supported color format. Depth, stencil, planar, and
    /// block-compressed formats are rejected.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vrlgraph::prelude::*;
    /// # fn run(graph: &mut gpu::Graph) -> Result<(), gpu::GraphError> {
    /// let mut frame = graph.begin_frame()?;
    /// let bb = frame.backbuffer;
    /// frame.render_pass("scene")
    ///     .write((bb, gpu::Access::ColorAttachment))
    ///     .execute(|_| {});
    /// let shot = frame.readback_image(bb)?;
    /// frame.submit()?;
    /// let _data = shot.wait(graph);
    /// # Ok(()) }
    /// ```
    pub fn readback_image(&mut self, image: Image) -> Result<ImageReadback, GraphError> {
        let (format, extent_w, extent_h) = {
            let entry = &self.graph.images[image.0 as usize];
            if entry.desc.format == vk::Format::UNDEFINED {
                let sc = self.graph.device.swapchain();
                let ext = sc.extent();
                (sc.format(), ext.width, ext.height)
            } else {
                (
                    entry.desc.format,
                    entry.desc.extent.width,
                    entry.desc.extent.height,
                )
            }
        };

        let bpp = bytes_per_pixel(format).ok_or(GraphError::UnsupportedReadbackFormat(format))?;

        let row_align = self
            .graph
            .device
            .properties()
            .limits
            .optimal_buffer_copy_row_pitch_alignment as u32;
        let row_pitch = align_up(extent_w * bpp, row_align.max(1));
        let staging_size = u64::from(row_pitch) * u64::from(extent_h);

        self.graph.images[image.0 as usize].usage |= vk::ImageUsageFlags::TRANSFER_SRC;

        let staging = self.graph.create_buffer(&BufferDesc {
            size: staging_size,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            location: MemoryLocation::GpuToCpu,
            label: format!("readback:image_{}", image.0),
        })?;
        let staging_handle = staging.0;

        let buf = self
            .graph
            .resources
            .get_buffer(staging_handle)
            .expect("staging buffer just created");
        let mapped_ptr = buf
            .mapped_ptr()
            .expect("readback staging buffer must be host-visible");
        let memory = buf.memory();
        let memory_offset = buf.memory_offset();

        let cycles = self.graph.frames.len() as u32 + 1;
        let inner = Arc::new(ReadbackInner {
            buffer: staging_handle,
            data_size: staging_size,
            mapped_ptr,
            memory,
            memory_offset,
            device: self.graph.device.ash_device().clone(),
            fence: OnceLock::new(),
            free_queue: Arc::clone(&self.graph.readback_free_queue),
            cycles_after_submit: cycles,
            image_meta: Some(ImageMeta {
                width: extent_w,
                height: extent_h,
                format,
                row_pitch,
            }),
        });

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(row_pitch / bpp)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: extent_w,
                height: extent_h,
                depth: 1,
            });

        let read_access = PassAccess {
            image,
            layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            stage: vk::PipelineStageFlags2::TRANSFER,
            access: vk::AccessFlags2::TRANSFER_READ,
            is_color: false,
            is_depth: false,
            load_op: LoadOp::Auto,
            layer: None,
            clear_color: None,
        };
        let staging_write = BufferAccess {
            handle: staging_handle,
            stage: vk::PipelineStageFlags2::TRANSFER,
            access: vk::AccessFlags2::TRANSFER_WRITE,
        };

        self.pending_passes.push(RecordedPass {
            name: "readback_image",
            reads: vec![read_access],
            writes: Vec::new(),
            buffer_reads: Vec::new(),
            buffer_writes: vec![staging_write],
            view_mask: 0,
            execute: Box::new(move |cmd| {
                let src_raw = cmd.image(image).raw;
                let dst_raw = cmd.buffer(staging).raw;
                cmd.copy_image_to_buffer_region(src_raw, dst_raw, std::slice::from_ref(&region));
            }),
        });

        self.pending_readbacks.push(Arc::clone(&inner));

        Ok(ImageReadback { inner })
    }

    /// Schedules a copy of `buffer` (full range) to a CPU-visible buffer at
    /// frame end. Inserts an implicit transfer pass that runs after the
    /// buffer's last write in the dependency graph.
    ///
    /// Returns a [`BufferReadback`] handle. Use [`BufferReadback::wait`] to
    /// block, or [`BufferReadback::wait_as`] for a typed view.
    ///
    /// # Panics
    ///
    /// Panics if the buffer handle is stale (the buffer was destroyed
    /// before the frame began) or if staging buffer allocation fails. The
    /// allocation panic is not recoverable in user code; ensure the
    /// graph has enough host-visible memory headroom for the readback size.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vrlgraph::prelude::*;
    /// # fn run(graph: &mut gpu::Graph, ssbo: gpu::Buffer) -> Result<(), gpu::GraphError> {
    /// let mut frame = graph.begin_frame()?;
    /// frame.compute_pass("histogram")
    ///     .write((ssbo, gpu::BufferUsage::StorageWrite))
    ///     .execute(|_| {});
    /// let bins = frame.readback_buffer(ssbo);
    /// frame.submit()?;
    /// let _counts: &[u32] = bins.wait_as::<u32>(graph);
    /// # Ok(()) }
    /// ```
    pub fn readback_buffer(&mut self, buffer: Buffer) -> BufferReadback {
        let size = self
            .graph
            .resources
            .get_buffer(buffer.0)
            .expect("readback_buffer: buffer handle stale")
            .size;

        let staging = self
            .graph
            .create_buffer(&BufferDesc {
                size,
                usage: vk::BufferUsageFlags::TRANSFER_DST,
                location: MemoryLocation::GpuToCpu,
                label: "readback:buffer_staging".to_string(),
            })
            .expect("readback_buffer: staging allocation failed");
        let staging_handle = staging.0;

        let buf = self
            .graph
            .resources
            .get_buffer(staging_handle)
            .expect("staging buffer just created");
        let mapped_ptr = buf
            .mapped_ptr()
            .expect("readback staging buffer must be host-visible");
        let memory = buf.memory();
        let memory_offset = buf.memory_offset();

        let cycles = self.graph.frames.len() as u32 + 1;
        let inner = Arc::new(ReadbackInner {
            buffer: staging_handle,
            data_size: size,
            mapped_ptr,
            memory,
            memory_offset,
            device: self.graph.device.ash_device().clone(),
            fence: OnceLock::new(),
            free_queue: Arc::clone(&self.graph.readback_free_queue),
            cycles_after_submit: cycles,
            image_meta: None,
        });

        let read_access = BufferAccess {
            handle: buffer.0,
            stage: vk::PipelineStageFlags2::TRANSFER,
            access: vk::AccessFlags2::TRANSFER_READ,
        };
        let write_access = BufferAccess {
            handle: staging_handle,
            stage: vk::PipelineStageFlags2::TRANSFER,
            access: vk::AccessFlags2::TRANSFER_WRITE,
        };

        let src = buffer;
        let dst = staging;
        self.pending_passes.push(RecordedPass {
            name: "readback_buffer",
            reads: Vec::new(),
            writes: Vec::new(),
            buffer_reads: vec![read_access],
            buffer_writes: vec![write_access],
            view_mask: 0,
            execute: Box::new(move |cmd| {
                let src_raw = cmd.buffer(src).raw;
                let dst_raw = cmd.buffer(dst).raw;
                cmd.copy_buffer_region(src_raw, dst_raw, size);
            }),
        });

        self.pending_readbacks.push(Arc::clone(&inner));

        BufferReadback { inner }
    }
}
