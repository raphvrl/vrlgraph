use ash::vk;
use smallvec::SmallVec;

use super::access::{Access, LoadOp};
use super::barrier::{BarrierState, compute_barriers, compute_buffer_barriers};
use super::command::Cmd;
use super::dag;
use super::image::Image;
use super::image::ImageEntry;
use super::pass::{
    BufferAccess, FrameResources, PassAccess, PassContext, ReadParam, RecordedPass, WithLayer,
    WithLayerLoadOp, WithLoadOp, WriteParam,
};
use super::query::{MAX_TIMESTAMP_PASSES, PassTiming};
use super::resources::register_bindless;
use super::{Frame, Graph, GraphError};
use crate::resource::ResourcePool;

pub struct PassSetup<'g> {
    graph: &'g mut Graph,
    name: &'static str,
    reads: Vec<PassAccess>,
    writes: Vec<PassAccess>,
    buffer_reads: Vec<BufferAccess>,
    buffer_writes: Vec<BufferAccess>,
    view_mask: u32,
}

impl<'g> PassSetup<'g> {
    fn with_ctx(&mut self, f: impl FnOnce(&mut PassContext<'_>)) {
        let graph = &mut *self.graph;
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
        F: FnOnce(&mut Cmd, &FrameResources<'_>) + 'static,
    {
        let PassSetup {
            graph,
            name,
            reads,
            writes,
            buffer_reads,
            buffer_writes,
            view_mask,
        } = self;
        graph.pending_passes.push(RecordedPass {
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

impl Graph {
    pub fn render_pass(&mut self, name: &'static str) -> PassSetup<'_> {
        assert!(
            self.frame_active,
            "render_pass() called outside begin_frame/flush"
        );
        PassSetup {
            graph: self,
            name,
            reads: Vec::new(),
            writes: Vec::new(),
            buffer_reads: Vec::new(),
            buffer_writes: Vec::new(),
            view_mask: 0,
        }
    }

    pub fn compute_pass(&mut self, name: &'static str) -> PassSetup<'_> {
        assert!(
            self.frame_active,
            "compute_pass() called outside begin_frame/flush"
        );
        PassSetup {
            graph: self,
            name,
            reads: Vec::new(),
            writes: Vec::new(),
            buffer_reads: Vec::new(),
            buffer_writes: Vec::new(),
            view_mask: 0,
        }
    }

    pub fn begin_frame(&mut self) -> Result<Frame, GraphError> {
        assert!(
            !self.frame_active,
            "begin_frame() called twice without end_frame()"
        );

        let resized = if let Some((w, h)) = self.pending_resize.take() {
            self.apply_resize(w, h)?
        } else {
            false
        };

        let idx = self.current;
        self.sync.wait(idx)?;

        self.last_timings.clear();
        if !self.timestamp_pools.is_empty() && self.timestamp_written[idx] {
            let n = self.timestamp_names[idx].len() as u32;
            if n > 0 {
                let mut results = vec![0u64; (n * 2) as usize];
                let ok = unsafe {
                    self.device.ash_device().get_query_pool_results(
                        self.timestamp_pools[idx].raw(),
                        0,
                        &mut results,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                };
                if ok.is_ok() {
                    let period = self.timestamp_period;
                    for (i, &name) in self.timestamp_names[idx].iter().enumerate() {
                        let begin = results[i * 2];
                        let end = results[i * 2 + 1];
                        if end >= begin {
                            let gpu_ns = ((end - begin) as f64 * period) as u64;
                            self.last_timings.push(PassTiming { name, gpu_ns });
                            tracing::debug!(gpu_ns, pass = name, "gpu_pass_timing");
                        }
                    }
                }
            }
            self.timestamp_written[idx] = false;
        }

        let image_index = match self
            .device
            .swapchain()
            .acquire_next_image(self.sync.image_available(idx))
        {
            Ok((i, suboptimal)) => {
                if suboptimal {
                    return Err(GraphError::SwapchainOutOfDate);
                }
                i
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Err(GraphError::SwapchainOutOfDate),
            Err(e) => return Err(GraphError::Vulkan(e)),
        };

        self.sync.reset(idx)?;

        self.frame_active = true;
        self.image_index = image_index;
        self.frame_index = idx;

        let extent = self.device.swapchain().extent();
        debug_assert!(
            (image_index as usize) < self.device.swapchain().image_count(),
            "swapchain image_index {} out of range (count = {})",
            image_index,
            self.device.swapchain().image_count(),
        );
        let raw_img = self.device.swapchain().images()[image_index as usize];
        let raw_view = self.device.swapchain().image_views()[image_index as usize];

        let backbuffer = Image(self.images.len() as u32);
        self.images
            .push(ImageEntry::external(raw_img, raw_view, extent));
        self.sc_graph_image = Some(backbuffer);

        Ok(Frame {
            backbuffer,
            extent,
            index: idx as u32,
            resized,
        })
    }

    pub fn end_frame(&mut self) -> Result<(), GraphError> {
        assert!(
            self.frame_active,
            "end_frame() called without begin_frame()"
        );

        let pending = std::mem::take(&mut self.pending_passes);
        let live_images = self.collect_live_images(&pending);
        let passes = dag::sort_and_cull_passes(pending, &live_images)
            .map_err(|e| GraphError::PassCycle(e.pass_name))?;

        let device = self.device.ash_device().clone();

        for entry in &mut self.images[..self.persistent_count] {
            if entry.handle.is_none() && entry.external.is_none() {
                let usage = entry.usage | vk::ImageUsageFlags::TRANSFER_DST;
                let handle = self.resources.create_image(
                    &device,
                    self.device.allocator_mut(),
                    &entry.desc,
                    usage,
                    entry.aspect,
                )?;
                let view = self
                    .resources
                    .get_image(handle)
                    .expect("image just created")
                    .view;
                register_bindless(entry, &mut self.bindless, view);
                entry.handle = Some(handle);
            }
        }

        self.transient_cache.allocate(
            &mut self.images,
            &passes,
            self.persistent_count,
            &mut self.resources,
            &device,
            self.device.allocator_mut(),
        )?;

        // Register transient images in the bindless table now that handles are assigned.
        for entry in &mut self.images[self.persistent_count..] {
            let Some(handle) = entry.handle else { continue };
            let Some(gpu_image) = self.resources.get_image(handle) else {
                continue;
            };
            let view = gpu_image.view;
            register_bindless(entry, &mut self.bindless, view);
        }

        let mut img_states: Vec<BarrierState> =
            self.images.iter().map(BarrierState::from_entry).collect();

        let raw = self.frames[self.frame_index].pool.reset_and_begin()?;
        let mut cmd = Cmd::new(
            raw,
            device.clone(),
            self.device.ext_dynamic_state3().clone(),
            self.device.debug_utils().cloned(),
        );

        cmd.bind_global_set(self.bindless.pipeline_layout(), self.bindless.set());

        if !self.timestamp_pools.is_empty() {
            let pool = self.timestamp_pools[self.frame_index].raw();
            cmd.reset_query_pool(pool, 0, MAX_TIMESTAMP_PASSES * 2);
            self.timestamp_names[self.frame_index].clear();
        }

        for pass in passes {
            let _cpu_span = tracing::info_span!("gpu_pass", pass_name = pass.name).entered();

            cmd.begin_debug_group(pass.name, [0.2, 0.6, 1.0, 1.0]);

            let pass_slot = self.timestamp_names[self.frame_index].len() as u32;
            let has_ts = !self.timestamp_pools.is_empty() && pass_slot < MAX_TIMESTAMP_PASSES;

            if has_ts {
                let pool = self.timestamp_pools[self.frame_index].raw();
                cmd.write_timestamp(vk::PipelineStageFlags2::TOP_OF_PIPE, pool, pass_slot * 2);
            }

            let color_load_ops: SmallVec<[vk::AttachmentLoadOp; 4]> = pass
                .writes
                .iter()
                .filter(|w| w.is_color)
                .map(|w| resolve_load_op(w.load_op, img_states[w.image.0 as usize].layout))
                .collect();
            let depth_write: Option<(&PassAccess, vk::AttachmentLoadOp)> =
                pass.writes.iter().find(|w| w.is_depth).map(|w| {
                    (
                        w,
                        resolve_load_op(w.load_op, img_states[w.image.0 as usize].layout),
                    )
                });

            let img_infos = compute_barriers(&pass.reads, &pass.writes, &mut img_states);
            let buf_infos = compute_buffer_barriers(
                &pass.buffer_reads,
                &pass.buffer_writes,
                &mut self.buffer_states,
            );

            if img_infos.is_some() || buf_infos.is_some() {
                let img_barriers: SmallVec<[vk::ImageMemoryBarrier2<'_>; 8]> = img_infos
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(|info| {
                        let (vk_image, _) =
                            self.images[info.image.0 as usize].resolve(&self.resources);
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(info.src_stage)
                            .src_access_mask(info.src_access)
                            .dst_stage_mask(info.dst_stage)
                            .dst_access_mask(info.dst_access)
                            .old_layout(info.old_layout)
                            .new_layout(info.new_layout)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(vk_image)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: self.images[info.image.0 as usize].aspect,
                                base_mip_level: 0,
                                level_count: vk::REMAINING_MIP_LEVELS,
                                base_array_layer: info.layer.unwrap_or(0),
                                layer_count: if info.layer.is_some() {
                                    1
                                } else {
                                    vk::REMAINING_ARRAY_LAYERS
                                },
                            })
                    })
                    .collect();

                let buf_barriers: SmallVec<[vk::BufferMemoryBarrier2<'_>; 4]> = buf_infos
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(|info| {
                        let raw_buf = self
                            .resources
                            .get_buffer(info.handle)
                            .expect("buffer referenced in pass no longer exists")
                            .raw;
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(info.src_stage)
                            .src_access_mask(info.src_access)
                            .dst_stage_mask(info.dst_stage)
                            .dst_access_mask(info.dst_access)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .buffer(raw_buf)
                            .offset(0)
                            .size(vk::WHOLE_SIZE)
                    })
                    .collect();

                cmd.pipeline_barrier2_mixed(&img_barriers, &buf_barriers);
            }

            let color_attachments: SmallVec<[vk::RenderingAttachmentInfo<'_>; 4]> = pass
                .writes
                .iter()
                .filter(|w| w.is_color)
                .zip(color_load_ops.iter())
                .map(|(w, &load_op)| {
                    let view = resolve_attachment_view(
                        &self.images[w.image.0 as usize],
                        w.layer,
                        &self.resources,
                    );
                    vk::RenderingAttachmentInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(load_op)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue { float32: [0.0; 4] },
                        })
                })
                .collect();

            let depth_attachment = depth_write.map(|(w, load_op)| {
                let view = resolve_attachment_view(
                    &self.images[w.image.0 as usize],
                    w.layer,
                    &self.resources,
                );
                vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(w.layout)
                    .load_op(load_op)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    })
            });

            let is_graphics_pass = !color_attachments.is_empty() || depth_attachment.is_some();

            if is_graphics_pass {
                let extent = pass
                    .writes
                    .iter()
                    .find(|w| w.is_color || w.is_depth)
                    .map(|w| {
                        let e = self.images[w.image.0 as usize].desc.extent;
                        vk::Extent2D {
                            width: e.width,
                            height: e.height,
                        }
                    })
                    .unwrap_or_default();

                let mut rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent,
                    })
                    .layer_count(1)
                    .color_attachments(&color_attachments);

                if pass.view_mask != 0 {
                    rendering_info = rendering_info.view_mask(pass.view_mask);
                }

                if let Some(ref depth) = depth_attachment {
                    rendering_info = rendering_info.depth_attachment(depth);
                }

                cmd.begin_rendering(&rendering_info);

                let n = color_attachments.len() as u32;
                if n > 0 {
                    cmd.set_default_blend_state(n);
                }
            }

            let frame_res = FrameResources::new(&self.images, &self.resources, self.frame_index);
            (pass.execute)(&mut cmd, &frame_res);

            if is_graphics_pass {
                cmd.end_rendering();
            }

            if has_ts {
                let pool = self.timestamp_pools[self.frame_index].raw();
                cmd.write_timestamp(
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    pool,
                    pass_slot * 2 + 1,
                );
                self.timestamp_names[self.frame_index].push(pass.name);
            }

            cmd.end_debug_group();
        }

        if !self.timestamp_pools.is_empty() && !self.timestamp_names[self.frame_index].is_empty() {
            self.timestamp_written[self.frame_index] = true;
        }

        if let Some(sc_h) = self.sc_graph_image {
            let state = &img_states[sc_h.0 as usize];
            if state.layout != vk::ImageLayout::PRESENT_SRC_KHR {
                let (sc_raw, _) = self.images[sc_h.0 as usize].resolve(&self.resources);
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(state.stage)
                    .src_access_mask(state.access)
                    .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(state.layout)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(sc_raw)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                cmd.pipeline_barrier2(&[barrier]);
            }
        }

        let buffer = cmd.finish()?;

        let fi = self.frame_index;
        let ii = self.image_index as usize;

        let render_finished = self.sync.render_finished(ii);

        let wait_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.sync.image_available(fi))
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);

        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);

        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(buffer);

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(std::slice::from_ref(&wait_info))
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_info));

        unsafe {
            device.queue_submit2(
                self.device.queue().raw(),
                &[submit_info],
                self.sync.in_flight_fence(fi),
            )?;
        }

        let signal_semaphores = [render_finished];

        match self.device.swapchain().present(
            self.device.queue().raw(),
            self.image_index,
            &signal_semaphores,
        ) {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.cleanup_frame();
                return Err(GraphError::SwapchainOutOfDate);
            }
            Err(e) => {
                self.cleanup_frame();
                return Err(GraphError::Vulkan(e));
            }
        }

        for (i, state) in img_states.iter().enumerate().take(self.persistent_count) {
            self.images[i].layout = state.layout;
            self.images[i].stage = state.stage;
            self.images[i].access = state.access;
        }

        self.cleanup_frame();
        self.current = (self.current + 1) % self.frames.len();
        Ok(())
    }
}

fn resolve_attachment_view(
    entry: &ImageEntry,
    layer: Option<u32>,
    pool: &ResourcePool,
) -> vk::ImageView {
    match layer {
        None => entry.view(pool),
        Some(l) => {
            let h = entry.handle.expect("image not yet allocated");
            let img = pool.get_image(h).expect("image destroyed");
            img.layer_views
                .get(l as usize)
                .copied()
                .expect("layer index out of range")
        }
    }
}

#[inline]
fn resolve_load_op(op: LoadOp, current_layout: vk::ImageLayout) -> vk::AttachmentLoadOp {
    match op {
        LoadOp::Auto => {
            if current_layout == vk::ImageLayout::UNDEFINED {
                vk::AttachmentLoadOp::CLEAR
            } else {
                vk::AttachmentLoadOp::LOAD
            }
        }
        LoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
        LoadOp::Load => vk::AttachmentLoadOp::LOAD,
        LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
    }
}
