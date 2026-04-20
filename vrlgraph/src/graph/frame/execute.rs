use ash::vk;
use smallvec::SmallVec;

use crate::graph::cmd::Cmd;
use crate::graph::image::ImageEntry;
use crate::graph::query::MAX_TIMESTAMP_PASSES;
use crate::graph::resources::register_bindless;
use crate::graph::schedule::access::LoadOp;
use crate::graph::schedule::barrier::{BarrierState, compute_barriers, compute_buffer_barriers};
use crate::graph::schedule::dag;
use crate::graph::schedule::pass::{PassAccess, RecordedPass};
use crate::graph::transfer::AcquireKind;
use crate::graph::{Graph, GraphError};
use crate::graph::image::ImageOrigin;
use crate::resource::ResourcePool;

impl Graph {
    pub(crate) fn execute_frame(
        &mut self,
        pending: Vec<RecordedPass<'_>>,
    ) -> Result<(), GraphError> {
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

        for entry in &mut self.images[self.persistent_count..] {
            let Some(handle) = entry.handle else {
                debug_assert!(
                    entry.origin == ImageOrigin::External,
                    "transient image '{}' has no GPU handle after allocation",
                    entry.desc.label
                );
                continue;
            };
            let Some(gpu_image) = self.resources.get_image(handle) else {
                continue;
            };
            let view = gpu_image.view;
            register_bindless(entry, &mut self.bindless, view);
        }

        let mut img_states: Vec<BarrierState> =
            self.images.iter().map(BarrierState::from_entry).collect();

        let acquires = self.transfer.take_completed_acquires();
        let mip_gens = self.transfer.take_completed_mip_gens();
        let transfer_timeline_value = acquires
            .iter()
            .map(|a| a.timeline_value)
            .chain(mip_gens.iter().map(|m| m.timeline_value))
            .max();

        let raw = self.frames[self.frame_index].pool.reset_and_begin()?;
        let mut cmd = Cmd::new_frame(
            raw,
            device.clone(),
            self.device.ext_dynamic_state3().clone(),
            self.device.debug_utils().cloned(),
            &self.images,
            &self.resources,
            &self.transfer,
            self.frame_index,
        );

        cmd.bind_global_set(self.bindless.pipeline_layout(), self.bindless.set());

        if !acquires.is_empty() {
            let mut acq_img_barriers: SmallVec<[vk::ImageMemoryBarrier2<'_>; 4]> = SmallVec::new();
            let mut acq_buf_barriers: SmallVec<[vk::BufferMemoryBarrier2<'_>; 4]> = SmallVec::new();
            let gfx_family = self.device.graphics_family();
            let xfer_family = self.transfer.transfer_family();

            for a in &acquires {
                match &a.kind {
                    AcquireKind::Buffer { raw, size } => {
                        acq_buf_barriers.push(
                            vk::BufferMemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                                .src_access_mask(vk::AccessFlags2::NONE)
                                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                                .dst_access_mask(
                                    vk::AccessFlags2::SHADER_READ
                                        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
                                        | vk::AccessFlags2::INDEX_READ,
                                )
                                .src_queue_family_index(xfer_family)
                                .dst_queue_family_index(gfx_family)
                                .buffer(*raw)
                                .offset(0)
                                .size(*size),
                        );
                    }
                    AcquireKind::Image {
                        raw,
                        dst_layout,
                        subresource_range,
                    } => {
                        acq_img_barriers.push(
                            vk::ImageMemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                                .src_access_mask(vk::AccessFlags2::NONE)
                                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                                .new_layout(*dst_layout)
                                .src_queue_family_index(xfer_family)
                                .dst_queue_family_index(gfx_family)
                                .image(*raw)
                                .subresource_range(*subresource_range),
                        );
                    }
                }
            }

            cmd.pipeline_barrier2_mixed(&acq_img_barriers, &acq_buf_barriers);
        }

        if !mip_gens.is_empty() {
            for mg in &mip_gens {
                cmd.generate_mipmaps(mg.image, mg.extent, mg.mip_levels);
            }
        }

        if self.timestamps.is_enabled() {
            let pool = self.timestamps.pools[self.frame_index].raw();
            cmd.reset_query_pool(pool, 0, MAX_TIMESTAMP_PASSES * 2);
            self.timestamps.names[self.frame_index].clear();
        }

        for pass in passes {
            let _cpu_span = tracing::info_span!("gpu_pass", pass_name = pass.name).entered();

            cmd.begin_debug_group(pass.name, [0.2, 0.6, 1.0, 1.0]);

            let pass_slot = self.timestamps.names[self.frame_index].len() as u32;
            let has_ts = self.timestamps.is_enabled() && pass_slot < MAX_TIMESTAMP_PASSES;

            if has_ts {
                let pool = self.timestamps.pools[self.frame_index].raw();
                cmd.write_timestamp(vk::PipelineStageFlags2::TOP_OF_PIPE, pool, pass_slot * 2);
            }

            let color_load_ops: SmallVec<[vk::AttachmentLoadOp; 4]> = pass
                .writes
                .iter()
                .filter(|w| w.is_color)
                .map(|w| {
                    let layer_idx = w.layer.unwrap_or(0) as usize;
                    resolve_load_op(
                        w.load_op,
                        img_states[w.image.0 as usize].layers[layer_idx].layout,
                    )
                })
                .collect();
            let depth_write: Option<(&PassAccess, vk::AttachmentLoadOp)> =
                pass.writes.iter().find(|w| w.is_depth).map(|w| {
                    let layer_idx = w.layer.unwrap_or(0) as usize;
                    (
                        w,
                        resolve_load_op(
                            w.load_op,
                            img_states[w.image.0 as usize].layers[layer_idx].layout,
                        ),
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

            debug_assert!(
                pass.view_mask == 0 || pass.writes.iter().all(|w| w.layer.is_none()),
                "multiview passes must use write(), not write_layer()"
            );

            let color_attachments: SmallVec<[vk::RenderingAttachmentInfo<'_>; 4]> = pass
                .writes
                .iter()
                .filter(|w| w.is_color)
                .zip(color_load_ops.iter())
                .map(|(w, &load_op)| {
                    let effective_layer = if pass.view_mask != 0 { None } else { w.layer };
                    let view = resolve_attachment_view(
                        &self.images[w.image.0 as usize],
                        effective_layer,
                        &self.resources,
                    );
                    vk::RenderingAttachmentInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(load_op)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: w.clear_color.unwrap_or([0.0; 4]),
                            },
                        })
                })
                .collect();

            let depth_attachment = depth_write.map(|(w, load_op)| {
                let effective_layer = if pass.view_mask != 0 { None } else { w.layer };
                let view = resolve_attachment_view(
                    &self.images[w.image.0 as usize],
                    effective_layer,
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

                let layer_count = if pass.view_mask != 0 { 0 } else { 1 };

                let mut rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent,
                    })
                    .layer_count(layer_count)
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

            cmd.reset_dynamic_state();
            (pass.execute)(&mut cmd);

            if is_graphics_pass {
                cmd.end_rendering();
            }

            if has_ts {
                let pool = self.timestamps.pools[self.frame_index].raw();
                cmd.write_timestamp(
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    pool,
                    pass_slot * 2 + 1,
                );
                self.timestamps.names[self.frame_index].push(pass.name);
            }

            cmd.end_debug_group();
        }

        if self.timestamps.is_enabled() && !self.timestamps.names[self.frame_index].is_empty() {
            self.timestamps.written[self.frame_index] = true;
        }

        if let Some(sc_h) = self.sc_graph_image {
            let sc_layer = &img_states[sc_h.0 as usize].layers[0];
            if sc_layer.layout != vk::ImageLayout::PRESENT_SRC_KHR {
                let (sc_raw, _) = self.images[sc_h.0 as usize].resolve(&self.resources);
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(sc_layer.stage)
                    .src_access_mask(sc_layer.access)
                    .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(sc_layer.layout)
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

        let image_available_wait = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.sync.image_available(fi))
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);

        let mut wait_infos: SmallVec<[vk::SemaphoreSubmitInfo<'_>; 2]> = SmallVec::new();
        wait_infos.push(image_available_wait);

        if let Some(tv) = transfer_timeline_value {
            wait_infos.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.transfer.timeline_semaphore())
                    .value(tv)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }

        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);

        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(buffer);

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_infos)
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
                return Err(GraphError::SwapchainOutOfDate);
            }
            Err(e) => {
                return Err(GraphError::Vulkan(e));
            }
        }

        for (i, state) in img_states.iter().enumerate().take(self.persistent_count) {
            let rep = state.representative();
            self.images[i].layout = rep.layout;
            self.images[i].stage = rep.stage;
            self.images[i].access = rep.access;
        }

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
