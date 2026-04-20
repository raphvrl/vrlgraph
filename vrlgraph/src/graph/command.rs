use std::ffi::CString;

use ash::vk;
use smallvec::SmallVec;
use thiserror::Error;

#[cfg(debug_assertions)]
use super::pipeline::validate::ReflectedPushConstants;
use super::bindless::Sampler;
use super::image::{Image, ImageEntry};
use super::transfer::TransferManager;
use crate::resource::{
    AsyncBuffer, Buffer, GpuBuffer, GpuImage, GpuPipeline, Pipeline, ResourcePool,
    StreamingBuffer,
};
use crate::types::{ColorWriteMask, CompareOp, CullMode, FrontFace, PolygonMode, Topology};
#[cfg(debug_assertions)]
use std::cell::Cell;

#[derive(Debug, Error)]
pub enum CommandError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

pub struct CommandPool {
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
    device: ash::Device,
}

impl CommandPool {
    pub fn new(device: &ash::Device, queue_family: u32) -> Result<Self, CommandError> {
        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(queue_family);

        let pool = unsafe { device.create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffer = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };

        Ok(Self {
            pool,
            buffer,
            device: device.clone(),
        })
    }

    pub fn reset_and_begin(&self) -> Result<vk::CommandBuffer, CommandError> {
        unsafe {
            self.device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
        }

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { self.device.begin_command_buffer(self.buffer, &begin_info)? };

        Ok(self.buffer)
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe { self.device.destroy_command_pool(self.pool, None) };
    }
}

pub(crate) struct FrameCtx<'a> {
    pub(crate) images: &'a [ImageEntry],
    pub(crate) pool: &'a ResourcePool,
    pub(crate) transfer: &'a TransferManager,
    pub(crate) frame_index: usize,
}

/// Command recorder passed to every pass closure.
///
/// `Cmd` wraps a `VkCommandBuffer` and exposes a typed API for binding
/// pipelines, setting dynamic state, issuing draw and dispatch commands, and
/// resolving handles declared in the pass to their underlying GPU objects.
/// You do not allocate or submit `Cmd` yourself — the graph creates one per
/// pass and passes it to the closure you provide to [`PassSetup::execute`](crate::graph::PassSetup::execute).
///
/// All rasterizer state (cull mode, depth test, depth clamp, blend, etc.) is dynamic.
/// Binding a graphics pipeline resets it to sensible defaults (no culling,
/// no depth test, no depth clamp, no blending). Override the state after binding the pipeline.
pub struct Cmd<'a> {
    raw: vk::CommandBuffer,
    device: ash::Device,
    ext_ds3: ash::ext::extended_dynamic_state3::Device,

    debug_utils: Option<ash::ext::debug_utils::Device>,
    frame: Option<FrameCtx<'a>>,
    bound_layout: Option<vk::PipelineLayout>,
    bound_bind_point: vk::PipelineBindPoint,
    #[cfg(debug_assertions)]
    reflected_pc: Option<ReflectedPushConstants>,
    #[cfg(debug_assertions)]
    pc_mismatch_warned: Cell<bool>,
}

impl<'a> Cmd<'a> {
    pub(crate) fn new_frame(
        raw: vk::CommandBuffer,
        device: ash::Device,
        ext_ds3: ash::ext::extended_dynamic_state3::Device,
        debug_utils: Option<ash::ext::debug_utils::Device>,
        images: &'a [ImageEntry],
        pool: &'a ResourcePool,
        transfer: &'a TransferManager,
        frame_index: usize,
    ) -> Self {
        Self {
            raw,
            device,
            ext_ds3,
            debug_utils,
            frame: Some(FrameCtx {
                images,
                pool,
                transfer,
                frame_index,
            }),
            bound_layout: None,
            bound_bind_point: vk::PipelineBindPoint::GRAPHICS,
            #[cfg(debug_assertions)]
            reflected_pc: None,
            #[cfg(debug_assertions)]
            pc_mismatch_warned: Cell::new(false),
        }
    }

    pub(crate) fn new_one_shot(
        raw: vk::CommandBuffer,
        device: ash::Device,
        ext_ds3: ash::ext::extended_dynamic_state3::Device,
        debug_utils: Option<ash::ext::debug_utils::Device>,
    ) -> Cmd<'static> {
        Cmd {
            raw,
            device,
            ext_ds3,
            debug_utils,
            frame: None,
            bound_layout: None,
            bound_bind_point: vk::PipelineBindPoint::GRAPHICS,
            #[cfg(debug_assertions)]
            reflected_pc: None,
            #[cfg(debug_assertions)]
            pc_mismatch_warned: Cell::new(false),
        }
    }

    fn frame_ctx(&self) -> &FrameCtx<'a> {
        self.frame
            .as_ref()
            .expect("Cmd: this method requires a frame context (called from a one-shot submit?)")
    }

    pub fn begin_rendering(&self, info: &vk::RenderingInfo) {
        unsafe { self.device.cmd_begin_rendering(self.raw, info) };
    }

    pub fn end_rendering(&self) {
        unsafe { self.device.cmd_end_rendering(self.raw) };
    }

    /// Binds a graphics pipeline. Dynamic rasterizer state is **not** reset;
    /// values persist across binds (OpenGL-like model). Call
    /// [`reset_dynamic_state`](Cmd::reset_dynamic_state) to restore defaults.
    pub fn bind_graphics_pipeline(&mut self, handle: Pipeline) {
        let pipe = self
            .frame_ctx()
            .pool
            .get_pipeline(handle.0)
            .expect("pipeline handle stale — destroyed before frame end");
        let raw_pipe = pipe.pipeline;
        let layout = pipe.layout;
        #[cfg(debug_assertions)]
        let reflected = pipe.reflected_pc.clone();
        unsafe {
            self.device
                .cmd_bind_pipeline(self.raw, vk::PipelineBindPoint::GRAPHICS, raw_pipe)
        };
        self.bound_layout = Some(layout);
        self.bound_bind_point = vk::PipelineBindPoint::GRAPHICS;
        #[cfg(debug_assertions)]
        {
            self.reflected_pc = reflected;
            self.pc_mismatch_warned.set(false);
        }
    }

    /// Resets all dynamic rasterizer state to defaults. Called once at the
    /// beginning of each pass. Defaults:
    /// - Rasterizer discard: off
    /// - Depth bias: off (constant = 0, clamp = 0, slope = 0)
    /// - Primitive restart: off
    /// - Cull mode: none
    /// - Front face: counter-clockwise
    /// - Topology: triangle list
    /// - Depth test/write: off
    /// - Depth compare: less-or-equal
    /// - Depth clamp: off
    /// - Polygon mode: fill
    /// - Blending: disabled (RGBA write mask, 1 attachment)
    pub fn reset_dynamic_state(&self) {
        self.set_rasterizer_discard_enable(false);
        self.set_depth_bias_enable(false);
        self.set_depth_bias(0.0, 0.0, 0.0);
        self.set_primitive_restart_enable(false);

        self.set_cull_mode(CullMode::NONE);
        self.set_front_face(FrontFace::CounterClockwise);
        self.set_primitive_topology(Topology::TriangleList);
        self.set_depth_test_enable(false);
        self.set_depth_write_enable(false);
        self.set_depth_compare_op(CompareOp::LessOrEqual);

        self.set_depth_clamp_enable(false);
        self.set_polygon_mode(PolygonMode::Fill);
        self.set_default_blend_state(1);
    }

    /// Binds a compute pipeline. Always call this before [`dispatch`](Cmd::dispatch).
    pub fn bind_compute_pipeline(&mut self, handle: Pipeline) {
        let pipe = self
            .frame_ctx()
            .pool
            .get_pipeline(handle.0)
            .expect("pipeline handle stale — destroyed before frame end");
        let raw_pipe = pipe.pipeline;
        let layout = pipe.layout;
        #[cfg(debug_assertions)]
        let reflected = pipe.reflected_pc.clone();
        unsafe {
            self.device
                .cmd_bind_pipeline(self.raw, vk::PipelineBindPoint::COMPUTE, raw_pipe)
        };
        self.bound_layout = Some(layout);
        self.bound_bind_point = vk::PipelineBindPoint::COMPUTE;
        #[cfg(debug_assertions)]
        {
            self.reflected_pc = reflected;
            self.pc_mismatch_warned.set(false);
        }
    }

    /// Sets the viewport. Use [`set_viewport_scissor`](Cmd::set_viewport_scissor)
    /// instead when the viewport and scissor cover the full surface.
    pub fn set_viewport(&self, viewport: vk::Viewport) {
        unsafe {
            self.device
                .cmd_set_viewport_with_count(self.raw, &[viewport])
        };
    }

    /// Sets the scissor rectangle.
    pub fn set_scissor(&self, scissor: vk::Rect2D) {
        unsafe { self.device.cmd_set_scissor_with_count(self.raw, &[scissor]) };
    }

    /// Sets the viewport and scissor to cover the full extent. Depth range is
    /// `[0.0, 1.0]`. This is the right call for most full-screen passes.
    pub fn set_viewport_scissor(&self, extent: vk::Extent2D) {
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        unsafe {
            self.device
                .cmd_set_viewport_with_count(self.raw, &[viewport]);
            self.device.cmd_set_scissor_with_count(self.raw, &[scissor]);
        }
    }

    pub fn set_cull_mode(&self, mode: CullMode) {
        unsafe { self.device.cmd_set_cull_mode(self.raw, mode.into()) };
    }

    pub fn set_front_face(&self, face: FrontFace) {
        unsafe { self.device.cmd_set_front_face(self.raw, face.into()) };
    }

    pub fn set_primitive_topology(&self, topology: Topology) {
        unsafe {
            self.device
                .cmd_set_primitive_topology(self.raw, topology.into())
        };
    }

    pub fn set_depth_test_enable(&self, enable: bool) {
        unsafe { self.device.cmd_set_depth_test_enable(self.raw, enable) };
    }

    pub fn set_depth_write_enable(&self, enable: bool) {
        unsafe { self.device.cmd_set_depth_write_enable(self.raw, enable) };
    }

    pub fn set_depth_compare_op(&self, op: CompareOp) {
        unsafe { self.device.cmd_set_depth_compare_op(self.raw, op.into()) };
    }

    pub fn set_depth_clamp_enable(&self, enable: bool) {
        unsafe { self.ext_ds3.cmd_set_depth_clamp_enable(self.raw, enable) };
    }

    pub fn set_polygon_mode(&self, mode: PolygonMode) {
        unsafe { self.ext_ds3.cmd_set_polygon_mode(self.raw, mode.into()) };
    }

    pub fn set_color_blend_enable(&self, first: u32, enables: &[vk::Bool32]) {
        unsafe {
            self.ext_ds3
                .cmd_set_color_blend_enable(self.raw, first, enables)
        };
    }

    pub fn set_color_blend_equation(&self, first: u32, equations: &[vk::ColorBlendEquationEXT]) {
        unsafe {
            self.ext_ds3
                .cmd_set_color_blend_equation(self.raw, first, equations)
        };
    }

    pub fn set_color_write_mask(&self, first: u32, masks: &[ColorWriteMask]) {
        let raw: SmallVec<[vk::ColorComponentFlags; 4]> =
            masks.iter().map(|m| (*m).into()).collect();
        unsafe { self.ext_ds3.cmd_set_color_write_mask(self.raw, first, &raw) };
    }

    pub fn set_rasterizer_discard_enable(&self, enable: bool) {
        unsafe {
            self.device
                .cmd_set_rasterizer_discard_enable(self.raw, enable)
        };
    }

    pub fn set_depth_bias_enable(&self, enable: bool) {
        unsafe { self.device.cmd_set_depth_bias_enable(self.raw, enable) };
    }

    pub fn set_depth_bias(&self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        unsafe {
            self.device
                .cmd_set_depth_bias(self.raw, constant_factor, clamp, slope_factor)
        };
    }

    pub fn set_primitive_restart_enable(&self, enable: bool) {
        unsafe {
            self.device
                .cmd_set_primitive_restart_enable(self.raw, enable)
        };
    }

    /// Binds a vertex buffer to slot 0.
    pub fn bind_vertex_buffer(&self, handle: Buffer, offset: vk::DeviceSize) {
        let raw_buf = self.buffer(handle).raw;
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(self.raw, 0, &[raw_buf], &[offset])
        };
    }

    /// Binds an index buffer. Indices are expected to be `u32`.
    pub fn bind_index_buffer(&self, handle: Buffer, offset: vk::DeviceSize) {
        let raw_buf = self.buffer(handle).raw;
        unsafe {
            self.device
                .cmd_bind_index_buffer(self.raw, raw_buf, offset, vk::IndexType::UINT32)
        };
    }

    /// Writes a [`ShaderType`](crate::ShaderType) value as push constant data
    /// with automatic scalar-layout padding.
    ///
    /// A pipeline must be bound first.
    pub fn push_constants<T: crate::ShaderType>(&self, data: &T) {
        #[cfg(debug_assertions)]
        if !self.pc_mismatch_warned.get()
            && let Some(reflected) = &self.reflected_pc
            && T::UNPADDED_SIZE != reflected.total_size
        {
            self.pc_mismatch_warned.set(true);
            super::pipeline::validate::validate_push_constants(
                reflected,
                T::UNPADDED_SIZE,
                std::any::type_name::<T>(),
            );
        }
        let mut buf = [0u8; 256];
        data.write_padded(&mut buf[..T::PADDED_SIZE]);
        self.push_constants_raw(&buf[..T::PADDED_SIZE]);
    }

    /// Writes raw bytes as push constant data.
    ///
    /// Prefer [`push_constants`](Cmd::push_constants) for typed values.
    /// Use this when the payload is assembled dynamically (e.g. a variable-length
    /// byte slice). A pipeline must be bound first.
    pub fn push_constants_raw(&self, data: &[u8]) {
        debug_assert!(
            self.bound_layout.is_some(),
            "Cmd: bind_pipeline() must be called before push_constants()"
        );
        if let Some(layout) = self.bound_layout {
            unsafe {
                self.device
                    .cmd_push_constants(self.raw, layout, vk::ShaderStageFlags::ALL, 0, data)
            };
        }
    }

    /// Draws `vertices` vertices and `instances` instances, starting from index 0.
    pub fn draw(&self, vertices: u32, instances: u32) {
        unsafe { self.device.cmd_draw(self.raw, vertices, instances, 0, 0) };
    }

    /// Draws using an index buffer. `first_index` is the byte offset into the
    /// index buffer divided by the index size. `vertex_offset` is added to each
    /// index value before fetching a vertex.
    pub fn draw_indexed(&self, indices: u32, instances: u32, first_index: u32, vertex_offset: i32) {
        unsafe {
            self.device.cmd_draw_indexed(
                self.raw,
                indices,
                instances,
                first_index,
                vertex_offset,
                0,
            )
        };
    }

    pub fn draw_indirect(
        &self,
        handle: Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        let raw_buf = self.buffer(handle).raw;
        unsafe {
            self.device
                .cmd_draw_indirect(self.raw, raw_buf, offset, draw_count, stride)
        };
    }

    pub fn draw_indexed_indirect(
        &self,
        handle: Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        let raw_buf = self.buffer(handle).raw;
        unsafe {
            self.device
                .cmd_draw_indexed_indirect(self.raw, raw_buf, offset, draw_count, stride)
        };
    }

    /// Dispatches a compute workload of `x * y * z` workgroups.
    /// A compute pipeline must be bound before calling this.
    pub fn dispatch(&self, x: u32, y: u32, z: u32) {
        debug_assert!(
            self.bound_bind_point == vk::PipelineBindPoint::COMPUTE,
            "Cmd: bind_compute_pipeline() must be called before dispatch()"
        );
        unsafe { self.device.cmd_dispatch(self.raw, x, y, z) };
    }

    /// Dispatches a compute workload using arguments read from a buffer at `offset`.
    /// The buffer must contain a `VkDispatchIndirectCommand`.
    pub fn dispatch_indirect(&self, handle: Buffer, offset: vk::DeviceSize) {
        debug_assert!(
            self.bound_bind_point == vk::PipelineBindPoint::COMPUTE,
            "Cmd: bind_compute_pipeline() must be called before dispatch_indirect()"
        );
        let raw_buf = self.buffer(handle).raw;
        unsafe {
            self.device
                .cmd_dispatch_indirect(self.raw, raw_buf, offset)
        };
    }

    pub fn clear_color(&self, image: vk::Image, color: [f32; 4]) {
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        unsafe {
            self.device.cmd_clear_color_image(
                self.raw,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: color },
                &[range],
            )
        };
    }

    pub(crate) fn copy_buffer_to_image_region(
        &self,
        buffer: vk::Buffer,
        image: vk::Image,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.raw,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                regions,
            )
        };
    }

    pub(crate) fn pipeline_barrier2(&self, image_barriers: &[vk::ImageMemoryBarrier2]) {
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(image_barriers);
        unsafe { self.device.cmd_pipeline_barrier2(self.raw, &dep_info) };
    }

    pub(crate) fn pipeline_barrier2_mixed(
        &self,
        image_barriers: &[vk::ImageMemoryBarrier2],
        buffer_barriers: &[vk::BufferMemoryBarrier2],
    ) {
        let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(image_barriers)
            .buffer_memory_barriers(buffer_barriers);
        unsafe { self.device.cmd_pipeline_barrier2(self.raw, &dep_info) };
    }

    pub(crate) fn generate_mipmaps(&self, image: vk::Image, extent: vk::Extent3D, mip_levels: u32) {
        assert!(mip_levels >= 2, "generate_mipmaps: mip_levels must be >= 2");

        let mut mip_w =
            i32::try_from(extent.width).expect("generate_mipmaps: image width exceeds i32::MAX");
        let mut mip_h =
            i32::try_from(extent.height).expect("generate_mipmaps: image height exceeds i32::MAX");

        for i in 1..mip_levels {
            self.pipeline_barrier2(&[vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })]);

            let next_w = (mip_w / 2).max(1);
            let next_h = (mip_h / 2).max(1);

            let blit = vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_w,
                        y: mip_h,
                        z: 1,
                    },
                ])
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: next_w,
                        y: next_h,
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            unsafe {
                self.device.cmd_blit_image(
                    self.raw,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }

            mip_w = next_w;
            mip_h = next_h;
        }

        let barriers = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip_levels - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        ];

        self.pipeline_barrier2(&barriers);
    }

    /// Binds the global bindless descriptor set at set 0 for both graphics and
    /// compute bind points. Called once at the start of the frame.
    pub(crate) fn bind_global_set(&self, layout: vk::PipelineLayout, set: vk::DescriptorSet) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.raw,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_bind_descriptor_sets(
                self.raw,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
        }
    }

    pub(crate) fn reset_query_pool(&self, pool: vk::QueryPool, first: u32, count: u32) {
        unsafe {
            self.device
                .cmd_reset_query_pool(self.raw, pool, first, count)
        };
    }

    pub(crate) fn write_timestamp(
        &self,
        stage: vk::PipelineStageFlags2,
        pool: vk::QueryPool,
        query: u32,
    ) {
        unsafe {
            self.device
                .cmd_write_timestamp2(self.raw, stage, pool, query)
        };
    }

    /// Disables blending and sets the write mask to RGBA for `count` color
    /// attachments. Call this after binding a graphics pipeline when no custom
    /// blend state is needed.
    pub fn set_default_blend_state(&self, count: u32) {
        let enables: SmallVec<[vk::Bool32; 4]> = smallvec::smallvec![vk::FALSE; count as usize];
        let masks: SmallVec<[ColorWriteMask; 4]> =
            smallvec::smallvec![ColorWriteMask::RGBA; count as usize];
        self.set_color_blend_enable(0, &enables);
        self.set_color_write_mask(0, &masks);
    }

    /// Opens a debug group visible in RenderDoc and Nsight. No-op if the
    /// debug utils extension is not enabled.
    pub fn begin_debug_group(&self, name: &str, color: [f32; 4]) {
        let Some(du) = &self.debug_utils else { return };
        let name_c = CString::new(name).unwrap_or_else(|_| c"<invalid>".to_owned());
        let label = vk::DebugUtilsLabelEXT::default()
            .label_name(&name_c)
            .color(color);
        unsafe { du.cmd_begin_debug_utils_label(self.raw, &label) };
    }

    /// Closes the current debug group opened with [`begin_debug_group`](Cmd::begin_debug_group).
    pub fn end_debug_group(&self) {
        let Some(du) = &self.debug_utils else { return };
        unsafe { du.cmd_end_debug_utils_label(self.raw) };
    }

    /// Inserts a single debug label at the current command position.
    pub fn insert_debug_label(&self, name: &str, color: [f32; 4]) {
        let Some(du) = &self.debug_utils else { return };
        let name_c = CString::new(name).unwrap_or_else(|_| c"<invalid>".to_owned());
        let label = vk::DebugUtilsLabelEXT::default()
            .label_name(&name_c)
            .color(color);
        unsafe { du.cmd_insert_debug_utils_label(self.raw, &label) };
    }

    pub(crate) fn finish(self) -> Result<vk::CommandBuffer, CommandError> {
        unsafe { self.device.end_command_buffer(self.raw)? };
        Ok(self.raw)
    }

    /// Returns the [`GpuImage`] for a graph image handle.
    ///
    /// # Panics
    ///
    /// Panics if the image is not allocated or if the handle is stale.
    pub fn image(&self, handle: Image) -> &GpuImage {
        let ctx = self.frame_ctx();
        let entry = &ctx.images[handle.0 as usize];
        let h = entry
            .handle
            .expect("image not allocated — declare it before recording the pass");
        ctx.pool
            .get_image(h)
            .expect("image handle stale — destroyed before frame end")
    }

    /// Returns the full `VkImageView` for a graph image (all layers, all mips).
    pub fn image_view(&self, handle: Image) -> vk::ImageView {
        let ctx = self.frame_ctx();
        ctx.images[handle.0 as usize].view(ctx.pool)
    }

    /// Returns a `VkImageView` for a single layer of an array image or cubemap.
    ///
    /// # Panics
    ///
    /// Panics if `layer` is out of range.
    pub fn layer_view(&self, handle: Image, layer: u32) -> vk::ImageView {
        let ctx = self.frame_ctx();
        let entry = &ctx.images[handle.0 as usize];
        let h = entry
            .handle
            .expect("image not allocated — declare it before recording the pass");
        let img = ctx
            .pool
            .get_image(h)
            .expect("image handle stale — destroyed before frame end");
        *img.layer_views
            .get(layer as usize)
            .unwrap_or_else(|| panic!("layer {layer} out of range (count: {})", img.layer_count))
    }

    /// Returns the [`GpuBuffer`] for a synchronous buffer handle.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale (buffer was destroyed before the frame ended).
    pub fn buffer(&self, handle: Buffer) -> &GpuBuffer {
        self.frame_ctx()
            .pool
            .get_buffer(handle.0)
            .expect("buffer handle stale — destroyed before frame end")
    }

    /// Returns the [`GpuBuffer`] for an async buffer if its transfer has
    /// completed, or `None` if the data is still being uploaded.
    pub fn try_buffer(&self, handle: AsyncBuffer) -> Option<&GpuBuffer> {
        let ctx = self.frame_ctx();
        if !ctx.transfer.is_buffer_ready_peek(handle.0) {
            return None;
        }
        ctx.pool.get_buffer(handle.0)
    }

    /// Returns the [`GpuBuffer`] for the current frame's slot of a streaming buffer.
    pub fn streaming_buffer(&self, handle: StreamingBuffer) -> &GpuBuffer {
        let ctx = self.frame_ctx();
        let slot = ctx
            .pool
            .streaming_slot(handle, ctx.frame_index)
            .expect("streaming buffer handle stale — destroyed before frame end");
        ctx.pool
            .get_buffer(slot)
            .expect("streaming buffer slot stale — internal error")
    }

    /// Returns the [`GpuPipeline`] for a pipeline handle.
    pub fn pipeline(&self, handle: Pipeline) -> &GpuPipeline {
        self.frame_ctx()
            .pool
            .get_pipeline(handle.0)
            .expect("pipeline handle stale — destroyed before frame end")
    }

    /// Returns the bindless sampled image index as a `u32` ready for push constants.
    ///
    /// The image must have been created with `SAMPLED` usage (e.g. via
    /// [`Graph::load_texture`](crate::graph::Graph::load_texture) builder or with
    /// `ash::vk::ImageUsageFlags::SAMPLED`).
    pub fn sampled_index(&self, handle: Image) -> u32 {
        self.frame_ctx().images[handle.0 as usize]
            .sampled_index
            .expect("image has no bindless sampled index — was it created with SAMPLED usage?")
            .raw()
    }

    /// Returns the bindless storage image index as a `u32` ready for push constants.
    ///
    /// The image must have been created with `ash::vk::ImageUsageFlags::STORAGE`.
    pub fn storage_index(&self, handle: Image) -> u32 {
        self.frame_ctx().images[handle.0 as usize]
            .storage_index
            .expect("image has no bindless storage index — was it created with STORAGE usage?")
            .raw()
    }

    /// Returns the bindless cubemap index as a `u32` ready for push constants.
    ///
    /// The image must have been created with [`gpu::ImageKind::Cubemap`](crate::gpu::ImageKind::Cubemap)
    /// (or `CubemapArray`) and `ash::vk::ImageUsageFlags::SAMPLED`.
    pub fn cubemap_index(&self, handle: Image) -> u32 {
        self.frame_ctx().images[handle.0 as usize]
            .cubemap_index
            .expect("image has no bindless cubemap index — was it created with Cubemap kind and SAMPLED usage?")
            .raw()
    }

    /// Returns the bindless 2D array index as a `u32` ready for push constants.
    ///
    /// The image must have been created with [`gpu::ImageKind::Image2DArray`](crate::gpu::ImageKind::Image2DArray)
    /// and `ash::vk::ImageUsageFlags::SAMPLED`.
    pub fn array_index(&self, handle: Image) -> u32 {
        self.frame_ctx().images[handle.0 as usize]
            .array_index
            .expect("image has no bindless array index — was it created with Image2DArray kind and SAMPLED usage?")
            .raw()
    }

    /// Returns the bindless sampler index as a `u32` ready for push constants.
    pub fn sampler_index(&self, sampler: Sampler) -> u32 {
        sampler.raw()
    }
}
