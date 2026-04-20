mod debug;
mod draw;
mod resources;
mod state;
mod transfer;

use ash::vk;
use thiserror::Error;

#[cfg(debug_assertions)]
use std::cell::Cell;

#[cfg(debug_assertions)]
use super::pipeline::validate::ReflectedPushConstants;
use super::image::ImageEntry;
use super::transfer::TransferManager;
use crate::resource::ResourcePool;

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
    pub(super) raw: vk::CommandBuffer,
    pub(super) device: ash::Device,
    pub(super) ext_ds3: ash::ext::extended_dynamic_state3::Device,

    pub(super) debug_utils: Option<ash::ext::debug_utils::Device>,
    pub(super) frame: Option<FrameCtx<'a>>,
    pub(super) bound_layout: Option<vk::PipelineLayout>,
    pub(super) bound_bind_point: vk::PipelineBindPoint,
    #[cfg(debug_assertions)]
    pub(super) reflected_pc: Option<ReflectedPushConstants>,
    #[cfg(debug_assertions)]
    pub(super) pc_mismatch_warned: Cell<bool>,
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

    pub(super) fn frame_ctx(&self) -> &FrameCtx<'a> {
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

    pub(crate) fn finish(self) -> Result<vk::CommandBuffer, CommandError> {
        unsafe { self.device.end_command_buffer(self.raw)? };
        Ok(self.raw)
    }
}
