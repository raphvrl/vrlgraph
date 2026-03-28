use std::path::Path;

use ash::vk;
use gpu_allocator::MemoryLocation;

use super::command::{Cmd, CommandPool};
use super::descriptor::{DescriptorSetBuilder, OwnedDescriptorResources};
use super::image::{GraphImage, ImageEntry};
use super::{Graph, GraphError};
use crate::resource::{
    BufferDesc, BufferHandle, GpuBuffer, ImageDesc, ImageHandle, ResourceError, ResourcePool,
    SamplerHandle, StreamingBufferHandle,
};

impl Graph {
    pub fn create_transient(&mut self, desc: ImageDesc) -> GraphImage {
        assert!(
            self.frame_active,
            "create_transient() must be called after begin_frame()"
        );
        let h = GraphImage(self.images.len() as u32);
        self.images.push(ImageEntry::transient(desc));
        h
    }

    pub fn create_persistent(&mut self, desc: ImageDesc) -> Result<GraphImage, GraphError> {
        assert!(
            !self.frame_active,
            "create_persistent() must be called outside the frame loop"
        );
        let h = GraphImage(self.images.len() as u32);
        self.images.push(ImageEntry::persistent(desc));
        self.persistent_count += 1;

        let entry = self.images.last_mut().unwrap();
        if !entry.usage.is_empty() {
            let device = self.device.ash_device().clone();
            let usage = entry.usage | vk::ImageUsageFlags::TRANSFER_DST;
            let handle = self.resources.create_image(
                &device,
                self.device.allocator_mut(),
                &entry.desc,
                usage,
                entry.aspect,
            )?;
            entry.handle = Some(handle);
        }

        Ok(h)
    }

    pub fn create_resizable(
        &mut self,
        desc_fn: impl Fn(vk::Extent2D) -> ImageDesc + 'static,
    ) -> Result<GraphImage, GraphError> {
        assert!(
            !self.frame_active,
            "create_resizable() must be called outside the frame loop"
        );
        let extent = self.device.swapchain().extent();
        let desc = desc_fn(extent);
        let idx = self.images.len();
        let h = GraphImage(idx as u32);
        self.images.push(ImageEntry::persistent(desc));
        self.persistent_count += 1;
        self.resizable_images.push((idx, Box::new(desc_fn)));

        let entry = &mut self.images[idx];
        if !entry.usage.is_empty() {
            let device = self.device.ash_device().clone();
            let usage = entry.usage | vk::ImageUsageFlags::TRANSFER_DST;
            let handle = self.resources.create_image(
                &device,
                self.device.allocator_mut(),
                &entry.desc,
                usage,
                entry.aspect,
            )?;
            entry.handle = Some(handle);
        }

        Ok(h)
    }

    pub fn load_texture(&mut self, path: impl AsRef<Path>) -> Result<GraphImage, GraphError> {
        self.load_texture_with(path, ImageDesc::default())
    }

    pub fn load_texture_with(
        &mut self,
        path: impl AsRef<Path>,
        mut desc: ImageDesc,
    ) -> Result<GraphImage, GraphError> {
        assert!(
            !self.frame_active,
            "load_texture() must be called outside the frame loop"
        );

        let path = path.as_ref();

        let img = image::open(path)
            .map_err(|e| GraphError::ImageLoad(e.to_string()))?
            .to_rgba8();

        let (width, height) = img.dimensions();
        let pixels = img.into_raw();

        desc.extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        if desc.label.is_empty() {
            desc.label = path.to_string_lossy().into_owned();
        }

        if desc.mip_levels == 0 {
            desc.mip_levels = compute_mip_levels(width, height);
        }

        let usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let aspect = vk::ImageAspectFlags::COLOR;

        let device = self.device.ash_device().clone();
        let handle = self.resources.create_image(
            &device,
            self.device.allocator_mut(),
            &desc,
            usage,
            aspect,
        )?;

        self.upload_image_data(handle, &pixels, desc.extent, desc.mip_levels)?;

        let h = GraphImage(self.images.len() as u32);
        self.images.push(ImageEntry::loaded(desc, handle));
        self.persistent_count += 1;
        Ok(h)
    }

    pub fn destroy_image(&mut self, handle: GraphImage) {
        assert!(
            !self.frame_active,
            "destroy_image() must be called outside the frame loop"
        );
        let entry = &mut self.images[handle.0 as usize];
        if let Some(h) = entry.handle.take() {
            let device = self.device.ash_device().clone();
            self.resources
                .destroy_image(&device, self.device.allocator_mut(), h);
        }
    }

    pub fn create_buffer(&mut self, desc: &BufferDesc) -> Result<BufferHandle, ResourceError> {
        let device = self.device.ash_device().clone();
        self.resources
            .create_buffer(&device, self.device.allocator_mut(), desc)
    }

    pub fn destroy_buffer(&mut self, handle: BufferHandle) {
        let device = self.device.ash_device().clone();
        self.resources
            .destroy_buffer(&device, self.device.allocator_mut(), handle);
        self.buffer_states.remove(&handle);
    }

    pub fn get_buffer(&self, handle: BufferHandle) -> Option<&GpuBuffer> {
        self.resources.get_buffer(handle)
    }

    /// Returns the GPU virtual address of a buffer created with
    /// [`vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS`], or `None` otherwise.
    pub fn buffer_device_address(&self, handle: BufferHandle) -> Option<vk::DeviceAddress> {
        self.resources.get_buffer(handle)?.device_address
    }

    pub fn create_streaming_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        label: &str,
    ) -> Result<StreamingBufferHandle, ResourceError> {
        let frames = self.frames.len();
        let device = self.device.ash_device().clone();
        self.resources.create_streaming_buffer(
            &device,
            self.device.allocator_mut(),
            size,
            usage,
            location,
            label,
            frames,
        )
    }

    pub fn destroy_streaming_buffer(&mut self, handle: StreamingBufferHandle) {
        let device = self.device.ash_device().clone();
        let frames = self.frames.len();
        for i in 0..frames {
            if let Some(slot) = self.resources.streaming_slot(handle, i) {
                self.buffer_states.remove(&slot);
            }
        }
        self.resources
            .destroy_streaming_buffer(&device, self.device.allocator_mut(), handle);
    }

    pub fn upload_buffer<T: bytemuck::Pod>(
        &mut self,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferHandle, GraphError> {
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        let size = bytes.len() as vk::DeviceSize;
        let device = self.device.ash_device().clone();

        let staging = self.resources.create_buffer(
            &device,
            self.device.allocator_mut(),
            &BufferDesc {
                size,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                location: MemoryLocation::CpuToGpu,
                label: "staging_buffer_upload".to_string(),
            },
        )?;

        {
            let buf = self.resources.get_buffer(staging).unwrap();
            let ptr = buf.mapped_ptr().expect("staging buffer not host visible");
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
        }

        let dst = self.resources.create_buffer(
            &device,
            self.device.allocator_mut(),
            &BufferDesc {
                size,
                usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
                location: MemoryLocation::GpuOnly,
                label: "uploaded_buffer".to_string(),
            },
        )?;

        let pool = CommandPool::new(&device, self.device.graphics_family())?;
        let raw_cb = pool.reset_and_begin()?;
        let cmd = Cmd::new(
            raw_cb,
            device.clone(),
            self.device.push_descriptor().clone(),
            self.device.ext_dynamic_state3().clone(),
            None,
        );

        let src_raw = self.resources.get_buffer(staging).unwrap().raw;
        let dst_raw = self.resources.get_buffer(dst).unwrap().raw;
        cmd.copy_buffer_to_buffer(src_raw, dst_raw, size);

        let buffer = cmd.finish()?;
        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(buffer);
        let submit =
            vk::SubmitInfo2::default().command_buffer_infos(std::slice::from_ref(&cmd_info));

        unsafe {
            device.queue_submit2(self.device.queue().raw(), &[submit], vk::Fence::null())?;
            device.queue_wait_idle(self.device.queue().raw())?;
        }

        self.resources
            .destroy_buffer(&device, self.device.allocator_mut(), staging);

        Ok(dst)
    }

    pub fn write_buffer<T: bytemuck::Pod>(&self, handle: BufferHandle, data: &[T]) {
        self.resources
            .get_buffer(handle)
            .expect("write_buffer: invalid buffer handle")
            .write(data);
    }

    pub fn create_sampler(
        &mut self,
        info: &vk::SamplerCreateInfo,
    ) -> Result<SamplerHandle, GraphError> {
        Ok(self
            .resources
            .create_sampler(self.device.ash_device(), info)?)
    }

    pub fn destroy_sampler(&mut self, handle: SamplerHandle) {
        self.resources
            .destroy_sampler(self.device.ash_device(), handle);
    }

    pub fn descriptor_set(&mut self) -> DescriptorSetBuilder<'_> {
        DescriptorSetBuilder::new(self)
    }

    pub(in crate::graph) fn push_owned_desc(&mut self, desc: OwnedDescriptorResources) {
        self.owned_descs.push(desc);
    }

    pub(in crate::graph) fn images_slice(&self) -> &[ImageEntry] {
        &self.images
    }

    pub(in crate::graph) fn resources_ref(&self) -> &ResourcePool {
        &self.resources
    }

    pub(in crate::graph) fn upload_image_data(
        &mut self,
        dst: ImageHandle,
        pixels: &[u8],
        extent: vk::Extent3D,
        mip_levels: u32,
    ) -> Result<(), GraphError> {
        let device = self.device.ash_device().clone();

        let staging = self.resources.create_buffer(
            &device,
            self.device.allocator_mut(),
            &BufferDesc {
                size: pixels.len() as u64,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                location: MemoryLocation::CpuToGpu,
                label: "staging_upload".to_string(),
            },
        )?;

        {
            let buf = self.resources.get_buffer(staging).unwrap();
            let ptr = buf.mapped_ptr().expect("staging buffer not host visible");
            unsafe { std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr, pixels.len()) };
        }

        let pool = CommandPool::new(&device, self.device.graphics_family())?;
        let raw = pool.reset_and_begin()?;
        let cmd = Cmd::new(
            raw,
            device.clone(),
            self.device.push_descriptor().clone(),
            self.device.ext_dynamic_state3().clone(),
            None,
        );

        let dst_img = self.resources.get_image(dst).unwrap();
        let vk_img = dst_img.raw;
        let stg_buf = self.resources.get_buffer(staging).unwrap().raw;

        cmd.pipeline_barrier2(&[vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(vk_img)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: 1,
            })]);

        cmd.copy_buffer_to_image(stg_buf, vk_img, extent, 0);

        if mip_levels > 1 {
            cmd.generate_mipmaps(vk_img, extent, mip_levels);
        } else {
            cmd.pipeline_barrier2(&[vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(vk_img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })]);
        }

        let buffer = cmd.finish()?;
        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(buffer);
        let submit =
            vk::SubmitInfo2::default().command_buffer_infos(std::slice::from_ref(&cmd_info));
        unsafe {
            device.queue_submit2(self.device.queue().raw(), &[submit], vk::Fence::null())?;
            device.queue_wait_idle(self.device.queue().raw())?;
        }

        self.resources
            .destroy_buffer(&device, self.device.allocator_mut(), staging);

        Ok(())
    }
}

#[inline]
fn compute_mip_levels(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2().floor() as u32 + 1
}
