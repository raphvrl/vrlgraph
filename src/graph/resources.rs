use std::path::Path;

use ash::vk;
use gpu_allocator::MemoryLocation;

use super::bindless::{BindlessDescriptorTable, Sampler};
use super::command::{Cmd, CommandPool};
use super::image::{Image, ImageBuilder, ImageEntry, ImageOrigin};
use super::{Graph, GraphError};
use crate::resource::{
    Buffer, BufferDesc, GpuBuffer, ImageDesc, ImageHandle, ImageKind, ResourceError,
    StreamingBufferHandle,
};

/// Routes a newly created image view into the correct bindless binding(s) based on
/// image kind and usage, and writes the resulting indices back into `entry`.
pub(super) fn register_bindless(
    entry: &mut ImageEntry,
    bindless: &mut BindlessDescriptorTable,
    view: vk::ImageView,
) {
    if entry.usage.contains(vk::ImageUsageFlags::SAMPLED) {
        match entry.desc.kind {
            ImageKind::Cubemap | ImageKind::CubemapArray { .. } => {
                entry.cubemap_index = Some(
                    bindless
                        .allocate_cubemap_image(view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                );
            }
            ImageKind::Image2DArray { .. } => {
                entry.array_index = Some(
                    bindless.allocate_array_image(view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                );
            }
            ImageKind::Image2D => {
                entry.sampled_index = Some(
                    bindless
                        .allocate_sampled_image(view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                );
            }
        }
    }
    if entry.usage.contains(vk::ImageUsageFlags::STORAGE) {
        entry.storage_index = Some(bindless.allocate_storage_image(view));
    }
}

/// Frees all bindless slots held by an entry back into the free-lists.
pub(super) fn free_bindless(entry: &mut ImageEntry, bindless: &mut BindlessDescriptorTable) {
    if let Some(idx) = entry.sampled_index.take() {
        bindless.free_sampled(idx);
    }
    if let Some(idx) = entry.storage_index.take() {
        bindless.free_storage(idx);
    }
    if let Some(idx) = entry.cubemap_index.take() {
        bindless.free_cubemap(idx);
    }
    if let Some(idx) = entry.array_index.take() {
        bindless.free_array(idx);
    }
}

/// Updates all bindless slots for an entry after its view has changed (e.g. resize).
pub(super) fn update_bindless(
    entry: &ImageEntry,
    bindless: &BindlessDescriptorTable,
    view: vk::ImageView,
) {
    if let Some(si) = entry.sampled_index {
        bindless.update_sampled_image(si, view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
    if let Some(si) = entry.storage_index {
        bindless.update_storage_image(si, view);
    }
    if let Some(si) = entry.cubemap_index {
        bindless.update_cubemap_image(si, view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
    if let Some(si) = entry.array_index {
        bindless.update_array_image(si, view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
}

impl Graph {
    pub fn transient_image(&mut self) -> ImageBuilder<'_> {
        ImageBuilder::new(self, ImageOrigin::Transient)
    }

    pub fn persistent_image(&mut self) -> ImageBuilder<'_> {
        ImageBuilder::new(self, ImageOrigin::Persistent)
    }

    pub fn load_texture(&mut self, path: impl AsRef<Path>) -> Result<Image, GraphError> {
        self.load_texture_with(path, ImageDesc::default())
    }

    pub(crate) fn load_texture_with(
        &mut self,
        path: impl AsRef<Path>,
        mut desc: ImageDesc,
    ) -> Result<Image, GraphError> {
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

        let view = self
            .resources
            .get_image(handle)
            .expect("image just created")
            .view;
        let h = Image(self.images.len() as u32);
        let mut entry = ImageEntry::loaded(desc, handle);
        register_bindless(&mut entry, &mut self.bindless, view);
        self.images.push(entry);
        self.persistent_count += 1;
        Ok(h)
    }

    pub fn destroy_image(&mut self, handle: Image) {
        assert!(
            !self.frame_active,
            "destroy_image() must be called outside the frame loop"
        );

        let entry = &mut self.images[handle.0 as usize];
        free_bindless(entry, &mut self.bindless);

        if let Some(h) = entry.handle.take() {
            let device = self.device.ash_device().clone();
            self.resources
                .destroy_image(&device, self.device.allocator_mut(), h);
        }
    }

    pub fn create_buffer(&mut self, desc: &BufferDesc) -> Result<Buffer, ResourceError> {
        let device = self.device.ash_device().clone();
        self.resources
            .create_buffer(&device, self.device.allocator_mut(), desc)
            .map(Buffer)
    }

    pub fn destroy_buffer(&mut self, handle: Buffer) {
        let device = self.device.ash_device().clone();
        self.resources
            .destroy_buffer(&device, self.device.allocator_mut(), handle.0);
        self.buffer_states.remove(&handle.0);
    }

    pub fn get_buffer(&self, handle: Buffer) -> Option<&GpuBuffer> {
        self.resources.get_buffer(handle.0)
    }

    pub fn buffer_device_address(&self, handle: Buffer) -> vk::DeviceAddress {
        self.resources
            .get_buffer(handle.0)
            .expect("buffer_device_address: invalid handle")
            .device_address
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
    ) -> Result<Buffer, GraphError> {
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        self.upload_buffer_labeled(bytes, usage, "uploaded_buffer")
    }

    fn one_shot_submit(&mut self, f: impl FnOnce(&Cmd)) -> Result<(), GraphError> {
        let device = self.device.ash_device().clone();
        let pool = CommandPool::new(&device, self.device.graphics_family())?;
        let raw_cb = pool.reset_and_begin()?;
        let cmd = Cmd::new(
            raw_cb,
            device.clone(),
            self.device.ext_dynamic_state3().clone(),
            None,
        );
        f(&cmd);
        let buffer = cmd.finish()?;
        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(buffer);
        let submit =
            vk::SubmitInfo2::default().command_buffer_infos(std::slice::from_ref(&cmd_info));
        unsafe {
            device.queue_submit2(self.device.queue().raw(), &[submit], vk::Fence::null())?;
            device.queue_wait_idle(self.device.queue().raw())?;
        }
        Ok(())
    }

    fn create_staging(
        &mut self,
        data: &[u8],
        label: &str,
    ) -> Result<crate::resource::BufferHandle, GraphError> {
        let device = self.device.ash_device().clone();
        let handle = self.resources.create_buffer(
            &device,
            self.device.allocator_mut(),
            &BufferDesc {
                size: data.len() as vk::DeviceSize,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                location: MemoryLocation::CpuToGpu,
                label: label.to_string(),
            },
        )?;
        let buf = self
            .resources
            .get_buffer(handle)
            .expect("buffer just created");
        let ptr = buf.mapped_ptr().expect("staging buffer not host visible");
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
        Ok(handle)
    }

    fn destroy_staging(&mut self, handle: crate::resource::BufferHandle) {
        let device = self.device.ash_device().clone();
        self.resources
            .destroy_buffer(&device, self.device.allocator_mut(), handle);
    }

    pub(crate) fn upload_buffer_labeled(
        &mut self,
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
        label: &str,
    ) -> Result<Buffer, GraphError> {
        let size = bytes.len() as vk::DeviceSize;
        let device = self.device.ash_device().clone();

        let staging = self.create_staging(bytes, &format!("{label}_staging"))?;

        let dst = self.resources.create_buffer(
            &device,
            self.device.allocator_mut(),
            &BufferDesc {
                size,
                usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
                location: MemoryLocation::GpuOnly,
                label: label.to_string(),
            },
        )?;

        let src_raw = self
            .resources
            .get_buffer(staging)
            .expect("buffer just created")
            .raw;
        let dst_raw = self
            .resources
            .get_buffer(dst)
            .expect("buffer just created")
            .raw;
        self.one_shot_submit(|cmd| cmd.copy_buffer_to_buffer(src_raw, dst_raw, size))?;
        self.destroy_staging(staging);

        Ok(Buffer(dst))
    }

    pub fn write_buffer<T: bytemuck::Pod>(&self, handle: Buffer, data: &[T]) {
        self.resources
            .get_buffer(handle.0)
            .expect("write_buffer: invalid buffer handle")
            .write(data);
    }

    /// Writes a single [`ShaderType`](crate::ShaderType) value into a buffer
    /// with automatic GPU-layout padding.
    pub fn write_shader<T: crate::ShaderType>(&self, handle: Buffer, value: &T) {
        self.resources
            .get_buffer(handle.0)
            .expect("write_shader: invalid buffer handle")
            .write_shader(value);
    }

    // ── Convenience buffer methods ───────────────────────────────────

    fn host_buffer(
        &mut self,
        label: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer, GraphError> {
        Ok(self.create_buffer(&BufferDesc {
            size,
            usage: usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            location: MemoryLocation::CpuToGpu,
            label: label.to_string(),
        })?)
    }

    fn host_buffer_with_shader<T: crate::ShaderType>(
        &mut self,
        label: &str,
        value: &T,
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer, GraphError> {
        let buf = self.host_buffer(label, T::PADDED_SIZE as vk::DeviceSize, usage)?;
        self.write_shader(buf, value);
        Ok(buf)
    }

    fn host_buffer_with_data<T: bytemuck::Pod>(
        &mut self,
        label: &str,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer, GraphError> {
        let buf = self.host_buffer(label, std::mem::size_of_val(data) as vk::DeviceSize, usage)?;
        self.write_buffer(buf, data);
        Ok(buf)
    }

    /// Allocates a `STORAGE_BUFFER` pre-filled with `data` in `CpuToGpu` memory.
    ///
    /// Includes `SHADER_DEVICE_ADDRESS` automatically. Retrieve the GPU pointer
    /// with [`Graph::buffer_device_address`] to pass it to shaders via push constants.
    pub fn storage_buffer<T: bytemuck::Pod>(
        &mut self,
        label: &str,
        data: &[T],
    ) -> Result<Buffer, GraphError> {
        self.host_buffer_with_data(label, data, vk::BufferUsageFlags::STORAGE_BUFFER)
    }

    /// Allocates a `STORAGE_BUFFER` pre-filled with a [`ShaderType`](crate::ShaderType)
    /// value (with automatic padding) in `CpuToGpu` memory.
    pub fn storage_shader<T: crate::ShaderType>(
        &mut self,
        label: &str,
        value: &T,
    ) -> Result<Buffer, GraphError> {
        self.host_buffer_with_shader(label, value, vk::BufferUsageFlags::STORAGE_BUFFER)
    }

    /// Allocates an uninitialised `STORAGE_BUFFER` of `size` bytes in `CpuToGpu` memory.
    ///
    /// Use [`Graph::write_buffer`] to fill it from the CPU before first use.
    pub fn storage_buffer_empty(
        &mut self,
        label: &str,
        size: vk::DeviceSize,
    ) -> Result<Buffer, GraphError> {
        self.host_buffer(label, size, vk::BufferUsageFlags::STORAGE_BUFFER)
    }

    /// Allocates a `UNIFORM_BUFFER` pre-filled with `data` in `CpuToGpu` memory.
    ///
    /// Includes `SHADER_DEVICE_ADDRESS` automatically. Update it each frame with
    /// [`Graph::write_buffer`].
    pub fn uniform_buffer<T: bytemuck::Pod>(
        &mut self,
        label: &str,
        data: &[T],
    ) -> Result<Buffer, GraphError> {
        self.host_buffer_with_data(label, data, vk::BufferUsageFlags::UNIFORM_BUFFER)
    }

    /// Allocates a `UNIFORM_BUFFER` pre-filled with a [`ShaderType`](crate::ShaderType)
    /// value (with automatic padding) in `CpuToGpu` memory.
    ///
    /// Update it each frame with [`Graph::write_shader`].
    pub fn uniform_shader<T: crate::ShaderType>(
        &mut self,
        label: &str,
        value: &T,
    ) -> Result<Buffer, GraphError> {
        self.host_buffer_with_shader(label, value, vk::BufferUsageFlags::UNIFORM_BUFFER)
    }

    /// Allocates an uninitialised `UNIFORM_BUFFER` of `size` bytes in `CpuToGpu` memory.
    pub fn uniform_buffer_empty(
        &mut self,
        label: &str,
        size: vk::DeviceSize,
    ) -> Result<Buffer, GraphError> {
        self.host_buffer(label, size, vk::BufferUsageFlags::UNIFORM_BUFFER)
    }

    /// Allocates a `VERTEX_BUFFER` pre-filled with `data` in `GpuOnly` memory.
    ///
    /// Data is transferred via a temporary staging buffer using a synchronous
    /// one-shot submit — intended for static geometry loaded once at startup.
    pub fn vertex_buffer<T: bytemuck::Pod>(
        &mut self,
        label: &str,
        data: &[T],
    ) -> Result<Buffer, GraphError> {
        self.upload_buffer_labeled(
            bytemuck::cast_slice(data),
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            label,
        )
    }

    /// Allocates an `INDEX_BUFFER` pre-filled with `data` in `GpuOnly` memory.
    ///
    /// Like [`vertex_buffer`](Graph::vertex_buffer), data is uploaded via a
    /// synchronous staging transfer. `T` is typically `u16` or `u32`.
    pub fn index_buffer(&mut self, label: &str, data: &[u32]) -> Result<Buffer, GraphError> {
        self.upload_buffer_labeled(
            bytemuck::cast_slice(data),
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            label,
        )
    }

    /// Allocates an uninitialised `VERTEX_BUFFER` of `size` bytes in `CpuToGpu` memory.
    ///
    /// Use [`Graph::write_buffer`] to fill it when the geometry is ready.
    /// Intended for pre-allocating chunk buffers at maximum capacity.
    pub fn vertex_buffer_dynamic_empty(
        &mut self,
        label: &str,
        size: vk::DeviceSize,
    ) -> Result<Buffer, GraphError> {
        self.host_buffer(label, size, vk::BufferUsageFlags::VERTEX_BUFFER)
    }

    /// Allocates an uninitialised `INDEX_BUFFER` of `size` bytes in `CpuToGpu` memory.
    ///
    /// Use [`Graph::write_buffer`] to fill it when the geometry is ready.
    pub fn index_buffer_dynamic_empty(
        &mut self,
        label: &str,
        size: vk::DeviceSize,
    ) -> Result<Buffer, GraphError> {
        self.host_buffer(label, size, vk::BufferUsageFlags::INDEX_BUFFER)
    }

    /// Allocates a `VERTEX_BUFFER` pre-filled with `data` in `CpuToGpu` memory.
    ///
    /// Unlike [`vertex_buffer`](Graph::vertex_buffer), no staging is used — the buffer
    /// is directly writable from the CPU via [`Graph::write_buffer`]. Intended for
    /// geometry that changes frequently (e.g. dynamic chunks).
    pub fn vertex_buffer_dynamic<T: bytemuck::Pod>(
        &mut self,
        label: &str,
        data: &[T],
    ) -> Result<Buffer, GraphError> {
        self.host_buffer_with_data(label, data, vk::BufferUsageFlags::VERTEX_BUFFER)
    }

    /// Allocates an `INDEX_BUFFER` pre-filled with `data` in `CpuToGpu` memory.
    ///
    /// Like [`vertex_buffer_dynamic`](Graph::vertex_buffer_dynamic), directly writable
    /// from the CPU via [`Graph::write_buffer`]. `T` is typically `u16` or `u32`.
    pub fn index_buffer_dynamic(
        &mut self,
        label: &str,
        data: &[u32],
    ) -> Result<Buffer, GraphError> {
        self.host_buffer_with_data(label, data, vk::BufferUsageFlags::INDEX_BUFFER)
    }

    pub fn create_sampler(&mut self) -> super::sampler::SamplerBuilder<'_> {
        super::sampler::SamplerBuilder::new(self)
    }

    pub fn destroy_sampler(&mut self, sampler: Sampler) {
        self.resources
            .destroy_sampler(self.device.ash_device(), sampler.handle);
    }

    pub(in crate::graph) fn upload_image_data(
        &mut self,
        dst: ImageHandle,
        pixels: &[u8],
        extent: vk::Extent3D,
        mip_levels: u32,
    ) -> Result<(), GraphError> {
        let staging = self.create_staging(pixels, "staging_upload")?;

        let vk_img = self
            .resources
            .get_image(dst)
            .expect("image just created")
            .raw;
        let stg_buf = self
            .resources
            .get_buffer(staging)
            .expect("buffer just created")
            .raw;

        self.one_shot_submit(|cmd| {
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
        })?;

        self.destroy_staging(staging);
        Ok(())
    }
}

#[inline]
fn compute_mip_levels(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2().floor() as u32 + 1
}
