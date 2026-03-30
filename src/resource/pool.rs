use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;
use slotmap::SlotMap;
use smallvec::SmallVec;

use super::ResourceError;
use super::buffer::{BufferDesc, GpuBuffer};
use super::handle::{BufferHandle, ImageHandle, PipelineHandle, SamplerHandle, ShaderModuleHandle};
use super::image::{GpuImage, ImageDesc};
use super::pipeline::GpuPipeline;
use super::shader::GpuShaderModule;
use super::streaming::{StreamingBuffer, StreamingBufferHandle};

pub(crate) struct ResourcePool {
    buffers: SlotMap<BufferHandle, GpuBuffer>,
    images: SlotMap<ImageHandle, GpuImage>,
    pipelines: SlotMap<PipelineHandle, GpuPipeline>,
    samplers: SlotMap<SamplerHandle, vk::Sampler>,
    shader_modules: SlotMap<ShaderModuleHandle, GpuShaderModule>,
    streaming_buffers: SlotMap<StreamingBufferHandle, StreamingBuffer>,
}

impl ResourcePool {
    pub(crate) fn new() -> Self {
        Self {
            buffers: SlotMap::with_key(),
            images: SlotMap::with_key(),
            pipelines: SlotMap::with_key(),
            samplers: SlotMap::with_key(),
            shader_modules: SlotMap::with_key(),
            streaming_buffers: SlotMap::with_key(),
        }
    }

    pub(crate) fn create_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        desc: &BufferDesc,
    ) -> Result<BufferHandle, ResourceError> {
        let buf = GpuBuffer::create(device, allocator, desc)?;
        Ok(self.buffers.insert(buf))
    }

    pub(crate) fn get_buffer(&self, handle: BufferHandle) -> Option<&GpuBuffer> {
        self.buffers.get(handle)
    }

    pub(crate) fn destroy_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        handle: BufferHandle,
    ) {
        if let Some(buf) = self.buffers.remove(handle) {
            buf.destroy(device, allocator);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_streaming_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        label: &str,
        frames_in_flight: usize,
    ) -> Result<StreamingBufferHandle, ResourceError> {
        let mut slots: SmallVec<[BufferHandle; 3]> = SmallVec::new();
        for i in 0..frames_in_flight {
            let desc = BufferDesc {
                size,
                usage,
                location,
                label: format!("{label}[{i}]"),
            };
            slots.push(self.create_buffer(device, allocator, &desc)?);
        }
        Ok(self.streaming_buffers.insert(StreamingBuffer::new(slots)))
    }

    pub(crate) fn streaming_slot(
        &self,
        handle: StreamingBufferHandle,
        frame_index: usize,
    ) -> Option<BufferHandle> {
        self.streaming_buffers
            .get(handle)
            .map(|sb| sb.slot(frame_index))
    }

    pub(crate) fn destroy_streaming_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        handle: StreamingBufferHandle,
    ) {
        if let Some(sb) = self.streaming_buffers.remove(handle) {
            for slot in sb.slots {
                self.destroy_buffer(device, allocator, slot);
            }
        }
    }

    pub(crate) fn create_image(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        desc: &ImageDesc,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> Result<ImageHandle, ResourceError> {
        let img = GpuImage::create(device, allocator, desc, usage, aspect)?;
        Ok(self.images.insert(img))
    }

    pub(crate) fn get_image(&self, handle: ImageHandle) -> Option<&GpuImage> {
        self.images.get(handle)
    }

    pub(crate) fn destroy_image(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        handle: ImageHandle,
    ) {
        if let Some(img) = self.images.remove(handle) {
            img.destroy(device, allocator);
        }
    }

    pub(crate) fn insert_pipeline(&mut self, pipeline: GpuPipeline) -> PipelineHandle {
        self.pipelines.insert(pipeline)
    }

    pub(crate) fn get_pipeline(&self, handle: PipelineHandle) -> Option<&GpuPipeline> {
        self.pipelines.get(handle)
    }

    pub(crate) fn destroy_pipeline(&mut self, device: &ash::Device, handle: PipelineHandle) {
        if let Some(pipe) = self.pipelines.remove(handle) {
            pipe.destroy(device);
        }
    }

    #[cfg(debug_assertions)]
    pub(crate) fn replace_pipeline(
        &mut self,
        device: &ash::Device,
        handle: PipelineHandle,
        new: GpuPipeline,
    ) {
        if let Some(slot) = self.pipelines.get_mut(handle) {
            std::mem::replace(slot, new).destroy(device);
        }
    }

    pub(crate) fn insert_shader_module(&mut self, module: GpuShaderModule) -> ShaderModuleHandle {
        self.shader_modules.insert(module)
    }

    pub(crate) fn get_shader_module(&self, handle: ShaderModuleHandle) -> Option<&GpuShaderModule> {
        self.shader_modules.get(handle)
    }

    pub(crate) fn destroy_shader_module(&mut self, handle: ShaderModuleHandle) {
        self.shader_modules.remove(handle);
    }

    #[cfg(debug_assertions)]
    pub(crate) fn update_shader_module_vk(
        &mut self,
        handle: ShaderModuleHandle,
        module: vk::ShaderModule,
    ) {
        if let Some(entry) = self.shader_modules.get_mut(handle) {
            entry.module = module;
        }
    }

    pub(crate) fn create_sampler(
        &mut self,
        device: &ash::Device,
        info: &vk::SamplerCreateInfo,
    ) -> Result<SamplerHandle, ResourceError> {
        let sampler = unsafe { device.create_sampler(info, None)? };
        Ok(self.samplers.insert(sampler))
    }

    pub(crate) fn get_sampler(&self, handle: SamplerHandle) -> Option<vk::Sampler> {
        self.samplers.get(handle).copied()
    }

    pub(crate) fn destroy_sampler(&mut self, device: &ash::Device, handle: SamplerHandle) {
        if let Some(s) = self.samplers.remove(handle) {
            unsafe { device.destroy_sampler(s, None) };
        }
    }

    pub(crate) fn drain_buffers(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for (_, buf) in self.buffers.drain() {
            buf.destroy(device, allocator);
        }
    }

    pub(crate) fn drain_pipelines(&mut self, device: &ash::Device) {
        for (_, pipe) in self.pipelines.drain() {
            pipe.destroy(device);
        }
    }

    pub(crate) fn drain_shader_modules(&mut self) {
        self.shader_modules.clear();
    }

    pub(crate) fn drain_samplers(&mut self, device: &ash::Device) {
        for (_, s) in self.samplers.drain() {
            unsafe { device.destroy_sampler(s, None) };
        }
    }
}

impl Default for ResourcePool {
    fn default() -> Self {
        Self::new()
    }
}
