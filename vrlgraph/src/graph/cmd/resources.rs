use ash::vk;

use super::Cmd;
use crate::graph::bindless::Sampler;
use crate::graph::image::Image;
use crate::resource::{
    AsyncBuffer, Buffer, GpuBuffer, GpuImage, GpuPipeline, Pipeline, StreamingBuffer,
};

impl<'a> Cmd<'a> {
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
