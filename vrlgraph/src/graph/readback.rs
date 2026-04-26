//! GPU → CPU readback of images and buffers.
//!
//! See [`FrameBuilder::readback_image`](crate::graph::FrameBuilder::readback_image)
//! and [`FrameBuilder::readback_buffer`](crate::graph::FrameBuilder::readback_buffer)
//! for the entry points.

use std::sync::{Arc, Mutex, OnceLock};

use ash::vk;

use crate::resource::BufferHandle;

use super::Graph;

/// Rounds `value` up to the next multiple of `alignment`.
///
/// `alignment` is assumed to be a power of two. `align_up(x, 1) == x`,
/// `align_up(x, 0)` is treated as `align_up(x, 1)` for safety.
#[inline]
pub(crate) fn align_up(value: u32, alignment: u32) -> u32 {
    if alignment <= 1 {
        return value;
    }
    let mask = alignment - 1;
    value.checked_add(mask).map(|v| v & !mask).unwrap_or(value)
}

/// Returns the bytes-per-pixel of `format` if it is a uniform color format
/// supported by readback, or `None` for depth/stencil/compressed/planar formats.
pub(crate) fn bytes_per_pixel(format: vk::Format) -> Option<u32> {
    use vk::Format as F;
    Some(match format {
        F::R8_UNORM
        | F::R8_SNORM
        | F::R8_UINT
        | F::R8_SINT
        | F::R8_SRGB
        | F::R8_USCALED
        | F::R8_SSCALED => 1,
        F::R8G8_UNORM
        | F::R8G8_SNORM
        | F::R8G8_UINT
        | F::R8G8_SINT
        | F::R8G8_SRGB
        | F::R8G8_USCALED
        | F::R8G8_SSCALED => 2,
        F::R8G8B8A8_UNORM
        | F::R8G8B8A8_SNORM
        | F::R8G8B8A8_UINT
        | F::R8G8B8A8_SINT
        | F::R8G8B8A8_SRGB
        | F::R8G8B8A8_USCALED
        | F::R8G8B8A8_SSCALED => 4,
        F::B8G8R8A8_UNORM
        | F::B8G8R8A8_SNORM
        | F::B8G8R8A8_UINT
        | F::B8G8R8A8_SINT
        | F::B8G8R8A8_SRGB
        | F::B8G8R8A8_USCALED
        | F::B8G8R8A8_SSCALED => 4,
        F::A8B8G8R8_UNORM_PACK32
        | F::A8B8G8R8_SNORM_PACK32
        | F::A8B8G8R8_UINT_PACK32
        | F::A8B8G8R8_SINT_PACK32
        | F::A8B8G8R8_SRGB_PACK32 => 4,
        F::A2R10G10B10_UNORM_PACK32
        | F::A2R10G10B10_SNORM_PACK32
        | F::A2R10G10B10_UINT_PACK32
        | F::A2R10G10B10_SINT_PACK32
        | F::A2B10G10R10_UNORM_PACK32
        | F::A2B10G10R10_SNORM_PACK32
        | F::A2B10G10R10_UINT_PACK32
        | F::A2B10G10R10_SINT_PACK32 => 4,
        F::B10G11R11_UFLOAT_PACK32 | F::E5B9G9R9_UFLOAT_PACK32 => 4,
        F::R16_UNORM | F::R16_SNORM | F::R16_UINT | F::R16_SINT | F::R16_SFLOAT => 2,
        F::R16G16_UNORM | F::R16G16_SNORM | F::R16G16_UINT | F::R16G16_SINT | F::R16G16_SFLOAT => 4,
        F::R16G16B16A16_UNORM
        | F::R16G16B16A16_SNORM
        | F::R16G16B16A16_UINT
        | F::R16G16B16A16_SINT
        | F::R16G16B16A16_SFLOAT => 8,
        F::R32_UINT | F::R32_SINT | F::R32_SFLOAT => 4,
        F::R32G32_UINT | F::R32G32_SINT | F::R32G32_SFLOAT => 8,
        F::R32G32B32A32_UINT | F::R32G32B32A32_SINT | F::R32G32B32A32_SFLOAT => 16,
        F::R64_UINT | F::R64_SINT | F::R64_SFLOAT => 8,
        F::R64G64_UINT | F::R64G64_SINT | F::R64G64_SFLOAT => 16,
        F::R64G64B64A64_UINT | F::R64G64B64A64_SINT | F::R64G64B64A64_SFLOAT => 32,
        F::R5G6B5_UNORM_PACK16
        | F::B5G6R5_UNORM_PACK16
        | F::R5G5B5A1_UNORM_PACK16
        | F::B5G5R5A1_UNORM_PACK16
        | F::A1R5G5B5_UNORM_PACK16
        | F::R4G4B4A4_UNORM_PACK16
        | F::B4G4R4A4_UNORM_PACK16 => 2,
        F::R4G4_UNORM_PACK8 => 1,
        _ => return None,
    })
}

#[derive(Clone, Copy)]
pub(crate) struct ImageMeta {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub row_pitch: u32,
}

pub(crate) struct DeferredReadbackFree {
    pub buffer: BufferHandle,
    pub cycles_remaining: u32,
}

pub(crate) struct ReadbackInner {
    pub buffer: BufferHandle,
    pub data_size: vk::DeviceSize,
    pub mapped_ptr: *mut u8,
    pub memory: vk::DeviceMemory,
    pub memory_offset: vk::DeviceSize,
    pub device: ash::Device,
    pub fence: OnceLock<vk::Fence>,
    pub free_queue: Arc<Mutex<Vec<DeferredReadbackFree>>>,
    pub cycles_after_submit: u32,
    pub image_meta: Option<ImageMeta>,
}

unsafe impl Send for ReadbackInner {}
unsafe impl Sync for ReadbackInner {}

impl ReadbackInner {
    fn fence_signaled(&self) -> bool {
        let Some(&fence) = self.fence.get() else {
            return false;
        };
        unsafe { self.device.get_fence_status(fence).unwrap_or(false) }
    }

    fn block_until_ready(&self) {
        let &fence = self
            .fence
            .get()
            .expect("readback handle: frame.submit() was never called or failed");
        unsafe {
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("wait_for_fences failed");
        }
    }

    fn invalidate(&self) {
        let range = vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(self.memory_offset)
            .size(vk::WHOLE_SIZE);
        unsafe {
            let _ = self.device.invalidate_mapped_memory_ranges(&[range]);
        }
    }

    fn data_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.mapped_ptr, self.data_size as usize) }
    }
}

impl Drop for ReadbackInner {
    fn drop(&mut self) {
        if let Ok(mut q) = self.free_queue.lock() {
            q.push(DeferredReadbackFree {
                buffer: self.buffer,
                cycles_remaining: self.cycles_after_submit,
            });
        }
    }
}

/// Handle to a pending image readback.
///
/// Returned by [`FrameBuilder::readback_image`](crate::graph::FrameBuilder::readback_image).
/// Cheap to clone (`Arc` internally). Use [`is_ready`](Self::is_ready) /
/// [`try_get`](Self::try_get) to poll without blocking, or
/// [`wait`](Self::wait) to block until the frame's GPU work completes.
///
/// The staging buffer is freed automatically once the GPU has released it
/// **and** the last clone of this handle is dropped. Dropping a handle
/// before the GPU is done is safe; the free is deferred internally.
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
///
/// let data = shot.wait(graph);
/// // data.bytes is row_pitch*height long — pack rows for tight RGBA buffers
/// # let _ = data;
/// # Ok(()) }
/// ```
#[derive(Clone)]
#[must_use = "drop without calling wait/try_get to discard the readback data"]
pub struct ImageReadback {
    pub(crate) inner: Arc<ReadbackInner>,
}

/// Borrowed view into a completed image readback.
pub struct ImageReadbackData<'a> {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Source format (matches the read image).
    pub format: vk::Format,
    /// Bytes per row of the staging buffer.
    ///
    /// May exceed `width * bytes_per_pixel(format)` because Vulkan requires
    /// rows to be aligned to `optimalBufferCopyRowPitchAlignment`. Always
    /// honour this when iterating rows; do not assume tight packing.
    pub row_pitch: u32,
    /// Raw pixel bytes, length `row_pitch * height`.
    pub bytes: &'a [u8],
}

impl ImageReadback {
    /// Non-blocking poll. Returns `true` once the GPU has finished writing
    /// the staging buffer.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.inner.fence_signaled()
    }

    /// Returns a borrowed view of the data if ready, or `None` if the GPU
    /// has not finished yet.
    pub fn try_get(&self) -> Option<ImageReadbackData<'_>> {
        if !self.is_ready() {
            return None;
        }
        self.inner.invalidate();
        let meta = self
            .inner
            .image_meta
            .expect("ImageReadback created without image metadata");
        Some(ImageReadbackData {
            width: meta.width,
            height: meta.height,
            format: meta.format,
            row_pitch: meta.row_pitch,
            bytes: self.inner.data_slice(),
        })
    }

    /// Blocks until the frame's GPU work completes, then returns the data.
    ///
    /// The `graph` reference ensures the graph is alive for the duration of
    /// the borrow. The returned slice borrows from `self`; drop the data
    /// before re-using the readback handle.
    ///
    /// # Panics
    ///
    /// Panics if the readback was never submitted (the [`FrameBuilder`](crate::graph::FrameBuilder)
    /// was dropped without calling `submit()`).
    pub fn wait<'a>(&'a self, _graph: &'a Graph) -> ImageReadbackData<'a> {
        self.inner.block_until_ready();
        self.inner.invalidate();
        let meta = self
            .inner
            .image_meta
            .expect("ImageReadback created without image metadata");
        ImageReadbackData {
            width: meta.width,
            height: meta.height,
            format: meta.format,
            row_pitch: meta.row_pitch,
            bytes: self.inner.data_slice(),
        }
    }
}

/// Handle to a pending buffer readback.
///
/// Returned by [`FrameBuilder::readback_buffer`](crate::graph::FrameBuilder::readback_buffer).
/// Cheap to clone (`Arc` internally). Use [`is_ready`](Self::is_ready) /
/// [`try_get`](Self::try_get) to poll without blocking, or
/// [`wait`](Self::wait) to block until the frame's GPU work completes.
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
///
/// let counts: &[u32] = bins.wait_as::<u32>(graph);
/// # let _ = counts;
/// # Ok(()) }
/// ```
#[derive(Clone)]
#[must_use = "drop without calling wait/try_get to discard the readback data"]
pub struct BufferReadback {
    pub(crate) inner: Arc<ReadbackInner>,
}

impl BufferReadback {
    /// Non-blocking poll. Returns `true` once the GPU has finished writing
    /// the staging buffer.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.inner.fence_signaled()
    }

    /// Returns the raw bytes if ready, or `None` if the GPU has not finished yet.
    pub fn try_get(&self) -> Option<&[u8]> {
        if !self.is_ready() {
            return None;
        }
        self.inner.invalidate();
        Some(self.inner.data_slice())
    }

    /// Blocks until the frame's GPU work completes, then returns the raw bytes.
    pub fn wait<'a>(&'a self, _graph: &'a Graph) -> &'a [u8] {
        self.inner.block_until_ready();
        self.inner.invalidate();
        self.inner.data_slice()
    }

    /// Typed view of the bytes via [`bytemuck::cast_slice`]. Returns `None`
    /// if the GPU has not finished yet.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length is not a multiple of `size_of::<T>()` or
    /// if the alignment is incompatible.
    pub fn try_get_as<T: bytemuck::Pod>(&self) -> Option<&[T]> {
        self.try_get().map(bytemuck::cast_slice)
    }

    /// Blocking typed view via [`bytemuck::cast_slice`].
    ///
    /// # Panics
    ///
    /// Panics if the buffer length is not a multiple of `size_of::<T>()` or
    /// if the alignment is incompatible.
    pub fn wait_as<'a, T: bytemuck::Pod>(&'a self, graph: &'a Graph) -> &'a [T] {
        bytemuck::cast_slice(self.wait(graph))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_zero_alignment() {
        assert_eq!(align_up(7, 0), 7);
    }

    #[test]
    fn align_up_alignment_one() {
        assert_eq!(align_up(7, 1), 7);
    }

    #[test]
    fn align_up_powers_of_two() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 4), 8);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    #[test]
    fn bytes_per_pixel_common_color_formats() {
        assert_eq!(bytes_per_pixel(vk::Format::R8_UNORM), Some(1));
        assert_eq!(bytes_per_pixel(vk::Format::R8G8_UNORM), Some(2));
        assert_eq!(bytes_per_pixel(vk::Format::R8G8B8A8_UNORM), Some(4));
        assert_eq!(bytes_per_pixel(vk::Format::R8G8B8A8_SRGB), Some(4));
        assert_eq!(bytes_per_pixel(vk::Format::B8G8R8A8_UNORM), Some(4));
        assert_eq!(bytes_per_pixel(vk::Format::B8G8R8A8_SRGB), Some(4));
        assert_eq!(bytes_per_pixel(vk::Format::R16_SFLOAT), Some(2));
        assert_eq!(bytes_per_pixel(vk::Format::R16G16_SFLOAT), Some(4));
        assert_eq!(bytes_per_pixel(vk::Format::R16G16B16A16_SFLOAT), Some(8));
        assert_eq!(bytes_per_pixel(vk::Format::R32_SFLOAT), Some(4));
        assert_eq!(bytes_per_pixel(vk::Format::R32G32B32A32_SFLOAT), Some(16));
    }

    #[test]
    fn bytes_per_pixel_packed_formats() {
        assert_eq!(
            bytes_per_pixel(vk::Format::A2B10G10R10_UNORM_PACK32),
            Some(4)
        );
        assert_eq!(
            bytes_per_pixel(vk::Format::B10G11R11_UFLOAT_PACK32),
            Some(4)
        );
        assert_eq!(bytes_per_pixel(vk::Format::R5G6B5_UNORM_PACK16), Some(2));
    }

    #[test]
    fn bytes_per_pixel_rejects_depth() {
        assert_eq!(bytes_per_pixel(vk::Format::D16_UNORM), None);
        assert_eq!(bytes_per_pixel(vk::Format::D32_SFLOAT), None);
        assert_eq!(bytes_per_pixel(vk::Format::D24_UNORM_S8_UINT), None);
        assert_eq!(bytes_per_pixel(vk::Format::D32_SFLOAT_S8_UINT), None);
    }

    #[test]
    fn bytes_per_pixel_rejects_compressed() {
        assert_eq!(bytes_per_pixel(vk::Format::BC1_RGB_UNORM_BLOCK), None);
        assert_eq!(bytes_per_pixel(vk::Format::BC7_UNORM_BLOCK), None);
        assert_eq!(bytes_per_pixel(vk::Format::BC7_SRGB_BLOCK), None);
        assert_eq!(bytes_per_pixel(vk::Format::ASTC_4X4_UNORM_BLOCK), None);
    }

    #[test]
    fn bytes_per_pixel_rejects_stencil_only() {
        assert_eq!(bytes_per_pixel(vk::Format::S8_UINT), None);
    }
}
