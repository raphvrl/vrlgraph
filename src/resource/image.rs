use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use super::ResourceError;
use crate::types::SampleCount;

/// Dimensionality and layer structure of an image.
///
/// Used in [`ImageDesc`] to declare the shape of an image. The graph
/// creates the correct `VkImageViewType` based on this value.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ImageKind {
    /// A standard 2D image. This is the default.
    #[default]
    Image2D,
    /// An array of 2D images with `layers` slices. Useful for shadow atlases
    /// or texture arrays.
    Image2DArray { layers: u32 },
    /// A 6-face cubemap. The 6 layers are ordered +X, -X, +Y, -Y, +Z, -Z.
    Cubemap,
    /// An array of `count` cubemaps, for a total of `count * 6` layers.
    CubemapArray { count: u32 },
}

impl ImageKind {
    pub fn array_layers(&self) -> u32 {
        match self {
            Self::Image2D => 1,
            Self::Image2DArray { layers } => *layers,
            Self::Cubemap => 6,
            Self::CubemapArray { count } => count * 6,
        }
    }

    pub fn vk_view_type(&self) -> vk::ImageViewType {
        match self {
            Self::Image2D => vk::ImageViewType::TYPE_2D,
            Self::Image2DArray { .. } => vk::ImageViewType::TYPE_2D_ARRAY,
            Self::Cubemap => vk::ImageViewType::CUBE,
            Self::CubemapArray { .. } => vk::ImageViewType::CUBE_ARRAY,
        }
    }

    fn create_flags(&self) -> vk::ImageCreateFlags {
        match self {
            Self::Cubemap | Self::CubemapArray { .. } => vk::ImageCreateFlags::CUBE_COMPATIBLE,
            _ => vk::ImageCreateFlags::empty(),
        }
    }
}

/// Description of a GPU image.
///
/// Passed to [`Graph::create_transient`](crate::graph::Graph::create_transient),
/// [`Graph::create_persistent`](crate::graph::Graph::create_persistent), and
/// [`Graph::create_resizable`](crate::graph::Graph::create_resizable).
///
/// The graph infers the minimum required `VkImageUsageFlags` from how the image
/// is used in declared passes. Only set `usage` when you need additional flags
/// that the graph cannot infer — for example `SAMPLED` on an image that is
/// only written as a color attachment but also sampled in a descriptor set.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageDesc {
    /// Width, height, and depth of the image. Depth must be `1` for 2D images.
    pub extent: vk::Extent3D,
    /// Pixel format.
    pub format: vk::Format,
    /// Number of mip levels. Use `1` unless you generate mipmaps manually.
    pub mip_levels: u32,
    /// MSAA sample count. Use `SampleCount::S1` for non-multisampled images.
    pub samples: SampleCount,
    /// Dimensionality and layer structure. Defaults to [`ImageKind::Image2D`].
    pub kind: ImageKind,
    /// Debug label shown in validation output and GPU debuggers.
    pub label: String,
    /// Additional Vulkan usage flags beyond what the graph infers from pass
    /// accesses. Leave empty in most cases.
    pub usage: vk::ImageUsageFlags,
}

impl Default for ImageDesc {
    fn default() -> Self {
        Self {
            extent: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            format: vk::Format::R8G8B8A8_UNORM,
            mip_levels: 1,
            samples: SampleCount::S1,
            kind: ImageKind::Image2D,
            label: String::new(),
            usage: vk::ImageUsageFlags::empty(),
        }
    }
}

pub struct GpuImage {
    pub raw: vk::Image,
    pub view: vk::ImageView,
    pub layer_views: Vec<vk::ImageView>,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub mip_levels: u32,
    pub layer_count: u32,
    allocation: Allocation,
}

impl GpuImage {
    pub(super) fn create(
        device: &ash::Device,
        allocator: &mut Allocator,
        desc: &ImageDesc,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> Result<Self, ResourceError> {
        let layer_count = desc.kind.array_layers();

        let create_info = vk::ImageCreateInfo::default()
            .flags(desc.kind.create_flags())
            .image_type(vk::ImageType::TYPE_2D)
            .extent(desc.extent)
            .mip_levels(desc.mip_levels)
            .array_layers(layer_count)
            .format(desc.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(desc.samples.into())
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let raw = unsafe { device.create_image(&create_info, None)? };

        let requirements = unsafe { device.get_image_memory_requirements(raw) };

        let allocation = match allocator.allocate(&AllocationCreateDesc {
            name: &desc.label,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => {
                unsafe { device.destroy_image(raw, None) };
                return Err(ResourceError::Allocation(e));
            }
        };

        if let Err(e) =
            unsafe { device.bind_image_memory(raw, allocation.memory(), allocation.offset()) }
        {
            allocator.free(allocation).ok();
            unsafe { device.destroy_image(raw, None) };
            return Err(ResourceError::Vulkan(e));
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(raw)
            .view_type(desc.kind.vk_view_type())
            .format(desc.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect)
                    .base_mip_level(0)
                    .level_count(desc.mip_levels)
                    .base_array_layer(0)
                    .layer_count(layer_count),
            );

        let view = match unsafe { device.create_image_view(&view_info, None) } {
            Ok(v) => v,
            Err(e) => {
                allocator.free(allocation).ok();
                unsafe { device.destroy_image(raw, None) };
                return Err(ResourceError::Vulkan(e));
            }
        };

        let layer_views = if layer_count > 1 {
            let mut views = Vec::with_capacity(layer_count as usize);
            for layer in 0..layer_count {
                let lv_info = vk::ImageViewCreateInfo::default()
                    .image(raw)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(desc.format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(aspect)
                            .base_mip_level(0)
                            .level_count(desc.mip_levels)
                            .base_array_layer(layer)
                            .layer_count(1),
                    );
                match unsafe { device.create_image_view(&lv_info, None) } {
                    Ok(v) => views.push(v),
                    Err(e) => {
                        for v in views {
                            unsafe { device.destroy_image_view(v, None) };
                        }
                        unsafe { device.destroy_image_view(view, None) };
                        allocator.free(allocation).ok();
                        unsafe { device.destroy_image(raw, None) };
                        return Err(ResourceError::Vulkan(e));
                    }
                }
            }
            views
        } else {
            Vec::new()
        };

        Ok(Self {
            raw,
            view,
            layer_views,
            extent: desc.extent,
            format: desc.format,
            mip_levels: desc.mip_levels,
            layer_count,
            allocation,
        })
    }

    pub(super) fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        for lv in self.layer_views {
            unsafe { device.destroy_image_view(lv, None) };
        }
        unsafe { device.destroy_image_view(self.view, None) };
        unsafe { device.destroy_image(self.raw, None) };
        if let Err(e) = allocator.free(self.allocation) {
            tracing::error!("failed to free image allocation: {e}");
        }
    }
}
