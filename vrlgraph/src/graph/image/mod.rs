mod builder;
mod texture;

pub use builder::ImageBuilder;
pub use texture::TextureBuilder;

use ash::vk;

use super::bindless::{Array2D, BindlessIndex, Cubemap, Sampled, Storage};
use crate::resource::{ImageDesc, ImageHandle, ResourcePool};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Image(pub(crate) u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ImageOrigin {
    Transient,

    Persistent,

    External,
}

pub(crate) struct ImageEntry {
    pub desc: ImageDesc,
    pub origin: ImageOrigin,

    pub handle: Option<ImageHandle>,

    pub external: Option<(vk::Image, vk::ImageView)>,
    pub aspect: vk::ImageAspectFlags,

    pub usage: vk::ImageUsageFlags,

    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,

    pub sampled_index: Option<BindlessIndex<Sampled>>,
    pub storage_index: Option<BindlessIndex<Storage>>,
    pub cubemap_index: Option<BindlessIndex<Cubemap>>,
    pub array_index: Option<BindlessIndex<Array2D>>,
}

impl Default for ImageEntry {
    fn default() -> Self {
        Self {
            desc: ImageDesc::default(),
            origin: ImageOrigin::Transient,
            handle: None,
            external: None,
            aspect: vk::ImageAspectFlags::COLOR,
            usage: vk::ImageUsageFlags::empty(),
            layout: vk::ImageLayout::UNDEFINED,
            stage: vk::PipelineStageFlags2::NONE,
            access: vk::AccessFlags2::NONE,
            sampled_index: None,
            storage_index: None,
            cubemap_index: None,
            array_index: None,
        }
    }
}

impl ImageEntry {
    pub(crate) fn transient(desc: ImageDesc) -> Self {
        let aspect = aspect_from_format(desc.format);
        Self {
            desc,
            aspect,
            ..Default::default()
        }
    }

    pub(crate) fn persistent(desc: ImageDesc) -> Self {
        let aspect = aspect_from_format(desc.format);
        let usage = desc.usage;
        Self {
            desc,
            origin: ImageOrigin::Persistent,
            aspect,
            usage,
            ..Default::default()
        }
    }

    pub(crate) fn loaded(desc: ImageDesc, handle: ImageHandle) -> Self {
        let aspect = aspect_from_format(desc.format);
        Self {
            desc,
            origin: ImageOrigin::Persistent,
            handle: Some(handle),
            aspect,
            usage: vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
            ..Default::default()
        }
    }

    pub(crate) fn external(raw: vk::Image, view: vk::ImageView, extent: vk::Extent2D) -> Self {
        Self {
            desc: ImageDesc {
                extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                format: vk::Format::UNDEFINED,
                ..Default::default()
            },
            origin: ImageOrigin::External,
            external: Some((raw, view)),
            ..Default::default()
        }
    }

    pub(crate) fn resolve(&self, pool: &ResourcePool) -> (vk::Image, vk::ImageView) {
        if let Some(ext) = self.external {
            return ext;
        }
        let h = self.handle.expect("image referenced before allocation");
        let img = pool.get_image(h).expect("image destroyed");
        (img.raw, img.view)
    }

    pub(crate) fn view(&self, pool: &ResourcePool) -> vk::ImageView {
        self.resolve(pool).1
    }

    pub(crate) fn layer_count(&self) -> u32 {
        self.desc.kind.array_layers()
    }
}

pub(crate) fn aspect_from_format(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }

        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,

        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }

        _ => vk::ImageAspectFlags::COLOR,
    }
}

pub(crate) struct ResizableTemplate {
    pub desc: ImageDesc,
}

#[inline]
pub(crate) fn compute_mip_levels(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2().floor() as u32 + 1
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::aspect_from_format;

    #[test]
    fn depth_only_formats() {
        for fmt in [
            vk::Format::D16_UNORM,
            vk::Format::D32_SFLOAT,
            vk::Format::X8_D24_UNORM_PACK32,
        ] {
            assert_eq!(aspect_from_format(fmt), vk::ImageAspectFlags::DEPTH);
        }
    }

    #[test]
    fn stencil_only_format() {
        assert_eq!(
            aspect_from_format(vk::Format::S8_UINT),
            vk::ImageAspectFlags::STENCIL
        );
    }

    #[test]
    fn depth_stencil_formats() {
        for fmt in [
            vk::Format::D16_UNORM_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
            vk::Format::D32_SFLOAT_S8_UINT,
        ] {
            assert_eq!(
                aspect_from_format(fmt),
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            );
        }
    }

    #[test]
    fn color_format() {
        assert_eq!(
            aspect_from_format(vk::Format::R8G8B8A8_UNORM),
            vk::ImageAspectFlags::COLOR
        );
    }

    #[test]
    fn mip_levels_1x1() {
        assert_eq!(super::compute_mip_levels(1, 1), 1);
    }

    #[test]
    fn mip_levels_power_of_two() {
        assert_eq!(super::compute_mip_levels(2, 2), 2);
        assert_eq!(super::compute_mip_levels(256, 256), 9);
        assert_eq!(super::compute_mip_levels(1024, 512), 11);
    }

    #[test]
    fn mip_levels_non_power_of_two() {
        assert_eq!(super::compute_mip_levels(300, 200), 9);
        assert_eq!(super::compute_mip_levels(1, 1024), 11);
    }
}
