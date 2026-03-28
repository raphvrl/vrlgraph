use ash::vk;

use crate::resource::{ImageDesc, ImageHandle, ResourcePool};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GraphImage(pub(crate) u32);

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
}

impl ImageEntry {
    pub(crate) fn transient(desc: ImageDesc) -> Self {
        let aspect = aspect_from_format(desc.format);
        Self {
            desc,
            origin: ImageOrigin::Transient,
            handle: None,
            external: None,
            aspect,
            usage: vk::ImageUsageFlags::empty(),
            layout: vk::ImageLayout::UNDEFINED,
            stage: vk::PipelineStageFlags2::NONE,
            access: vk::AccessFlags2::NONE,
        }
    }

    pub(crate) fn persistent(desc: ImageDesc) -> Self {
        let aspect = aspect_from_format(desc.format);
        let usage = desc.usage;
        Self {
            desc,
            origin: ImageOrigin::Persistent,
            handle: None,
            external: None,
            aspect,
            usage,
            layout: vk::ImageLayout::UNDEFINED,
            stage: vk::PipelineStageFlags2::NONE,
            access: vk::AccessFlags2::NONE,
        }
    }

    pub(crate) fn loaded(desc: ImageDesc, handle: ImageHandle) -> Self {
        let aspect = aspect_from_format(desc.format);
        Self {
            desc,
            origin: ImageOrigin::Persistent,
            handle: Some(handle),
            external: None,
            aspect,

            usage: vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
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
            handle: None,
            external: Some((raw, view)),
            aspect: vk::ImageAspectFlags::COLOR,
            usage: vk::ImageUsageFlags::empty(),
            layout: vk::ImageLayout::UNDEFINED,
            stage: vk::PipelineStageFlags2::NONE,
            access: vk::AccessFlags2::NONE,
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
}
