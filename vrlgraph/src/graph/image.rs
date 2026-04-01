use ash::vk;

use super::bindless::{Array2D, BindlessIndex, Cubemap, Sampled, Storage};
use super::resources::register_bindless;
use super::{Graph, GraphError};
use crate::resource::{ImageDesc, ImageHandle, ImageKind, ResourcePool};
use crate::types::SampleCount;

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

pub struct ImageBuilder<'g> {
    graph: &'g mut Graph,
    origin: ImageOrigin,
    format: Option<vk::Format>,
    width: Option<u32>,
    height: Option<u32>,
    depth: u32,
    mip_levels: u32,
    samples: SampleCount,
    kind: ImageKind,
    label: String,
    usage: vk::ImageUsageFlags,
    resizable: bool,
}

impl<'g> ImageBuilder<'g> {
    pub(super) fn new(graph: &'g mut Graph, origin: ImageOrigin) -> Self {
        Self {
            graph,
            origin,
            format: None,
            width: None,
            height: None,
            depth: 1,
            mip_levels: 1,
            samples: SampleCount::S1,
            kind: ImageKind::Image2D,
            label: String::new(),
            usage: vk::ImageUsageFlags::empty(),
            resizable: false,
        }
    }

    pub fn format(mut self, format: vk::Format) -> Self {
        self.format = Some(format);
        self
    }

    pub fn extent(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    pub fn extent_3d(mut self, width: u32, height: u32, depth: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self.depth = depth;
        self
    }

    pub fn mip_levels(mut self, levels: u32) -> Self {
        self.mip_levels = levels;
        self
    }

    pub fn samples(mut self, samples: SampleCount) -> Self {
        self.samples = samples;
        self
    }

    pub fn array_2d(mut self, layers: u32) -> Self {
        self.kind = ImageKind::Image2DArray { layers };
        self
    }

    pub fn cubemap(mut self) -> Self {
        self.kind = ImageKind::Cubemap;
        self
    }

    pub fn cubemap_array(mut self, count: u32) -> Self {
        self.kind = ImageKind::CubemapArray { count };
        self
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    pub fn usage(mut self, usage: vk::ImageUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    pub fn resizable(mut self) -> Self {
        self.resizable = true;
        self
    }

    fn build_desc(&self) -> ImageDesc {
        let format = self.format.expect("ImageBuilder: format is required");

        let extent = match (self.width, self.height) {
            (Some(w), Some(h)) => vk::Extent3D {
                width: w,
                height: h,
                depth: self.depth,
            },
            _ => {
                let sc = self.graph.device.swapchain().extent();
                vk::Extent3D {
                    width: sc.width,
                    height: sc.height,
                    depth: self.depth,
                }
            }
        };

        ImageDesc {
            extent,
            format,
            mip_levels: self.mip_levels,
            samples: self.samples,
            kind: self.kind.clone(),
            label: self.label.clone(),
            usage: self.usage,
        }
    }

    pub fn build(self) -> Result<Image, GraphError> {
        let desc = self.build_desc();

        match self.origin {
            ImageOrigin::Transient => {
                let h = Image(self.graph.images.len() as u32);
                self.graph.images.push(ImageEntry::transient(desc));
                Ok(h)
            }
            ImageOrigin::Persistent => {
                assert!(
                    !self.graph.frame_active,
                    "persistent_image().build() must be called outside the frame loop"
                );

                let idx = self.graph.images.len();
                let h = Image(idx as u32);
                self.graph.images.push(ImageEntry::persistent(desc.clone()));
                self.graph.persistent_count += 1;

                if self.resizable {
                    self.graph
                        .resizable_images
                        .push((idx, ResizableTemplate { desc: desc.clone() }));
                }

                let entry = &mut self.graph.images[idx];
                if !entry.usage.is_empty() {
                    let device = self.graph.device.ash_device().clone();
                    let usage = entry.usage | vk::ImageUsageFlags::TRANSFER_DST;
                    let handle = self.graph.resources.create_image(
                        &device,
                        self.graph.device.allocator_mut(),
                        &entry.desc,
                        usage,
                        entry.aspect,
                    )?;
                    let view = self
                        .graph
                        .resources
                        .get_image(handle)
                        .expect("image just created")
                        .view;
                    let entry = &mut self.graph.images[idx];
                    register_bindless(entry, &mut self.graph.bindless, view);
                    entry.handle = Some(handle);
                }

                Ok(h)
            }
            ImageOrigin::External => {
                unreachable!("ImageBuilder does not support external images")
            }
        }
    }
}

pub struct TextureBuilder<'g> {
    graph: &'g mut Graph,
    label: String,
    pixels: Option<&'g [u8]>,
    width: Option<u32>,
    height: Option<u32>,
    format: Option<vk::Format>,
    mip_levels: u32,
}

impl<'g> TextureBuilder<'g> {
    pub(super) fn new(graph: &'g mut Graph, label: String) -> Self {
        Self {
            graph,
            label,
            pixels: None,
            width: None,
            height: None,
            format: None,
            mip_levels: 0,
        }
    }

    pub fn pixels(mut self, data: &'g [u8]) -> Self {
        self.pixels = Some(data);
        self
    }

    pub fn extent(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    pub fn format(mut self, format: vk::Format) -> Self {
        self.format = Some(format);
        self
    }

    pub fn mip_levels(mut self, levels: u32) -> Self {
        self.mip_levels = levels;
        self
    }

    pub fn build(self) -> Result<Image, GraphError> {
        assert!(
            !self.graph.frame_active,
            "load_texture().build() must be called outside the frame loop"
        );

        let pixels = self.pixels.expect("TextureBuilder: pixels() is required");
        let width = self.width.expect("TextureBuilder: extent() is required");
        let height = self.height.expect("TextureBuilder: extent() is required");
        let format = self.format.expect("TextureBuilder: format() is required");

        let mip_levels = if self.mip_levels == 0 {
            compute_mip_levels(width, height)
        } else {
            self.mip_levels
        };

        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let desc = ImageDesc {
            extent,
            format,
            mip_levels,
            samples: SampleCount::S1,
            kind: ImageKind::Image2D,
            label: self.label,
            usage: vk::ImageUsageFlags::empty(),
        };

        let usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let aspect = vk::ImageAspectFlags::COLOR;

        let device = self.graph.device.ash_device().clone();
        let handle = self.graph.resources.create_image(
            &device,
            self.graph.device.allocator_mut(),
            &desc,
            usage,
            aspect,
        )?;

        self.graph
            .upload_image_data(handle, pixels, extent, mip_levels)?;

        let view = self
            .graph
            .resources
            .get_image(handle)
            .expect("image just created")
            .view;
        let h = Image(self.graph.images.len() as u32);
        let mut entry = ImageEntry::loaded(desc, handle);
        register_bindless(&mut entry, &mut self.graph.bindless, view);
        self.graph.images.push(entry);
        self.graph.persistent_count += 1;
        Ok(h)
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
}
