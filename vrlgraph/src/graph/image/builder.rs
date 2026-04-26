use ash::vk;

use super::{Image, ImageEntry, ImageOrigin, ResizableTemplate};
use crate::graph::resources::register_bindless;
use crate::graph::{Graph, GraphError};
use crate::resource::{ImageDesc, ImageKind};
use crate::types::SampleCount;

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
    pub(in crate::graph) fn new(
        graph: &'g mut Graph,
        origin: ImageOrigin,
        label: impl Into<String>,
    ) -> Self {
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
            label: label.into(),
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
                    let usage = entry.usage
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC;
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
