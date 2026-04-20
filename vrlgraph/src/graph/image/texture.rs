use ash::vk;

use super::{Image, ImageEntry, compute_mip_levels};
use crate::graph::resources::register_bindless;
use crate::graph::{Graph, GraphError};
use crate::resource::{ImageDesc, ImageKind};
use crate::types::SampleCount;

pub struct TextureBuilder<'g> {
    graph: &'g mut Graph,
    label: String,
    pixels: Option<&'g [u8]>,
    mip_data: Option<Vec<&'g [u8]>>,
    width: Option<u32>,
    height: Option<u32>,
    format: Option<vk::Format>,
    mip_levels: u32,
}

impl<'g> TextureBuilder<'g> {
    pub(in crate::graph) fn new(graph: &'g mut Graph, label: String) -> Self {
        Self {
            graph,
            label,
            pixels: None,
            mip_data: None,
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

    pub fn mip_data(mut self, levels: &[&'g [u8]]) -> Self {
        self.mip_data = Some(levels.to_vec());
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

        let width = self.width.expect("TextureBuilder: extent() is required");
        let height = self.height.expect("TextureBuilder: extent() is required");
        let format = self.format.expect("TextureBuilder: format() is required");

        assert!(
            self.pixels.is_some() || self.mip_data.is_some(),
            "TextureBuilder: pixels() or mip_data() is required"
        );
        assert!(
            self.pixels.is_none() || self.mip_data.is_none(),
            "TextureBuilder: pixels() and mip_data() are mutually exclusive"
        );

        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let (mip_levels, usage) = if self.mip_data.is_some() {
            let levels = self.mip_data.as_ref().unwrap().len() as u32;
            (
                levels,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            )
        } else {
            let levels = if self.mip_levels == 0 {
                compute_mip_levels(width, height)
            } else {
                self.mip_levels
            };
            (
                levels,
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
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

        let aspect = vk::ImageAspectFlags::COLOR;

        let device = self.graph.device.ash_device().clone();
        let handle = self.graph.resources.create_image(
            &device,
            self.graph.device.allocator_mut(),
            &desc,
            usage,
            aspect,
        )?;

        if let Some(ref mip_data) = self.mip_data {
            self.graph
                .upload_image_data_with_mips(handle, mip_data, extent)?;
        } else {
            let pixels = self.pixels.unwrap();
            let id = self
                .graph
                .upload_image_data_async(handle, pixels, extent, mip_levels)?;
            self.graph.wait_for_transfer(id)?;
        }

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

    /// Uploads texture data via the transfer queue without blocking.
    /// The image data is uploaded asynchronously — use [`Cmd::try_buffer`](crate::graph::Cmd::try_buffer)
    /// pattern to check readiness.
    ///
    /// Only supports single-level pixel data (not pre-computed mip_data).
    /// Mipmap generation is deferred to the graphics queue after the transfer completes.
    pub fn build_async(self) -> Result<Image, GraphError> {
        assert!(
            !self.graph.frame_active,
            "load_texture().build_async() must be called outside the frame loop"
        );
        assert!(
            self.pixels.is_some(),
            "TextureBuilder: pixels() is required for build_async()"
        );
        assert!(
            self.mip_data.is_none(),
            "TextureBuilder: mip_data() is not supported with build_async(), use build() instead"
        );

        let width = self.width.expect("TextureBuilder: extent() is required");
        let height = self.height.expect("TextureBuilder: extent() is required");
        let format = self.format.expect("TextureBuilder: format() is required");

        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let mip_levels = if self.mip_levels == 0 {
            compute_mip_levels(width, height)
        } else {
            self.mip_levels
        };

        let usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC;

        let desc = ImageDesc {
            extent,
            format,
            mip_levels,
            samples: SampleCount::S1,
            kind: ImageKind::Image2D,
            label: self.label,
            usage: vk::ImageUsageFlags::empty(),
        };

        let aspect = vk::ImageAspectFlags::COLOR;

        let device = self.graph.device.ash_device().clone();
        let handle = self.graph.resources.create_image(
            &device,
            self.graph.device.allocator_mut(),
            &desc,
            usage,
            aspect,
        )?;

        let pixels = self.pixels.unwrap();
        let _id = self
            .graph
            .upload_image_data_async(handle, pixels, extent, mip_levels)?;

        let view = self
            .graph
            .resources
            .get_image(handle)
            .expect("image just created")
            .view;
        let h = Image(self.graph.images.len() as u32);
        let mut entry = ImageEntry::loaded(desc, handle);
        entry.layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        entry.stage = vk::PipelineStageFlags2::TRANSFER;
        entry.access = vk::AccessFlags2::TRANSFER_WRITE;
        register_bindless(&mut entry, &mut self.graph.bindless, view);
        self.graph.images.push(entry);
        self.graph.persistent_count += 1;
        Ok(h)
    }
}
