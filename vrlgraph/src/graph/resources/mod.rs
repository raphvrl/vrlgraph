mod bindless_ops;
mod buffer_ops;
mod image_ops;
mod staging;

pub(in crate::graph) use bindless_ops::{register_bindless, update_bindless};

use super::Graph;
use super::bindless::Sampler;
use super::image::{ImageBuilder, ImageOrigin, TextureBuilder};
use super::sampler::SamplerBuilder;

impl Graph {
    pub fn transient_image(&mut self) -> ImageBuilder<'_> {
        ImageBuilder::new(self, ImageOrigin::Transient)
    }

    pub fn persistent_image(&mut self, label: impl Into<String>) -> ImageBuilder<'_> {
        ImageBuilder::new(self, ImageOrigin::Persistent).label(label)
    }

    pub fn load_texture(&mut self, label: impl Into<String>) -> TextureBuilder<'_> {
        TextureBuilder::new(self, label.into())
    }

    pub fn create_sampler(&mut self) -> SamplerBuilder<'_> {
        SamplerBuilder::new(self)
    }

    pub fn destroy_sampler(&mut self, sampler: Sampler) {
        self.resources
            .destroy_sampler(self.device.ash_device(), sampler.handle);
    }
}
