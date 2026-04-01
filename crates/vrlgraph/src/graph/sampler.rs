use ash::vk;

use super::{Graph, GraphError};
use crate::graph::bindless::Sampler;
use crate::types::{AddressMode, BorderColor, CompareOp, Filter, MipmapMode};

pub struct SamplerBuilder<'g> {
    graph: &'g mut Graph,
    mag_filter: Filter,
    min_filter: Filter,
    mipmap_mode: MipmapMode,
    address_mode_u: AddressMode,
    address_mode_v: AddressMode,
    address_mode_w: AddressMode,
    max_anisotropy: Option<f32>,
    compare_op: Option<CompareOp>,
    min_lod: f32,
    max_lod: f32,
    mip_lod_bias: f32,
    border_color: BorderColor,
    unnormalized_coordinates: bool,
}

impl<'g> SamplerBuilder<'g> {
    pub(super) fn new(graph: &'g mut Graph) -> Self {
        Self {
            graph,
            mag_filter: Filter::LINEAR,
            min_filter: Filter::LINEAR,
            mipmap_mode: MipmapMode::LINEAR,
            address_mode_u: AddressMode::REPEAT,
            address_mode_v: AddressMode::REPEAT,
            address_mode_w: AddressMode::REPEAT,
            max_anisotropy: None,
            compare_op: None,
            min_lod: 0.0,
            max_lod: vk::LOD_CLAMP_NONE,
            mip_lod_bias: 0.0,
            border_color: BorderColor::default(),
            unnormalized_coordinates: false,
        }
    }

    pub fn mag_filter(mut self, filter: Filter) -> Self {
        self.mag_filter = filter;
        self
    }

    pub fn min_filter(mut self, filter: Filter) -> Self {
        self.min_filter = filter;
        self
    }

    pub fn filter(mut self, filter: Filter) -> Self {
        self.mag_filter = filter;
        self.min_filter = filter;
        self
    }

    pub fn mipmap_mode(mut self, mode: MipmapMode) -> Self {
        self.mipmap_mode = mode;
        self
    }

    pub fn address_mode_u(mut self, mode: AddressMode) -> Self {
        self.address_mode_u = mode;
        self
    }

    pub fn address_mode_v(mut self, mode: AddressMode) -> Self {
        self.address_mode_v = mode;
        self
    }

    pub fn address_mode_w(mut self, mode: AddressMode) -> Self {
        self.address_mode_w = mode;
        self
    }

    pub fn address_mode(mut self, mode: AddressMode) -> Self {
        self.address_mode_u = mode;
        self.address_mode_v = mode;
        self.address_mode_w = mode;
        self
    }

    pub fn anisotropy(mut self, max: f32) -> Self {
        self.max_anisotropy = Some(max);
        self
    }

    pub fn compare_op(mut self, op: CompareOp) -> Self {
        self.compare_op = Some(op);
        self
    }

    pub fn lod(mut self, min: f32, max: f32) -> Self {
        self.min_lod = min;
        self.max_lod = max;
        self
    }

    pub fn mip_lod_bias(mut self, bias: f32) -> Self {
        self.mip_lod_bias = bias;
        self
    }

    pub fn border_color(mut self, color: BorderColor) -> Self {
        self.border_color = color;
        self
    }

    pub fn unnormalized_coordinates(mut self) -> Self {
        self.unnormalized_coordinates = true;
        self
    }

    pub fn build(self) -> Result<Sampler, GraphError> {
        let mut info = vk::SamplerCreateInfo::default()
            .mag_filter(self.mag_filter.into())
            .min_filter(self.min_filter.into())
            .mipmap_mode(self.mipmap_mode.into())
            .address_mode_u(self.address_mode_u.into())
            .address_mode_v(self.address_mode_v.into())
            .address_mode_w(self.address_mode_w.into())
            .min_lod(self.min_lod)
            .max_lod(self.max_lod)
            .mip_lod_bias(self.mip_lod_bias)
            .border_color(self.border_color.into())
            .unnormalized_coordinates(self.unnormalized_coordinates);

        if let Some(max) = self.max_anisotropy {
            info = info.anisotropy_enable(true).max_anisotropy(max);
        }

        if let Some(op) = self.compare_op {
            info = info.compare_enable(true).compare_op(op.into());
        }

        let handle = self
            .graph
            .resources
            .create_sampler(self.graph.device.ash_device(), &info)?;
        let raw = self
            .graph
            .resources
            .get_sampler(handle)
            .expect("sampler just created");
        let index = self.graph.bindless.write_sampler(raw);
        Ok(Sampler::new(handle, index))
    }
}
