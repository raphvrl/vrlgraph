use ash::vk;

use super::pipeline::{ComputePipelineBuilder, PipelineBuilder};
#[cfg(debug_assertions)]
use super::reload::{PipelineDesc, PipelineKind};
use super::{Graph, GraphError};
use crate::resource::{GpuPipeline, PipelineHandle};

impl Graph {
    pub fn graphics_pipeline(&mut self) -> PipelineBuilder<'_> {
        PipelineBuilder::new(self)
    }

    pub fn compute_pipeline(&mut self) -> ComputePipelineBuilder<'_> {
        ComputePipelineBuilder::new(self)
    }

    pub fn destroy_pipeline(&mut self, handle: PipelineHandle) {
        let device = self.device.ash_device().clone();
        self.resources.destroy_pipeline(&device, handle);
    }

    pub(in crate::graph) fn insert_pipeline(&mut self, pipeline: GpuPipeline) -> PipelineHandle {
        self.resources.insert_pipeline(pipeline)
    }

    pub(in crate::graph) fn pipeline_cache(&self) -> vk::PipelineCache {
        self.pipeline_cache
    }

    #[cfg(debug_assertions)]
    pub fn reload_shaders(&mut self) -> Result<(), GraphError> {
        let changed = self.shader_watcher.drain_changed();
        if changed.is_empty() {
            return Ok(());
        }

        let affected: Vec<PipelineHandle> = self
            .pipeline_descs
            .iter()
            .filter(|(_, d)| {
                d.shader_paths()
                    .iter()
                    .any(|p| changed.iter().any(|c| c == p))
            })
            .map(|(h, _)| *h)
            .collect();

        if affected.is_empty() {
            return Ok(());
        }

        tracing::info!("hot reload: {} pipeline(s)", affected.len());
        unsafe { self.device.ash_device().device_wait_idle()? };

        for handle in affected {
            let desc = self.pipeline_descs[&handle].clone();
            match self.rebuild_pipeline(&desc) {
                Ok(new_pipe) => {
                    self.resources
                        .replace_pipeline(self.device.ash_device(), handle, new_pipe);
                }
                Err(e) => tracing::error!("pipeline rebuild failed: {e}"),
            }
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    pub(in crate::graph) fn register_pipeline_desc(
        &mut self,
        handle: PipelineHandle,
        desc: PipelineDesc,
    ) {
        desc.shader_paths()
            .iter()
            .for_each(|p| self.shader_watcher.watch(p));
        self.pipeline_descs.insert(handle, desc);
    }

    #[cfg(debug_assertions)]
    fn rebuild_pipeline(&self, desc: &PipelineDesc) -> Result<GpuPipeline, GraphError> {
        use super::pipeline::load_spv;

        let device = self.device.ash_device();

        match &desc.kind {
            PipelineKind::Graphics {
                vertex_path,
                fragment_path,
                color_formats,
                depth_format,
                vertex_bindings,
                vertex_attributes,
            } => {
                let vert_spv = load_spv(vertex_path)?;
                let frag_spv = load_spv(fragment_path)?;

                let vert_module = unsafe {
                    device.create_shader_module(
                        &vk::ShaderModuleCreateInfo::default().code(&vert_spv),
                        None,
                    )
                }?;
                let frag_module = unsafe {
                    device.create_shader_module(
                        &vk::ShaderModuleCreateInfo::default().code(&frag_spv),
                        None,
                    )
                }
                .inspect_err(|_| unsafe { device.destroy_shader_module(vert_module, None) })?;

                let entry = c"main";
                let stages = [
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::VERTEX)
                        .module(vert_module)
                        .name(entry),
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .module(frag_module)
                        .name(entry),
                ];

                let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_binding_descriptions(vertex_bindings)
                    .vertex_attribute_descriptions(vertex_attributes);
                let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
                let viewport_state = vk::PipelineViewportStateCreateInfo::default();
                let rasterization =
                    vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);
                let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                    .rasterization_samples(vk::SampleCountFlags::TYPE_1);
                let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
                let blend_attachments: Vec<_> = color_formats
                    .iter()
                    .map(|_| vk::PipelineColorBlendAttachmentState::default())
                    .collect();
                let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
                    .attachments(&blend_attachments);

                let dynamic_states = [
                    vk::DynamicState::VIEWPORT_WITH_COUNT,
                    vk::DynamicState::SCISSOR_WITH_COUNT,
                    vk::DynamicState::CULL_MODE,
                    vk::DynamicState::FRONT_FACE,
                    vk::DynamicState::PRIMITIVE_TOPOLOGY,
                    vk::DynamicState::DEPTH_TEST_ENABLE,
                    vk::DynamicState::DEPTH_WRITE_ENABLE,
                    vk::DynamicState::DEPTH_COMPARE_OP,
                    vk::DynamicState::RASTERIZER_DISCARD_ENABLE,
                    vk::DynamicState::DEPTH_BIAS_ENABLE,
                    vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
                    vk::DynamicState::POLYGON_MODE_EXT,
                    vk::DynamicState::COLOR_BLEND_ENABLE_EXT,
                    vk::DynamicState::COLOR_BLEND_EQUATION_EXT,
                    vk::DynamicState::COLOR_WRITE_MASK_EXT,
                ];
                let dynamic_state =
                    vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

                let layout_info = vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&desc.descriptor_set_layouts)
                    .push_constant_ranges(&desc.push_constant_ranges);
                let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
                    .inspect_err(|_| unsafe {
                        device.destroy_shader_module(vert_module, None);
                        device.destroy_shader_module(frag_module, None);
                    })?;

                let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
                    .color_attachment_formats(color_formats);
                if let Some(depth_fmt) = depth_format {
                    rendering_info = rendering_info.depth_attachment_format(*depth_fmt);
                }

                let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                    .stages(&stages)
                    .vertex_input_state(&vertex_input)
                    .input_assembly_state(&input_assembly)
                    .viewport_state(&viewport_state)
                    .rasterization_state(&rasterization)
                    .multisample_state(&multisample)
                    .depth_stencil_state(&depth_stencil)
                    .color_blend_state(&color_blend)
                    .dynamic_state(&dynamic_state)
                    .layout(layout)
                    .push_next(&mut rendering_info);

                let raw = unsafe {
                    device
                        .create_graphics_pipelines(self.pipeline_cache, &[pipeline_info], None)
                        .map_err(|(_, e)| {
                            device.destroy_shader_module(vert_module, None);
                            device.destroy_shader_module(frag_module, None);
                            device.destroy_pipeline_layout(layout, None);
                            e
                        })?
                };

                unsafe {
                    device.destroy_shader_module(vert_module, None);
                    device.destroy_shader_module(frag_module, None);
                }

                Ok(GpuPipeline {
                    pipeline: raw[0],
                    layout,
                })
            }

            PipelineKind::Compute { path } => {
                let spv = load_spv(path)?;

                let module = unsafe {
                    device.create_shader_module(
                        &vk::ShaderModuleCreateInfo::default().code(&spv),
                        None,
                    )
                }?;

                let entry = c"main";
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(entry);

                let layout_info = vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&desc.descriptor_set_layouts)
                    .push_constant_ranges(&desc.push_constant_ranges);
                let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
                    .inspect_err(|_| unsafe { device.destroy_shader_module(module, None) })?;

                let pipeline_info = vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout);

                let raw = unsafe {
                    device
                        .create_compute_pipelines(self.pipeline_cache, &[pipeline_info], None)
                        .map_err(|(_, e)| {
                            device.destroy_pipeline_layout(layout, None);
                            e
                        })?
                };

                unsafe { device.destroy_shader_module(module, None) };

                Ok(GpuPipeline {
                    pipeline: raw[0],
                    layout,
                })
            }
        }
    }
}
