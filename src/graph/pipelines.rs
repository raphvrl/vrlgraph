use std::ffi::CString;
use std::path::Path;

use ash::vk;

use super::pipeline::{ComputePipelineBuilder, PipelineBuilder, load_spv, resolve_shader_path};
#[cfg(debug_assertions)]
use super::reload::{PipelineDesc, PipelineKind};
use super::{Graph, GraphError};
#[cfg(debug_assertions)]
use crate::resource::ShaderModuleHandle;
use crate::resource::{GpuPipeline, GpuShaderModule, Pipeline, PipelineHandle, ShaderModule};

impl Graph {
    /// Loads a SPIR-V shader from `path`, registers it as a [`ShaderModule`]
    /// in the resource pool, and returns an opaque handle to it.
    ///
    /// `entry` is the name of the entry-point function in the shader (e.g.
    /// `"main"`, `"vs_main"`, `"fs_main"`). Relative paths are resolved from
    /// the directory of the current executable.
    pub fn shader_module(
        &mut self,
        path: impl AsRef<Path>,
        entry: impl AsRef<str>,
    ) -> Result<ShaderModule, GraphError> {
        let resolved = resolve_shader_path(path.as_ref());

        let module = if let Some(&cached) = self.spirv_module_cache.get(&resolved) {
            cached
        } else {
            let spv = load_spv(&resolved)?;
            let m = unsafe {
                self.ash_device()
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
            }?;
            self.spirv_module_cache.insert(resolved.clone(), m);
            m
        };

        let entry =
            CString::new(entry.as_ref()).expect("shader entry point must not contain null bytes");
        let handle = self
            .resources
            .insert_shader_module(GpuShaderModule { module, entry });

        #[cfg(debug_assertions)]
        {
            self.shader_watcher.watch(&resolved);
            self.shader_module_paths.insert(handle, resolved);
        }

        Ok(ShaderModule(handle))
    }

    /// Destroys a shader module handle. The underlying `ash::vk::ShaderModule` is shared via the
    /// path cache and will be freed when the graph is dropped.
    pub fn destroy_shader_module(&mut self, handle: ShaderModule) {
        self.resources.destroy_shader_module(handle.0);
        #[cfg(debug_assertions)]
        self.shader_module_paths.remove(&handle.0);
    }

    pub fn graphics_pipeline(&mut self, label: impl Into<String>) -> PipelineBuilder<'_> {
        PipelineBuilder::new(self, label)
    }

    pub fn compute_pipeline(&mut self, label: impl Into<String>) -> ComputePipelineBuilder<'_> {
        ComputePipelineBuilder::new(self, label)
    }

    pub fn destroy_pipeline(&mut self, handle: Pipeline) {
        let device = self.device.ash_device().clone();
        self.resources.destroy_pipeline(&device, handle.0);
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

        let changed_paths: std::collections::HashSet<std::path::PathBuf> = self
            .shader_module_paths
            .values()
            .filter(|p| changed.iter().any(|c| c == *p))
            .cloned()
            .collect();

        if changed_paths.is_empty() {
            return Ok(());
        }

        unsafe { self.device.ash_device().device_wait_idle()? };

        let mut affected_modules: Vec<ShaderModuleHandle> = Vec::new();

        for path in &changed_paths {
            match load_spv(path).and_then(|spv| unsafe {
                self.device
                    .ash_device()
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
                    .map_err(GraphError::from)
            }) {
                Ok(new_vk_module) => {
                    if let Some(old) = self.spirv_module_cache.insert(path.clone(), new_vk_module) {
                        unsafe { self.device.ash_device().destroy_shader_module(old, None) };
                    }
                    let handles: Vec<ShaderModuleHandle> = self
                        .shader_module_paths
                        .iter()
                        .filter(|(_, p)| *p == path)
                        .map(|(h, _)| *h)
                        .collect();
                    for handle in handles {
                        self.resources
                            .update_shader_module_vk(handle, new_vk_module);
                        affected_modules.push(handle);
                    }
                }
                Err(e) => {
                    tracing::error!("shader module reload failed for {}: {e}", path.display())
                }
            }
        }

        let affected_pipelines: Vec<PipelineHandle> = self
            .pipeline_descs
            .iter()
            .filter(|(_, d)| {
                d.shader_module_handles()
                    .iter()
                    .any(|h| affected_modules.contains(h))
            })
            .map(|(h, _)| *h)
            .collect();

        if affected_pipelines.is_empty() {
            return Ok(());
        }

        tracing::info!("hot reload: {} pipeline(s)", affected_pipelines.len());

        for handle in affected_pipelines {
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
        self.pipeline_descs.insert(handle, desc);
    }

    #[cfg(debug_assertions)]
    fn rebuild_pipeline(&self, desc: &PipelineDesc) -> Result<GpuPipeline, GraphError> {
        let device = self.device.ash_device();

        match &desc.kind {
            PipelineKind::Graphics {
                vertex,
                fragment,
                color_formats,
                depth_format,
                vertex_bindings,
                vertex_attributes,
            } => {
                let vert_module = self
                    .resources
                    .get_shader_module(*vertex)
                    .expect("vertex shader module must exist")
                    .module;
                let vert_entry = self
                    .resources
                    .get_shader_module(*vertex)
                    .expect("vertex shader module must exist")
                    .entry
                    .clone();
                let frag_module = self
                    .resources
                    .get_shader_module(*fragment)
                    .expect("fragment shader module must exist")
                    .module;
                let frag_entry = self
                    .resources
                    .get_shader_module(*fragment)
                    .expect("fragment shader module must exist")
                    .entry
                    .clone();

                let stages = [
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::VERTEX)
                        .module(vert_module)
                        .name(vert_entry.as_c_str()),
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .module(frag_module)
                        .name(frag_entry.as_c_str()),
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

                let layout = self.bindless.pipeline_layout();

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
                        .map_err(|(_, e)| e)?
                };

                Ok(GpuPipeline {
                    pipeline: raw[0],
                    layout,
                })
            }

            PipelineKind::Compute { shader } => {
                let compute_module = self
                    .resources
                    .get_shader_module(*shader)
                    .expect("compute shader module must exist")
                    .module;
                let compute_entry = self
                    .resources
                    .get_shader_module(*shader)
                    .expect("compute shader module must exist")
                    .entry
                    .clone();

                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(compute_module)
                    .name(compute_entry.as_c_str());

                let layout = self.bindless.pipeline_layout();

                let pipeline_info = vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout);

                let raw = unsafe {
                    device
                        .create_compute_pipelines(self.pipeline_cache, &[pipeline_info], None)
                        .map_err(|(_, e)| e)?
                };

                Ok(GpuPipeline {
                    pipeline: raw[0],
                    layout,
                })
            }
        }
    }
}
