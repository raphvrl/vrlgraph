use std::ffi::CString;
use std::path::Path;

use ash::vk;

use super::pipeline::{ComputePipelineBuilder, PipelineBuilder, load_spv, resolve_shader_path};
#[cfg(debug_assertions)]
use super::pipeline::{create_compute_pipeline_raw, create_graphics_pipeline_raw};
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

    pub fn shader_module_from_spirv(
        &mut self,
        spirv: &[u8],
        entry: impl AsRef<str>,
    ) -> Result<ShaderModule, GraphError> {
        assert!(
            spirv.len().is_multiple_of(4),
            "SPIR-V byte slice length must be a multiple of 4"
        );
        let code: &[u32] =
            unsafe { std::slice::from_raw_parts(spirv.as_ptr().cast(), spirv.len() / 4) };

        let module = unsafe {
            self.ash_device()
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None)
        }?;

        let entry =
            CString::new(entry.as_ref()).expect("shader entry point must not contain null bytes");
        let handle = self
            .resources
            .insert_shader_module(GpuShaderModule { module, entry });

        Ok(ShaderModule(handle))
    }

    /// Destroys a shader module handle. For file-backed modules the underlying
    /// `VkShaderModule` is shared via the path cache and freed when the graph
    /// is dropped. For SPIR-V byte-backed modules we are the sole owner, so
    /// the Vulkan object is destroyed immediately.
    pub fn destroy_shader_module(&mut self, handle: ShaderModule) {
        if let Some(m) = self.resources.get_shader_module(handle.0) {
            let vk_module = m.module;
            if !self.spirv_module_cache.values().any(|&c| c == vk_module) {
                unsafe { self.ash_device().destroy_shader_module(vk_module, None) };
            }
        }
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

        let changed_paths: rustc_hash::FxHashSet<std::path::PathBuf> = self
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
        let layout = self.bindless.pipeline_layout();

        match &desc.kind {
            PipelineKind::Graphics {
                vertex,
                fragment,
                color_formats,
                depth_format,
                vertex_bindings,
                vertex_attributes,
                view_mask,
            } => {
                let vert = self
                    .resources
                    .get_shader_module(*vertex)
                    .expect("vertex shader module must exist");
                let vert_module = vert.module;
                let vert_entry = vert.entry.clone();

                let frag = self
                    .resources
                    .get_shader_module(*fragment)
                    .expect("fragment shader module must exist");
                let frag_module = frag.module;
                let frag_entry = frag.entry.clone();

                create_graphics_pipeline_raw(
                    device,
                    self.pipeline_cache,
                    layout,
                    vert_module,
                    &vert_entry,
                    frag_module,
                    &frag_entry,
                    color_formats,
                    *depth_format,
                    vertex_bindings,
                    vertex_attributes,
                    *view_mask,
                )
            }

            PipelineKind::Compute { shader } => {
                let sm = self
                    .resources
                    .get_shader_module(*shader)
                    .expect("compute shader module must exist");
                let compute_module = sm.module;
                let compute_entry = sm.entry.clone();

                create_compute_pipeline_raw(
                    device,
                    self.pipeline_cache,
                    layout,
                    compute_module,
                    &compute_entry,
                )
            }
        }
    }
}
