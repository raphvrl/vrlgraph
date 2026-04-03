use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};

use ash::vk;

use crate::resource::{GpuPipeline, Pipeline, ShaderModule};
use crate::vertex::VertexInput;

#[cfg(debug_assertions)]
use super::reload::{PipelineDesc, PipelineKind};
use super::{Graph, GraphError};

const DYNAMIC_STATES: &[vk::DynamicState] = &[
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

#[allow(clippy::too_many_arguments)]
pub(super) fn create_graphics_pipeline_raw(
    device: &ash::Device,
    cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    vert_module: vk::ShaderModule,
    vert_entry: &CStr,
    frag_module: vk::ShaderModule,
    frag_entry: &CStr,
    color_formats: &[vk::Format],
    depth_format: Option<vk::Format>,
    vertex_bindings: &[vk::VertexInputBindingDescription],
    vertex_attributes: &[vk::VertexInputAttributeDescription],
) -> Result<GpuPipeline, GraphError> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(vert_entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(frag_entry),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(vertex_bindings)
        .vertex_attribute_descriptions(vertex_attributes);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default();
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();

    let blend_attachments: Vec<_> = color_formats
        .iter()
        .map(|_| vk::PipelineColorBlendAttachmentState::default())
        .collect();
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(DYNAMIC_STATES);

    let mut rendering_info =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(color_formats);
    if let Some(depth_fmt) = depth_format {
        rendering_info = rendering_info.depth_attachment_format(depth_fmt);
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
            .create_graphics_pipelines(cache, &[pipeline_info], None)
            .map_err(|(_, e)| e)?
    };

    Ok(GpuPipeline {
        pipeline: raw[0],
        layout,
    })
}

pub(super) fn create_compute_pipeline_raw(
    device: &ash::Device,
    cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    compute_module: vk::ShaderModule,
    compute_entry: &CStr,
) -> Result<GpuPipeline, GraphError> {
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(compute_module)
        .name(compute_entry);

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);

    let raw = unsafe {
        device
            .create_compute_pipelines(cache, &[pipeline_info], None)
            .map_err(|(_, e)| e)?
    };

    Ok(GpuPipeline {
        pipeline: raw[0],
        layout,
    })
}

/// Builder for a graphics pipeline.
///
/// Obtained from [`Graph::graphics_pipeline`]. At minimum you must provide a
/// vertex and a fragment shader module. All rasterizer state is dynamic — you
/// set it per draw call via [`Cmd`](super::command::Cmd).
///
/// Color formats default to the swapchain format. Override with
/// [`color_formats`](PipelineBuilder::color_formats) when rendering to
/// off-screen targets.
pub struct PipelineBuilder<'g> {
    graph: &'g mut Graph,
    label: String,
    vertex: Option<ShaderModule>,
    fragment: Option<ShaderModule>,
    color_formats: Vec<vk::Format>,
    depth_format: Option<vk::Format>,
    vertex_bindings: Vec<vk::VertexInputBindingDescription>,
    vertex_attributes: Vec<vk::VertexInputAttributeDescription>,
}

impl<'g> PipelineBuilder<'g> {
    pub(super) fn new(graph: &'g mut Graph, label: impl Into<String>) -> Self {
        let swapchain_format = graph.device().swapchain().format();
        Self {
            graph,
            label: label.into(),
            vertex: None,
            fragment: None,
            color_formats: vec![swapchain_format],
            depth_format: None,
            vertex_bindings: Vec::new(),
            vertex_attributes: Vec::new(),
        }
    }

    /// Sets the vertex shader module. Required.
    pub fn vertex_shader(mut self, module: ShaderModule) -> Self {
        self.vertex = Some(module);
        self
    }

    /// Sets the fragment shader module. Required.
    pub fn fragment_shader(mut self, module: ShaderModule) -> Self {
        self.fragment = Some(module);
        self
    }

    /// Overrides the color attachment formats. By default the swapchain format
    /// is used. Set this when the pipeline renders to off-screen images.
    pub fn color_formats(mut self, formats: &[vk::Format]) -> Self {
        self.color_formats = formats.to_vec();
        self
    }

    /// Sets the depth attachment format. Required if the pass writes a depth
    /// attachment.
    pub fn depth_format(mut self, format: vk::Format) -> Self {
        self.depth_format = Some(format);
        self
    }

    /// Declares the vertex input layout from a type that implements [`VertexInput`].
    ///
    /// Use `#[derive(VertexInput)]` on your vertex struct and call
    /// `.vertex_input::<MyVertex>()`. Skip this for shader-only draws (e.g.
    /// fullscreen triangles with no vertex buffer).
    pub fn vertex_input<V: VertexInput>(mut self) -> Self {
        self.vertex_bindings = V::BINDINGS.to_vec();
        self.vertex_attributes = V::ATTRIBUTES.to_vec();
        self
    }

    /// Raw override for vertex input — accepts Vulkan descriptors directly.
    ///
    /// Prefer [`vertex_input`](Self::vertex_input) with `#[derive(VertexInput)]`.
    /// Use this only when the layout cannot be expressed as a `VertexInput` impl.
    pub fn vertex_input_raw(
        mut self,
        bindings: &[vk::VertexInputBindingDescription],
        attributes: &[vk::VertexInputAttributeDescription],
    ) -> Self {
        self.vertex_bindings = bindings.to_vec();
        self.vertex_attributes = attributes.to_vec();
        self
    }

    /// Compiles the pipeline and registers it with the graph.
    /// Returns a [`Pipeline`] that can be passed to [`FrameResources::pipeline`](super::pass::FrameResources::pipeline).
    pub fn build(self) -> Result<Pipeline, GraphError> {
        let vertex = self
            .vertex
            .expect("PipelineBuilder: vertex_shader() is required");
        let fragment = self
            .fragment
            .expect("PipelineBuilder: fragment_shader() is required");

        let vert = self
            .graph
            .resources
            .get_shader_module(vertex.0)
            .expect("PipelineBuilder: vertex ShaderModule not found in pool");
        let vert_module = vert.module;
        let vert_entry = vert.entry.clone();

        let frag = self
            .graph
            .resources
            .get_shader_module(fragment.0)
            .expect("PipelineBuilder: fragment ShaderModule not found in pool");
        let frag_module = frag.module;
        let frag_entry = frag.entry.clone();

        let layout = self.graph.bindless.pipeline_layout();

        let gpu_pipeline = create_graphics_pipeline_raw(
            self.graph.ash_device(),
            self.graph.pipeline_cache(),
            layout,
            vert_module,
            &vert_entry,
            frag_module,
            &frag_entry,
            &self.color_formats,
            self.depth_format,
            &self.vertex_bindings,
            &self.vertex_attributes,
        )?;

        if let Some(du) = self.graph.device().debug_utils() {
            let name = CString::new(self.label.as_str()).unwrap();
            let info = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(gpu_pipeline.pipeline)
                .object_name(&name);
            let _ = unsafe { du.set_debug_utils_object_name(&info) };
        }

        let handle = self.graph.insert_pipeline(gpu_pipeline);

        #[cfg(debug_assertions)]
        self.graph.register_pipeline_desc(
            handle,
            PipelineDesc {
                kind: PipelineKind::Graphics {
                    vertex: vertex.0,
                    fragment: fragment.0,
                    color_formats: self.color_formats,
                    depth_format: self.depth_format,
                    vertex_bindings: self.vertex_bindings,
                    vertex_attributes: self.vertex_attributes,
                },
            },
        );

        Ok(Pipeline(handle))
    }
}

/// Builder for a compute pipeline.
///
/// Obtained from [`Graph::compute_pipeline`]. Provide a compute shader module.
pub struct ComputePipelineBuilder<'g> {
    graph: &'g mut Graph,
    label: String,
    shader: Option<ShaderModule>,
}

impl<'g> ComputePipelineBuilder<'g> {
    pub(super) fn new(graph: &'g mut Graph, label: impl Into<String>) -> Self {
        Self {
            graph,
            label: label.into(),
            shader: None,
        }
    }

    /// Sets the compute shader module. Required.
    pub fn shader(mut self, module: ShaderModule) -> Self {
        self.shader = Some(module);
        self
    }

    /// Compiles the pipeline and registers it with the graph.
    pub fn build(self) -> Result<Pipeline, GraphError> {
        let shader = self
            .shader
            .expect("ComputePipelineBuilder: shader() is required");

        let sm = self
            .graph
            .resources
            .get_shader_module(shader.0)
            .expect("ComputePipelineBuilder: ShaderModule not found in pool");
        let compute_module = sm.module;
        let compute_entry = sm.entry.clone();

        let layout = self.graph.bindless.pipeline_layout();

        let gpu_pipeline = create_compute_pipeline_raw(
            self.graph.ash_device(),
            self.graph.pipeline_cache(),
            layout,
            compute_module,
            &compute_entry,
        )?;

        if let Some(du) = self.graph.device().debug_utils() {
            let name = CString::new(self.label.as_str()).unwrap();
            let info = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(gpu_pipeline.pipeline)
                .object_name(&name);
            let _ = unsafe { du.set_debug_utils_object_name(&info) };
        }

        let handle = self.graph.insert_pipeline(gpu_pipeline);

        #[cfg(debug_assertions)]
        self.graph.register_pipeline_desc(
            handle,
            PipelineDesc {
                kind: PipelineKind::Compute { shader: shader.0 },
            },
        );

        Ok(Pipeline(handle))
    }
}

pub(super) fn resolve_shader_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_owned();
    }
    let exe_relative = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(path)));
    if let Some(p) = &exe_relative {
        if p.exists() {
            return p.clone();
        }
    }
    std::env::current_dir()
        .ok()
        .map(|dir| dir.join(path))
        .unwrap_or_else(|| path.to_owned())
}

pub(super) fn load_spv(path: &Path) -> Result<Vec<u32>, super::GraphError> {
    let bytes = std::fs::read(path)
        .map_err(|e| super::GraphError::ShaderLoad(format!("{}: {e}", path.display())))?;

    if bytes.len() % 4 != 0 {
        return Err(super::GraphError::ShaderLoad(format!(
            "{}: SPIR-V size is not aligned to 4 bytes",
            path.display()
        )));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("chunks_exact(4) guarantees 4 bytes")))
        .collect())
}
