use std::path::{Path, PathBuf};

use ash::vk;

use crate::resource::{GpuPipeline, Pipeline};
use crate::vertex::VertexInput;

#[cfg(debug_assertions)]
use super::reload::{PipelineDesc, PipelineKind};
use super::{Graph, GraphError};

/// Builder for a graphics pipeline.
///
/// Obtained from [`Graph::graphics_pipeline`]. At minimum you must provide a
/// vertex and a fragment shader. All rasterizer state is dynamic — you set it
/// per draw call via [`Cmd`](super::command::Cmd).
///
/// Color formats default to the swapchain format. Override with
/// [`color_formats`](PipelineBuilder::color_formats) when rendering to
/// off-screen targets.
pub struct PipelineBuilder<'g> {
    graph: &'g mut Graph,
    vertex_spv: Option<Vec<u32>>,
    fragment_spv: Option<Vec<u32>>,
    color_formats: Vec<vk::Format>,
    depth_format: Option<vk::Format>,
    vertex_bindings: Vec<vk::VertexInputBindingDescription>,
    vertex_attributes: Vec<vk::VertexInputAttributeDescription>,
    #[cfg(debug_assertions)]
    vertex_path: Option<PathBuf>,
    #[cfg(debug_assertions)]
    fragment_path: Option<PathBuf>,
}

impl<'g> PipelineBuilder<'g> {
    pub(super) fn new(graph: &'g mut Graph) -> Self {
        let swapchain_format = graph.device().swapchain().format();
        Self {
            graph,
            vertex_spv: None,
            fragment_spv: None,
            color_formats: vec![swapchain_format],
            depth_format: None,
            vertex_bindings: Vec::new(),
            vertex_attributes: Vec::new(),
            #[cfg(debug_assertions)]
            vertex_path: None,
            #[cfg(debug_assertions)]
            fragment_path: None,
        }
    }

    /// Loads the vertex shader from a SPIR-V file. Required.
    ///
    /// Relative paths are resolved from the directory of the current executable.
    pub fn vertex_shader(mut self, path: impl AsRef<Path>) -> Result<Self, GraphError> {
        let resolved = resolve_shader_path(path.as_ref());
        #[cfg(debug_assertions)]
        {
            self.vertex_path = Some(resolved.clone());
        }
        self.vertex_spv = Some(load_spv(&resolved)?);
        Ok(self)
    }

    /// Loads the fragment shader from a SPIR-V file. Required.
    pub fn fragment_shader(mut self, path: impl AsRef<Path>) -> Result<Self, GraphError> {
        let resolved = resolve_shader_path(path.as_ref());
        #[cfg(debug_assertions)]
        {
            self.fragment_path = Some(resolved.clone());
        }
        self.fragment_spv = Some(load_spv(&resolved)?);
        Ok(self)
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
        let vert_spv = self
            .vertex_spv
            .expect("PipelineBuilder: vertex_shader() is required");
        let frag_spv = self
            .fragment_spv
            .expect("PipelineBuilder: fragment_shader() is required");

        let device = self.graph.ash_device().clone();

        let vert_module = unsafe {
            device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_spv), None)
        }?;

        let frag_module = unsafe {
            device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_spv), None)
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
            .vertex_binding_descriptions(&self.vertex_bindings)
            .vertex_attribute_descriptions(&self.vertex_attributes);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default();

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();

        let blend_attachments: Vec<_> = self
            .color_formats
            .iter()
            .map(|_| vk::PipelineColorBlendAttachmentState::default())
            .collect();

        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

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

        let layout = self.graph.bindless.pipeline_layout();

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&self.color_formats);
        if let Some(depth_fmt) = self.depth_format {
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
                .create_graphics_pipelines(self.graph.pipeline_cache(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?
        };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        let handle = self.graph.insert_pipeline(GpuPipeline {
            pipeline: raw[0],
            layout,
        });

        #[cfg(debug_assertions)]
        self.graph.register_pipeline_desc(
            handle,
            PipelineDesc {
                kind: PipelineKind::Graphics {
                    vertex_path: self.vertex_path.expect("set together with spv"),
                    fragment_path: self.fragment_path.expect("set together with spv"),
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
/// Obtained from [`Graph::compute_pipeline`]. Provide a compute shader and,
/// optionally, push constants and descriptor set layouts.
pub struct ComputePipelineBuilder<'g> {
    graph: &'g mut Graph,
    compute_spv: Option<Vec<u32>>,
    #[cfg(debug_assertions)]
    compute_path: Option<PathBuf>,
}

impl<'g> ComputePipelineBuilder<'g> {
    pub(super) fn new(graph: &'g mut Graph) -> Self {
        Self {
            graph,
            compute_spv: None,
            #[cfg(debug_assertions)]
            compute_path: None,
        }
    }

    /// Loads the compute shader from a SPIR-V file. Required.
    ///
    /// Relative paths are resolved from the directory of the current executable.
    pub fn shader(mut self, path: impl AsRef<Path>) -> Result<Self, GraphError> {
        let resolved = resolve_shader_path(path.as_ref());
        #[cfg(debug_assertions)]
        {
            self.compute_path = Some(resolved.clone());
        }
        self.compute_spv = Some(load_spv(&resolved)?);
        Ok(self)
    }

    /// Compiles the pipeline and registers it with the graph.
    pub fn build(self) -> Result<Pipeline, GraphError> {
        let spv = self
            .compute_spv
            .expect("ComputePipelineBuilder: shader() is required");

        let device = self.graph.ash_device().clone();

        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
        }?;

        let entry = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);

        let layout = self.graph.bindless.pipeline_layout();

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

        let raw = unsafe {
            device
                .create_compute_pipelines(self.graph.pipeline_cache(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?
        };

        unsafe { device.destroy_shader_module(module, None) };

        let handle = self.graph.insert_pipeline(GpuPipeline {
            pipeline: raw[0],
            layout,
        });

        #[cfg(debug_assertions)]
        self.graph.register_pipeline_desc(
            handle,
            PipelineDesc {
                kind: PipelineKind::Compute {
                    path: self.compute_path.expect("set together with spv"),
                },
            },
        );

        Ok(Pipeline(handle))
    }
}

fn resolve_shader_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_owned();
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(path)))
        .unwrap_or_else(|| path.to_owned())
}

pub(super) fn load_spv(path: &Path) -> Result<Vec<u32>, GraphError> {
    let bytes = std::fs::read(path)
        .map_err(|e| GraphError::ShaderLoad(format!("{}: {e}", path.display())))?;

    if bytes.len() % 4 != 0 {
        return Err(GraphError::ShaderLoad(format!(
            "{}: taille SPIR-V non alignée sur 4 octets",
            path.display()
        )));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("chunks_exact(4) guarantees 4 bytes")))
        .collect())
}
