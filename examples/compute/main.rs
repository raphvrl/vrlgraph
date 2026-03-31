#[path = "../common/mod.rs"]
mod common;

use vrlgraph::ash::vk;
use vrlgraph::prelude::*;
use winit::window::Window;

#[derive(ShaderType)]
struct FillParams {
    width: u32,
    height: u32,
    storage_idx: u32,
}

#[derive(ShaderType)]
struct BlitParams {
    sampled_idx: u32,
    sampler_idx: u32,
}

struct State {
    graph: Graph,
    window: Window,
    compute_pipeline: Pipeline,
    graphics_pipeline: Pipeline,
    storage_image: Image,
    sampler: Sampler,
}

impl common::Example for State {
    fn init(window: Window) -> Result<Self, GraphError> {
        let size = window.inner_size();

        let mut graph = Graph::builder()
            .window(&window)
            .size(size.width, size.height)
            .validation(cfg!(debug_assertions))
            .present_mode(PresentMode::Fifo)
            .build()?;

        let storage_image = graph
            .persistent_image()
            .format(vk::Format::R8G8B8A8_UNORM)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
            .label("triangle_storage")
            .resizable()
            .build()?;

        let sampler = graph
            .create_sampler()
            .filter(Filter::NEAREST)
            .address_mode_u(AddressMode::CLAMP_TO_EDGE)
            .address_mode_v(AddressMode::CLAMP_TO_EDGE)
            .build()?;

        let cs = graph.shader_module("shaders/fill.comp.spv", "main")?;
        let compute_pipeline = graph.compute_pipeline("fill").shader(cs).build()?;

        let vs = graph.shader_module("shaders/fullscreen.vert.spv", "main")?;
        let fs = graph.shader_module("shaders/blit.frag.spv", "main")?;
        let graphics_pipeline = graph
            .graphics_pipeline("blit")
            .vertex_shader(vs)
            .fragment_shader(fs)
            .build()?;

        Ok(Self {
            graph,
            window,
            compute_pipeline,
            graphics_pipeline,
            storage_image,
            sampler,
        })
    }

    fn draw(&mut self) -> Result<(), GraphError> {
        self.window.request_redraw();

        let frame = self.graph.begin_frame()?;

        let width = frame.extent.width;
        let height = frame.extent.height;
        let compute_pipe = self.compute_pipeline;
        let graphics_pipe = self.graphics_pipeline;
        let storage_image = self.storage_image;
        let sampler = self.sampler;

        self.graph
            .compute_pass("fill")
            .write((storage_image, Access::ComputeWrite))
            .execute(move |cmd, res| {
                cmd.bind_compute_pipeline(res.pipeline(compute_pipe));

                cmd.push_shader(&FillParams {
                    width,
                    height,
                    storage_idx: res.storage_index(storage_image),
                });

                cmd.dispatch(width.div_ceil(8), height.div_ceil(8), 1);
            });

        self.graph
            .render_pass("blit")
            .read((storage_image, Access::ShaderRead))
            .write((frame.backbuffer, Access::ColorAttachment))
            .execute(move |cmd, res| {
                cmd.bind_graphics_pipeline(res.pipeline(graphics_pipe));

                cmd.set_viewport_scissor(frame.extent);

                cmd.push_shader(&BlitParams {
                    sampled_idx: res.sampled_index(storage_image),
                    sampler_idx: res.sampler_index(sampler),
                });
                cmd.draw(3, 1);
            });

        self.graph.end_frame(frame)?;
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.graph.resize(width, height);
    }

    fn window(&self) -> &Window {
        &self.window
    }
}

fn main() {
    common::run::<State>();
}
