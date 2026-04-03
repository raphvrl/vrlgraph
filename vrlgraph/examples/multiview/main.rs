#[path = "../common/mod.rs"]
mod common;

use vrlgraph::ash::vk;
use vrlgraph::prelude::*;
use winit::window::Window;

#[derive(ShaderType)]
struct StereoParams {
    time: f32,
}

#[derive(ShaderType)]
struct ComposeParams {
    array_idx: u32,
    sampler_idx: u32,
}

struct State {
    graph: Graph,
    window: Window,
    stereo_pipeline: Pipeline,
    compose_pipeline: Pipeline,
    stereo_image: Image,
    sampler: Sampler,
    start: std::time::Instant,
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

        let stereo_image = graph
            .persistent_image("stereo_target")
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(512, 512)
            .array_2d(2)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .build()?;

        let sampler = graph
            .create_sampler()
            .filter(Filter::LINEAR)
            .address_mode_u(AddressMode::CLAMP_TO_EDGE)
            .address_mode_v(AddressMode::CLAMP_TO_EDGE)
            .build()?;

        let stereo_vs = graph.shader_module("shaders/stereo.vert.spv", "main")?;
        let stereo_fs = graph.shader_module("shaders/stereo.frag.spv", "main")?;
        let compose_vs = graph.shader_module("shaders/composite.vert.spv", "main")?;
        let compose_fs = graph.shader_module("shaders/composite.frag.spv", "main")?;

        let stereo_pipeline = graph
            .graphics_pipeline("stereo_render")
            .vertex_shader(stereo_vs)
            .fragment_shader(stereo_fs)
            .color_formats(&[vk::Format::R8G8B8A8_UNORM])
            .view_mask(0b11)
            .build()?;

        let compose_pipeline = graph
            .graphics_pipeline("composite")
            .vertex_shader(compose_vs)
            .fragment_shader(compose_fs)
            .build()?;

        Ok(Self {
            graph,
            window,
            stereo_pipeline,
            compose_pipeline,
            stereo_image,
            sampler,
            start: std::time::Instant::now(),
        })
    }

    fn draw(&mut self) -> Result<(), GraphError> {
        self.window.request_redraw();

        let frame = self.graph.begin_frame()?;
        let time = self.start.elapsed().as_secs_f32();

        let stereo_pipe = self.stereo_pipeline;
        let compose_pipe = self.compose_pipeline;
        let stereo_image = self.stereo_image;
        let sampler = self.sampler;
        let stereo_extent = vk::Extent2D {
            width: 512,
            height: 512,
        };

        self.graph
            .render_pass("stereo_geometry")
            .write((stereo_image, Access::ColorAttachment))
            .multiview(0b11)
            .execute(move |cmd, res| {
                cmd.bind_graphics_pipeline(res.pipeline(stereo_pipe));
                cmd.set_viewport_scissor(stereo_extent);
                cmd.push_shader(&StereoParams { time });
                cmd.draw(3, 1);
            });

        self.graph
            .render_pass("compose")
            .read((stereo_image, Access::ShaderRead))
            .write((frame.backbuffer, Access::ColorAttachment))
            .execute(move |cmd, res| {
                cmd.bind_graphics_pipeline(res.pipeline(compose_pipe));
                cmd.set_viewport_scissor(frame.extent);
                cmd.push_shader(&ComposeParams {
                    array_idx: res.array_index(stereo_image),
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
