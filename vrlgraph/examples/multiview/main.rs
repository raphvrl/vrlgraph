#[path = "../common/mod.rs"]
mod common;

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
    graph: gpu::Graph,
    window: Window,
    stereo_pipeline: gpu::Pipeline,
    compose_pipeline: gpu::Pipeline,
    stereo_image: gpu::Image,
    sampler: gpu::Sampler,
    start: std::time::Instant,
}

impl common::Example for State {
    fn init(window: Window) -> Result<Self, gpu::GraphError> {
        let size = window.inner_size();

        let mut graph = gpu::Graph::builder()
            .window(&window)
            .size(size.width, size.height)
            .validation(cfg!(debug_assertions))
            .present_mode(gpu::PresentMode::Fifo)
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
            .filter(gpu::Filter::LINEAR)
            .address_mode_u(gpu::AddressMode::CLAMP_TO_EDGE)
            .address_mode_v(gpu::AddressMode::CLAMP_TO_EDGE)
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

    fn draw(&mut self) -> Result<(), gpu::GraphError> {
        self.window.request_redraw();

        let time = self.start.elapsed().as_secs_f32();

        let mut frame = self.graph.begin_frame()?;
        let backbuffer = frame.backbuffer;
        let extent = frame.extent;

        let stereo_extent = vk::Extent2D {
            width: 512,
            height: 512,
        };

        frame
            .render_pass("stereo_geometry")
            .write((self.stereo_image, gpu::Access::ColorAttachment))
            .multiview(0b11)
            .execute(|cmd| {
                cmd.bind_graphics_pipeline(self.stereo_pipeline);
                cmd.set_viewport_scissor(stereo_extent);
                cmd.push_constants(&StereoParams { time });
                cmd.draw(3, 1);
            });

        frame
            .render_pass("compose")
            .read((self.stereo_image, gpu::Access::ShaderRead))
            .write((backbuffer, gpu::Access::ColorAttachment))
            .execute(|cmd| {
                cmd.bind_graphics_pipeline(self.compose_pipeline);
                cmd.set_viewport_scissor(extent);
                cmd.push_constants(&ComposeParams {
                    array_idx: cmd.array_index(self.stereo_image),
                    sampler_idx: cmd.sampler_index(self.sampler),
                });
                cmd.draw(3, 1);
            });

        frame.submit()?;
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
