#[path = "../common/mod.rs"]
mod common;

use vrlgraph::prelude::*;
use winit::window::Window;

#[derive(ShaderType)]
struct FillParams {
    color: [f32; 4],
    layer: u32,
}

#[derive(ShaderType)]
struct CompositeParams {
    array_idx: u32,
    sampler_idx: u32,
}

const LAYER_NAMES: [&str; 4] = ["fill_0", "fill_1", "fill_2", "fill_3"];
const LAYER_COLORS: [[f32; 4]; 4] = [
    [0.8, 0.2, 0.2, 1.0],
    [0.2, 0.8, 0.2, 1.0],
    [0.2, 0.2, 0.8, 1.0],
    [0.8, 0.8, 0.2, 1.0],
];

struct State {
    graph: gpu::Graph,
    window: Window,
    fill_pipeline: gpu::Pipeline,
    composite_pipeline: gpu::Pipeline,
    array_image: gpu::Image,
    sampler: gpu::Sampler,
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

        let array_image = graph
            .persistent_image("layer_array")
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(256, 256)
            .array_2d(4)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .build()?;

        let sampler = graph
            .create_sampler()
            .filter(gpu::Filter::NEAREST)
            .address_mode_u(gpu::AddressMode::CLAMP_TO_EDGE)
            .address_mode_v(gpu::AddressMode::CLAMP_TO_EDGE)
            .build()?;

        let vs = graph.shader_module("shaders/screen.vert.spv", "main")?;
        let fill_fs = graph.shader_module("shaders/fill.frag.spv", "main")?;
        let composite_fs = graph.shader_module("shaders/compose.frag.spv", "main")?;

        let fill_pipeline = graph
            .graphics_pipeline("fill_layer")
            .vertex_shader(vs)
            .fragment_shader(fill_fs)
            .color_formats(&[vk::Format::R8G8B8A8_UNORM])
            .build()?;

        let composite_pipeline = graph
            .graphics_pipeline("composite")
            .vertex_shader(vs)
            .fragment_shader(composite_fs)
            .build()?;

        Ok(Self {
            graph,
            window,
            fill_pipeline,
            composite_pipeline,
            array_image,
            sampler,
        })
    }

    fn draw(&mut self) -> Result<(), gpu::GraphError> {
        self.window.request_redraw();

        let mut frame = self.graph.begin_frame()?;
        let backbuffer = frame.backbuffer;
        let extent = frame.extent;

        let layer_extent = vk::Extent2D {
            width: 256,
            height: 256,
        };

        let fill_pipe = self.fill_pipeline;
        let array_image = self.array_image;

        for i in 0..4u32 {
            let color = LAYER_COLORS[i as usize];

            frame
                .render_pass(LAYER_NAMES[i as usize])
                .write(gpu::WithLayerClearColor(
                    array_image,
                    gpu::Access::ColorAttachment,
                    color,
                    i,
                ))
                .execute(move |cmd| {
                    cmd.bind_graphics_pipeline(fill_pipe);
                    cmd.set_viewport_scissor(layer_extent);
                    cmd.push_constants(&FillParams { color, layer: i });
                    cmd.draw(3, 1);
                });
        }

        frame
            .render_pass("composite")
            .read((self.array_image, gpu::Access::ShaderRead))
            .write((backbuffer, gpu::Access::ColorAttachment))
            .execute(|cmd| {
                cmd.bind_graphics_pipeline(self.composite_pipeline);
                cmd.set_viewport_scissor(extent);
                cmd.push_constants(&CompositeParams {
                    array_idx: cmd.array_index(self.array_image),
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
