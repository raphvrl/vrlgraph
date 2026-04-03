#[path = "../common/mod.rs"]
mod common;

use vrlgraph::ash::vk;
use vrlgraph::graph::WithLayerClearColor;
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
    graph: Graph,
    window: Window,
    fill_pipeline: Pipeline,
    composite_pipeline: Pipeline,
    array_image: Image,
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

        let array_image = graph
            .persistent_image("layer_array")
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(256, 256)
            .array_2d(4)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .build()?;

        let sampler = graph
            .create_sampler()
            .filter(Filter::NEAREST)
            .address_mode_u(AddressMode::CLAMP_TO_EDGE)
            .address_mode_v(AddressMode::CLAMP_TO_EDGE)
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

    fn draw(&mut self) -> Result<(), GraphError> {
        self.window.request_redraw();

        let frame = self.graph.begin_frame()?;

        let fill_pipe = self.fill_pipeline;
        let composite_pipe = self.composite_pipeline;
        let array_image = self.array_image;
        let sampler = self.sampler;
        let layer_extent = vk::Extent2D {
            width: 256,
            height: 256,
        };

        for i in 0..4u32 {
            let color = LAYER_COLORS[i as usize];

            self.graph
                .render_pass(LAYER_NAMES[i as usize])
                .write(WithLayerClearColor(
                    array_image,
                    Access::ColorAttachment,
                    color,
                    i,
                ))
                .execute(move |cmd, res| {
                    cmd.bind_graphics_pipeline(res.pipeline(fill_pipe));
                    cmd.set_viewport_scissor(layer_extent);
                    cmd.push_shader(&FillParams { color, layer: i });
                    cmd.draw(3, 1);
                });
        }

        self.graph
            .render_pass("composite")
            .read((array_image, Access::ShaderRead))
            .write((frame.backbuffer, Access::ColorAttachment))
            .execute(move |cmd, res| {
                cmd.bind_graphics_pipeline(res.pipeline(composite_pipe));
                cmd.set_viewport_scissor(frame.extent);
                cmd.push_shader(&CompositeParams {
                    array_idx: res.array_index(array_image),
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
