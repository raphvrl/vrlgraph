#[path = "../common/mod.rs"]
mod common;

use vrlgraph::graph::WithClearColor;
use vrlgraph::prelude::*;
use winit::window::Window;

struct State {
    graph: Graph,
    window: Window,
    pipeline: Pipeline,
}

impl common::Example for State {
    fn init(window: Window) -> Result<Self, GraphError> {
        let size = window.inner_size();

        let mut graph = Graph::builder()
            .window(&window)
            .size(size.width, size.height)
            .validation(cfg!(debug_assertions))
            .present_mode(PresentMode::Mailbox)
            .build()?;

        let vs = graph.shader_module("shaders/triangle.vert.spv", "main")?;
        let fs = graph.shader_module("shaders/triangle.frag.spv", "main")?;

        let pipeline = graph
            .graphics_pipeline("triangle")
            .vertex_shader(vs)
            .fragment_shader(fs)
            .build()?;

        Ok(Self {
            graph,
            window,
            pipeline,
        })
    }

    fn draw(&mut self) -> Result<(), GraphError> {
        self.window.request_redraw();

        let frame = self.graph.begin_frame()?;

        let pipeline = self.pipeline;
        let extent = frame.extent;
        let backbuffer = frame.backbuffer;

        self.graph
            .render_pass("triangle")
            .write(WithClearColor(
                backbuffer,
                Access::ColorAttachment,
                [0.1, 0.2, 0.3, 1.0],
            ))
            .execute(move |cmd, res| {
                cmd.bind_graphics_pipeline(res.pipeline(pipeline));
                cmd.set_viewport_scissor(extent);
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
