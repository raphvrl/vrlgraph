use vrlgraph::graph::WithClearColor;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use vrlgraph::prelude::*;

struct State {
    graph: Graph,
    window: Window,
    pipeline: Pipeline,
}

impl State {
    fn new(window: Window) -> Result<Self, GraphError> {
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

        self.graph.end_frame()?;
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.graph.resize(width, height);
    }
}

struct App {
    state: Option<State>,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();
        match State::new(window) {
            Ok(state) => self.state = Some(state),
            Err(e) => {
                tracing::error!("init error: {e}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height);
            }
            WindowEvent::RedrawRequested => match state.draw() {
                Ok(()) => {}
                Err(GraphError::SwapchainOutOfDate) => {
                    let size = state.window.inner_size();
                    state.resize(size.width, size.height);
                }
                Err(e) => {
                    tracing::error!("draw error: {e}");
                    event_loop.exit();
                }
            },
            _ => {}
        }
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
