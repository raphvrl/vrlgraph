use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use vrlgraph::graph::WithClearColor;
use vrlgraph::prelude::*;
use vrlgraph_egui::EguiRenderer;

struct State {
    graph: Graph,
    window: Window,
    egui_renderer: EguiRenderer,
    egui_state: egui_winit::State,
    name: String,
    age: u32,
}

impl State {
    fn new(window: Window) -> Result<Self, GraphError> {
        let size = window.inner_size();

        let mut graph = Graph::builder()
            .window(&window)
            .size(size.width, size.height)
            .validation(cfg!(debug_assertions))
            .present_mode(PresentMode::Fifo)
            .build()?;

        let egui_renderer = EguiRenderer::new(&mut graph)?;

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        Ok(Self {
            graph,
            window,
            egui_renderer,
            egui_state,
            name: "World".into(),
            age: 42,
        })
    }

    fn draw(&mut self) -> Result<(), GraphError> {
        let input = self.egui_state.take_egui_input(&self.window);
        let ppp = self.egui_state.egui_ctx().pixels_per_point();

        let name = &mut self.name;
        let age = &mut self.age;

        let output = self.egui_state.egui_ctx().run_ui(input, |ctx| {
            egui::CentralPanel::default().show_inside(ctx, |ui| {
                ui.heading("vrlgraph-egui demo");
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Your name: ");
                    ui.text_edit_singleline(name);
                });
                ui.add(egui::Slider::new(age, 0..=120).text("age"));
                if ui.button("Click me").clicked() {
                    println!("Hello '{}', age {}", name, age);
                }
                ui.separator();
                ui.label(format!("Hello '{}', age {}", name, age));
            });
        });

        self.egui_state
            .handle_platform_output(&self.window, output.platform_output);

        let primitives = self.egui_state.egui_ctx().tessellate(output.shapes, ppp);

        self.egui_renderer
            .prepare(&mut self.graph, &output.textures_delta)?;

        let frame = self.graph.begin_frame()?;

        self.graph
            .render_pass("clear")
            .write(WithClearColor(
                frame.backbuffer,
                Access::ColorAttachment,
                [0.1, 0.1, 0.1, 1.0],
            ))
            .execute(|_, _| {});

        self.egui_renderer
            .paint(&mut self.graph, &frame, &primitives, ppp)?;

        self.graph.end_frame(frame)?;
        Ok(())
    }
}

struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes().with_title("vrlgraph-egui demo"))
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

        let response = state.egui_state.on_window_event(&state.window, &event);
        if response.consumed {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.graph.resize(size.width, size.height),
            WindowEvent::RedrawRequested => match state.draw() {
                Ok(()) => {}
                Err(GraphError::SwapchainOutOfDate) => {
                    let size = state.window.inner_size();
                    state.graph.resize(size.width, size.height);
                }
                Err(e) => {
                    tracing::error!("draw error: {e}");
                    event_loop.exit();
                }
            },
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App { state: None };
    event_loop.run_app(&mut app).unwrap();
}
