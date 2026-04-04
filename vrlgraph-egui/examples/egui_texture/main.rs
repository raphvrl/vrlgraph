use ash::vk;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use vrlgraph::graph::WithClearColor;
use vrlgraph::prelude::*;
use vrlgraph_egui::EguiRenderer;

const OFFSCREEN_SIZE: u32 = 256;

struct State {
    graph: Graph,
    window: Window,
    egui_renderer: EguiRenderer,
    egui_state: egui_winit::State,
    triangle_pipeline: Pipeline,
    offscreen_image: Image,
    offscreen_texture_id: egui::TextureId,
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

        let mut egui_renderer = EguiRenderer::new(&mut graph)?;

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let vs = graph.shader_module("shaders/egui_triangle.vert.spv", "main")?;
        let fs = graph.shader_module("shaders/egui_triangle.frag.spv", "main")?;

        let triangle_pipeline = graph
            .graphics_pipeline("triangle")
            .vertex_shader(vs)
            .fragment_shader(fs)
            .color_formats(&[vk::Format::R8G8B8A8_SRGB])
            .build()?;

        graph.destroy_shader_module(vs);
        graph.destroy_shader_module(fs);

        let offscreen_image = graph
            .persistent_image("offscreen")
            .format(vk::Format::R8G8B8A8_SRGB)
            .extent(OFFSCREEN_SIZE, OFFSCREEN_SIZE)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .build()?;

        let offscreen_texture_id = egui_renderer.register_texture(offscreen_image);

        Ok(Self {
            graph,
            window,
            egui_renderer,
            egui_state,
            triangle_pipeline,
            offscreen_image,
            offscreen_texture_id,
        })
    }

    fn draw(&mut self) -> Result<(), GraphError> {
        let input = self.egui_state.take_egui_input(&self.window);
        let ppp = self.egui_state.egui_ctx().pixels_per_point();

        let texture_id = self.offscreen_texture_id;

        let output = self.egui_state.egui_ctx().run_ui(input, |ctx| {
            egui::CentralPanel::default().show_inside(ctx, |ui| {
                ui.heading("Render to egui");
                ui.separator();
                ui.label("Triangle rendered to an offscreen image and displayed in egui:");
                ui.image(egui::load::SizedTexture::new(
                    texture_id,
                    [OFFSCREEN_SIZE as f32, OFFSCREEN_SIZE as f32],
                ));
            });
        });

        self.egui_state
            .handle_platform_output(&self.window, output.platform_output);

        let primitives = self.egui_state.egui_ctx().tessellate(output.shapes, ppp);

        self.egui_renderer
            .prepare(&mut self.graph, &output.textures_delta)?;

        let frame = self.graph.begin_frame()?;

        let pipeline = self.triangle_pipeline;
        let offscreen = self.offscreen_image;
        let offscreen_extent = vk::Extent2D {
            width: OFFSCREEN_SIZE,
            height: OFFSCREEN_SIZE,
        };

        self.graph
            .render_pass("offscreen_triangle")
            .write(WithClearColor(
                offscreen,
                Access::ColorAttachment,
                [0.0, 0.0, 0.2, 1.0],
            ))
            .execute(move |cmd, res| {
                cmd.bind_graphics_pipeline(res.pipeline(pipeline));
                cmd.set_viewport_scissor(offscreen_extent);
                cmd.draw(3, 1);
            });

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
            .create_window(
                Window::default_attributes().with_title("vrlgraph-egui: render to egui"),
            )
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
