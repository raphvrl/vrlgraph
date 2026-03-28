use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use bytemuck::{Pod, Zeroable};

use vrlgraph::prelude::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FillParams {
    width: u32,
    height: u32,
    storage_idx: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlitParams {
    sampled_idx: u32,
    sampler_idx: u32,
}

struct State {
    graph: Graph,
    window: Window,
    compute_pipeline: PipelineHandle,
    graphics_pipeline: PipelineHandle,
    storage_image: Image,
    sampler: Sampler,
}

impl State {
    fn new(window: Window) -> Self {
        let size = window.inner_size();

        let mut graph = Graph::builder()
            .window(&window)
            .size(size.width, size.height)
            .validation(cfg!(debug_assertions))
            .present_mode(PresentMode::Fifo)
            .build()
            .unwrap();

        let storage_image = graph
            .create_resizable(|ext| ImageDesc {
                extent: vk::Extent3D {
                    width: ext.width,
                    height: ext.height,
                    depth: 1,
                },
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                label: "triangle_storage".to_string(),
                ..Default::default()
            })
            .unwrap();

        let sampler = graph
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            )
            .unwrap();

        let compute_pipeline = graph
            .compute_pipeline()
            .shader("shaders/fill.comp.spv")
            .unwrap()
            .build()
            .unwrap();

        let graphics_pipeline = graph
            .graphics_pipeline()
            .vertex_shader("shaders/fullscreen.vert.spv")
            .unwrap()
            .fragment_shader("shaders/blit.frag.spv")
            .unwrap()
            .build()
            .unwrap();

        Self {
            graph,
            window,
            compute_pipeline,
            graphics_pipeline,
            storage_image,
            sampler,
        }
    }

    fn draw(&mut self) -> Result<(), GraphError> {
        self.window.request_redraw();

        let frame = self.graph.begin_frame()?;

        let width = frame.extent.width;
        let height = frame.extent.height;
        let compute_pipe = self.compute_pipeline;
        let graphics_pipe = self.graphics_pipeline;
        let storage_image = self.storage_image;
        let sampler_idx = self.sampler.index;

        self.graph
            .compute_pass("fill")
            .write((storage_image, Access::ComputeWrite))
            .execute(move |cmd, res| {
                cmd.bind_compute_pipeline(res.pipeline(compute_pipe));

                let params = FillParams {
                    width,
                    height,
                    storage_idx: res.storage_index(storage_image).0,
                    _pad: 0,
                };

                cmd.push_constants(bytemuck::bytes_of(&params));

                cmd.dispatch(width.div_ceil(8), height.div_ceil(8), 1);
            });

        self.graph
            .render_pass("blit")
            .read((storage_image, Access::ShaderRead))
            .write((frame.backbuffer, Access::ColorAttachment))
            .execute(move |cmd, res| {
                cmd.bind_graphics_pipeline(res.pipeline(graphics_pipe));

                cmd.set_viewport_scissor(frame.extent);

                let params = BlitParams {
                    sampled_idx: res.sampled_index(storage_image).0,
                    sampler_idx,
                };

                cmd.push_constants(bytemuck::bytes_of(&params));
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
        self.state = Some(State::new(window));
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
