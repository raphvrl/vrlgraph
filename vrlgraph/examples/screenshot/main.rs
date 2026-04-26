//! Renders a triangle, reads the backbuffer back to CPU, and saves it as a PNG.
//!
//! Run with: `cargo run --example screenshot`

use vrlgraph::prelude::*;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

const OUTPUT_PATH: &str = "screenshot.png";

struct State {
    graph: gpu::Graph,
    window: Window,
    pipeline: gpu::Pipeline,
    saved: bool,
}

impl State {
    fn init(window: Window) -> Result<Self, gpu::GraphError> {
        let size = window.inner_size();

        let mut graph = gpu::Graph::builder()
            .window(&window)
            .size(size.width, size.height)
            .validation(cfg!(debug_assertions))
            .present_mode(gpu::PresentMode::Fifo)
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
            saved: false,
        })
    }

    fn draw(&mut self) -> Result<(), gpu::GraphError> {
        let mut frame = self.graph.begin_frame()?;
        let bb = frame.backbuffer;
        let extent = frame.extent;
        let pipeline = self.pipeline;

        frame
            .render_pass("triangle")
            .write(gpu::WithClearColor(
                bb,
                gpu::Access::ColorAttachment,
                [0.1, 0.2, 0.3, 1.0],
            ))
            .execute(move |cmd| {
                cmd.bind_graphics_pipeline(pipeline);
                cmd.set_viewport_scissor(extent);
                cmd.draw(3, 1);
            });

        let shot = if !self.saved {
            Some(frame.readback_image(bb)?)
        } else {
            None
        };

        frame.submit()?;

        if let Some(shot) = shot {
            let data = shot.wait(&self.graph);
            let rgba = pack_rows_to_rgba(
                data.bytes,
                data.row_pitch,
                data.width,
                data.height,
                data.format,
            );

            match image::RgbaImage::from_raw(data.width, data.height, rgba) {
                Some(img) => {
                    if let Err(e) = img.save(OUTPUT_PATH) {
                        tracing::error!("failed to save {OUTPUT_PATH}: {e}");
                    } else {
                        tracing::info!(
                            "wrote {OUTPUT_PATH} ({} x {}, format {:?})",
                            data.width,
                            data.height,
                            data.format
                        );
                    }
                }
                None => tracing::error!("failed to construct RgbaImage from raw bytes"),
            }
            self.saved = true;
        }

        Ok(())
    }
}

/// Re-packs a `row_pitch * height` byte slice into a tight `width * height * 4`
/// RGBA buffer, swapping channels if the source format is BGRA.
fn pack_rows_to_rgba(
    bytes: &[u8],
    row_pitch: u32,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Vec<u8> {
    let row_pitch = row_pitch as usize;
    let width = width as usize;
    let height = height as usize;
    let mut out = vec![0u8; width * height * 4];

    let swap = matches!(
        format,
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB
    );

    for y in 0..height {
        let src = &bytes[y * row_pitch..y * row_pitch + width * 4];
        let dst = &mut out[y * width * 4..(y + 1) * width * 4];
        if swap {
            for x in 0..width {
                dst[x * 4] = src[x * 4 + 2];
                dst[x * 4 + 1] = src[x * 4 + 1];
                dst[x * 4 + 2] = src[x * 4];
                dst[x * 4 + 3] = src[x * 4 + 3];
            }
        } else {
            dst.copy_from_slice(src);
        }
    }

    out
}

struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("vrlgraph screenshot")
            .with_inner_size(winit::dpi::LogicalSize::new(800u32, 600u32));
        let window = event_loop.create_window(attrs).unwrap();
        match State::init(window) {
            Ok(s) => self.state = Some(s),
            Err(e) => {
                tracing::error!("init error: {e}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => match state.draw() {
                Ok(()) => {
                    if state.saved {
                        event_loop.exit();
                    }
                }
                Err(gpu::GraphError::SwapchainOutOfDate) => {
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
        if let Some(state) = &self.state
            && !state.saved
        {
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
