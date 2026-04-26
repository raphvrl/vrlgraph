//! Runs a compute shader that writes `i * 2` into a GPU-only SSBO, reads the
//! buffer back to CPU via [`gpu::FrameBuilder::readback_buffer`], and prints
//! the result.
//!
//! Run with: `cargo run --example compute_readback`

use gpu_allocator::MemoryLocation;
use vrlgraph::prelude::*;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

const COUNT: u32 = 256;

#[derive(ShaderType)]
struct FillParams {
    addr: u64,
    count: u32,
}

struct State {
    graph: gpu::Graph,
    window: Window,
    pipeline: gpu::Pipeline,
    ssbo: gpu::Buffer,
    done: bool,
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

        let cs = graph.shader_module("shaders/fill_buffer.comp.spv", "main")?;
        let pipeline = graph.compute_pipeline("fill_buffer").shader(cs).build()?;

        let bytes = u64::from(COUNT) * std::mem::size_of::<u32>() as u64;
        let ssbo = graph.create_buffer(&gpu::BufferDesc {
            size: bytes,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            location: MemoryLocation::GpuOnly,
            label: "fill_target".into(),
        })?;

        Ok(Self {
            graph,
            window,
            pipeline,
            ssbo,
            done: false,
        })
    }

    fn draw(&mut self) -> Result<(), gpu::GraphError> {
        let mut frame = self.graph.begin_frame()?;
        let bb = frame.backbuffer;
        let pipeline = self.pipeline;
        let ssbo = self.ssbo;
        let params = FillParams {
            addr: ssbo.address(),
            count: COUNT,
        };

        frame
            .compute_pass("fill_buffer")
            .write((ssbo, gpu::BufferUsage::StorageWrite))
            .execute(move |cmd| {
                cmd.bind_compute_pipeline(pipeline);
                cmd.push_constants(&params);
                cmd.dispatch(COUNT.div_ceil(64), 1, 1);
            });

        // The example needs a write to the swapchain image to keep the frame
        // alive (the DAG culls passes that don't reach a presented image
        // unless they write a buffer — our compute pass does, so it survives,
        // but we still need to transition the backbuffer to PRESENT_SRC).
        frame
            .render_pass("clear")
            .write(gpu::WithClearColor(
                bb,
                gpu::Access::ColorAttachment,
                [0.0, 0.0, 0.0, 1.0],
            ))
            .execute(|_| {});

        let bins = if !self.done {
            Some(frame.readback_buffer(ssbo))
        } else {
            None
        };

        frame.submit()?;

        if let Some(bins) = bins {
            let counts = bins.wait_as::<u32>(&self.graph);
            let mismatches = (0..COUNT as usize)
                .filter(|&i| counts[i] != (i as u32) * 2)
                .count();
            if mismatches == 0 {
                println!(
                    "compute_readback: {COUNT} values verified, first 8 = {:?}",
                    &counts[..8]
                );
            } else {
                println!(
                    "compute_readback: {mismatches} mismatches out of {COUNT}, first 8 = {:?}",
                    &counts[..8]
                );
            }
            self.done = true;
        }

        Ok(())
    }
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
            .with_title("vrlgraph compute_readback")
            .with_inner_size(winit::dpi::LogicalSize::new(320u32, 200u32));
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
                    if state.done {
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
            && !state.done
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
