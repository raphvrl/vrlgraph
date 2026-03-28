use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use vrlgraph::prelude::*;

const PALETTE: [[f32; 4]; 8] = [
    [0.05, 0.05, 0.20, 1.0],
    [0.15, 0.10, 0.50, 1.0],
    [0.40, 0.10, 0.70, 1.0],
    [0.80, 0.20, 0.50, 1.0],
    [0.95, 0.40, 0.10, 1.0],
    [0.95, 0.75, 0.10, 1.0],
    [0.60, 0.95, 0.30, 1.0],
    [0.20, 0.90, 0.80, 1.0],
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstants {
    width: u32,
    height: u32,
    palette_addr: u64,
}

struct State {
    graph: Graph,
    window: Window,
    compute_pipeline: PipelineHandle,
    graphics_pipeline: PipelineHandle,
    storage_image: GraphImage,
    palette_buffer: BufferHandle,
    compute_set: DynamicDescriptorSet,
    graphics_set: DynamicDescriptorSet,
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

        let palette_buffer = graph
            .create_buffer(&BufferDesc {
                size: (PALETTE.len() * std::mem::size_of::<[f32; 4]>()) as vk::DeviceSize,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                location: MemoryLocation::CpuToGpu,
                label: "palette".to_string(),
            })
            .unwrap();

        graph
            .get_buffer(palette_buffer)
            .unwrap()
            .write(bytemuck::cast_slice::<[f32; 4], u8>(&PALETTE));

        let storage_image = graph
            .create_resizable(|ext| ImageDesc {
                extent: vk::Extent3D {
                    width: ext.width,
                    height: ext.height,
                    depth: 1,
                },
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                label: "bda_output".to_string(),
                ..Default::default()
            })
            .unwrap();

        let sampler = graph
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            )
            .unwrap();

        let compute_set = graph
            .descriptor_set()
            .storage_image(vk::ShaderStageFlags::COMPUTE, storage_image)
            .build_dynamic()
            .unwrap();

        let graphics_set = graph
            .descriptor_set()
            .combined_image_sampler(vk::ShaderStageFlags::FRAGMENT, sampler, storage_image)
            .build_dynamic()
            .unwrap();

        let compute_pipeline = graph
            .compute_pipeline()
            .shader("shaders/bda_fill.comp.spv")
            .unwrap()
            .push_constants::<PushConstants>(vk::ShaderStageFlags::COMPUTE)
            .descriptor_set_layouts(&[compute_set.layout])
            .build()
            .unwrap();

        let graphics_pipeline = graph
            .graphics_pipeline()
            .vertex_shader("shaders/fullscreen.vert.spv")
            .unwrap()
            .fragment_shader("shaders/blit.frag.spv")
            .unwrap()
            .descriptor_set_layouts(&[graphics_set.layout])
            .build()
            .unwrap();

        Self {
            graph,
            window,
            compute_pipeline,
            graphics_pipeline,
            storage_image,
            palette_buffer,
            compute_set,
            graphics_set,
        }
    }

    fn draw(&mut self) -> Result<(), GraphError> {
        self.window.request_redraw();

        let frame = self.graph.begin_frame()?;

        if frame.resized {
            self.compute_set.update(&self.graph);
            self.graphics_set.update(&self.graph);
        }

        let palette_addr = self
            .graph
            .buffer_device_address(self.palette_buffer)
            .expect("palette buffer was not created with SHADER_DEVICE_ADDRESS");

        let width = frame.extent.width;
        let height = frame.extent.height;
        let compute_pipe = self.compute_pipeline;
        let graphics_pipe = self.graphics_pipeline;
        let storage_image = self.storage_image;
        let compute_set = self.compute_set.set;
        let graphics_set = self.graphics_set.set;

        self.graph
            .compute_pass("bda_fill")
            .write((storage_image, Access::ComputeWrite))
            .execute(move |cmd, res| {
                cmd.bind_compute_pipeline(res.pipeline(compute_pipe));
                cmd.bind_descriptor_sets(0, &[compute_set]);

                let pc = PushConstants {
                    width,
                    height,
                    palette_addr,
                };

                cmd.push_constants(vk::ShaderStageFlags::COMPUTE, bytemuck::bytes_of(&pc));
                cmd.dispatch(width.div_ceil(8), height.div_ceil(8), 1);
            });

        self.graph
            .render_pass("blit")
            .read((storage_image, Access::ShaderRead))
            .write((frame.backbuffer, Access::ColorAttachment))
            .execute(move |cmd, res| {
                cmd.bind_graphics_pipeline(res.pipeline(graphics_pipe));
                cmd.set_viewport_scissor(frame.extent);
                cmd.bind_descriptor_sets(0, &[graphics_set]);
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
            .create_window(Window::default_attributes().with_title("vrlgraph — bda"))
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
