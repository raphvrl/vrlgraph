#[path = "../common/mod.rs"]
mod common;

use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec2, Vec3};
use vrlgraph::prelude::*;
use winit::window::{Window, WindowAttributes};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, VertexInput)]
struct Vertex {
    pos: Vec2,
}

#[derive(ShaderType)]
struct Transform {
    matrix: Mat4,
}

#[derive(ShaderType)]
struct PC {
    transform_addr: u64,
    colors_addr: u64,
}

const VERTICES: [Vertex; 4] = [
    Vertex { pos: Vec2::new(-0.5, -0.5) },
    Vertex { pos: Vec2::new(0.5, -0.5) },
    Vertex { pos: Vec2::new(0.5, 0.5) },
    Vertex { pos: Vec2::new(-0.5, 0.5) },
];

const INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

const COLORS: [[f32; 4]; 4] = [
    [1.0, 0.2, 0.2, 1.0],
    [0.2, 1.0, 0.2, 1.0],
    [0.2, 0.2, 1.0, 1.0],
    [1.0, 1.0, 0.2, 1.0],
];

struct State {
    graph: Graph,
    window: Window,
    pipeline: Pipeline,
    vertex_buf: Buffer,
    index_buf: Buffer,
    transform_buf: Buffer,
    colors_buf: Buffer,
    start: Instant,
}

impl common::Example for State {
    fn window_attributes() -> WindowAttributes {
        Window::default_attributes().with_title("vrlgraph — buffers")
    }

    fn init(window: Window) -> Result<Self, GraphError> {
        let size = window.inner_size();

        let mut graph = Graph::builder()
            .window(&window)
            .size(size.width, size.height)
            .validation(cfg!(debug_assertions))
            .present_mode(PresentMode::Fifo)
            .build()?;

        let vertex_buf = graph.vertex_buffer("quad_verts", &VERTICES)?;
        let index_buf = graph.index_buffer("quad_indices", &INDICES)?;

        let transform_buf = graph.uniform_shader("transform", &Transform {
            matrix: Mat4::IDENTITY,
        })?;
        let colors_buf = graph.storage_buffer("colors", &COLORS)?;

        let vs = graph.shader_module("shaders/mesh.vert.spv", "main")?;
        let fs = graph.shader_module("shaders/mesh.frag.spv", "main")?;

        let pipeline = graph
            .graphics_pipeline("buffers")
            .vertex_shader(vs)
            .fragment_shader(fs)
            .vertex_input::<Vertex>()
            .build()?;

        Ok(Self {
            graph,
            window,
            pipeline,
            vertex_buf,
            index_buf,
            transform_buf,
            colors_buf,
            start: Instant::now(),
        })
    }

    fn draw(&mut self) -> Result<(), GraphError> {
        self.window.request_redraw();

        let angle = self.start.elapsed().as_secs_f32();
        let matrix = Mat4::from_scale_rotation_translation(
            Vec3::splat(0.8),
            Quat::from_rotation_z(angle),
            Vec3::ZERO,
        );

        self.graph.write_shader(self.transform_buf, &Transform { matrix });

        let frame = self.graph.begin_frame()?;

        let transform_addr = self.graph.buffer_device_address(self.transform_buf);
        let colors_addr = self.graph.buffer_device_address(self.colors_buf);

        let pipeline = self.pipeline;
        let vertex_buf = self.vertex_buf;
        let index_buf = self.index_buf;

        self.graph
            .render_pass("mesh")
            .read((vertex_buf, BufferUsage::VertexRead))
            .read((index_buf, BufferUsage::IndexRead))
            .read((self.transform_buf, BufferUsage::UniformRead))
            .read((self.colors_buf, BufferUsage::StorageRead))
            .write((frame.backbuffer, Access::ColorAttachment))
            .execute(move |cmd, res| {
                cmd.bind_graphics_pipeline(res.pipeline(pipeline));
                cmd.set_viewport_scissor(frame.extent);

                cmd.bind_vertex_buffer(res.buffer(vertex_buf), 0);
                cmd.bind_index_buffer(res.buffer(index_buf), 0);

                cmd.push_shader(&PC {
                    transform_addr,
                    colors_addr,
                });

                cmd.draw_indexed(INDICES.len() as u32, 1, 0, 0);
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
