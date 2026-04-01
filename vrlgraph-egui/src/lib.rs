#![doc = include_str!("../README.md")]

use std::collections::VecDeque;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use rustc_hash::FxHashMap;

use vrlgraph::prelude::*;

const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/egui.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/egui.frag.spv"));

const INITIAL_VERTEX_BYTES: u64 = 64 * 1024;
const INITIAL_INDEX_BYTES: u64 = 64 * 1024;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, VertexInput)]
struct EguiVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    color: [u8; 4],
}

#[derive(ShaderType)]
struct PushConstants {
    screen_size: [f32; 2],
    texture_index: u32,
    sampler_index: u32,
}

pub struct EguiRenderer {
    pipeline: Pipeline,
    sampler: Sampler,
    textures: FxHashMap<egui::TextureId, Image>,
    pending_frees: VecDeque<Vec<egui::TextureId>>,
    vertex_bufs: Vec<Buffer>,
    index_bufs: Vec<Buffer>,
    vertex_capacities: Vec<u64>,
    index_capacities: Vec<u64>,
    vertices: Vec<EguiVertex>,
    indices: Vec<u32>,
    draw_calls: Vec<DrawCall>,
}

impl EguiRenderer {
    pub fn new(graph: &mut Graph) -> Result<Self, GraphError> {
        let vs = graph.shader_module_from_spirv(VERT_SPV, "main")?;
        let fs = graph.shader_module_from_spirv(FRAG_SPV, "main")?;

        let pipeline = graph
            .graphics_pipeline("egui")
            .vertex_shader(vs)
            .fragment_shader(fs)
            .vertex_input::<EguiVertex>()
            .build()?;

        graph.destroy_shader_module(vs);
        graph.destroy_shader_module(fs);

        let sampler = graph
            .create_sampler()
            .filter(Filter::LINEAR)
            .address_mode(AddressMode::CLAMP_TO_EDGE)
            .build()?;

        let n = graph.frames_in_flight();

        let mut vertex_bufs = Vec::with_capacity(n);
        let mut index_bufs = Vec::with_capacity(n);
        for i in 0..n {
            vertex_bufs.push(graph.create_buffer(&BufferDesc {
                size: INITIAL_VERTEX_BYTES,
                usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                location: MemoryLocation::CpuToGpu,
                label: format!("egui_vertices_{i}"),
            })?);
            index_bufs.push(graph.create_buffer(&BufferDesc {
                size: INITIAL_INDEX_BYTES,
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                location: MemoryLocation::CpuToGpu,
                label: format!("egui_indices_{i}"),
            })?);
        }

        Ok(Self {
            pipeline,
            sampler,
            textures: FxHashMap::default(),
            pending_frees: (0..n).map(|_| Vec::new()).collect(),
            vertex_bufs,
            index_bufs,
            vertex_capacities: vec![INITIAL_VERTEX_BYTES; n],
            index_capacities: vec![INITIAL_INDEX_BYTES; n],
            vertices: Vec::new(),
            indices: Vec::new(),
            draw_calls: Vec::new(),
        })
    }

    /// Process texture uploads and frees. Must be called **before** [`Graph::begin_frame`].
    pub fn prepare(
        &mut self,
        graph: &mut Graph,
        textures_delta: &egui::TexturesDelta,
    ) -> Result<(), GraphError> {
        if let Some(to_free) = self.pending_frees.pop_front() {
            for id in to_free {
                if let Some(tex) = self.textures.remove(&id) {
                    graph.destroy_image(tex);
                }
            }
        }
        self.pending_frees.push_back(Vec::new());

        for (id, delta) in &textures_delta.set {
            self.apply_texture_delta(graph, *id, delta)?;
        }

        if let Some(back) = self.pending_frees.back_mut() {
            back.extend(textures_delta.free.iter().copied());
        }
        Ok(())
    }

    pub fn paint(
        &mut self,
        graph: &mut Graph,
        frame: &Frame,
        primitives: &[egui::ClippedPrimitive],
        pixels_per_point: f32,
    ) -> Result<(), GraphError> {
        self.tessellate(primitives);

        if self.vertices.is_empty() {
            return Ok(());
        }

        let fi = frame.index as usize;
        let vertex_byte_len = (self.vertices.len() * size_of::<EguiVertex>()) as u64;
        let index_byte_len = (self.indices.len() * size_of::<u32>()) as u64;

        self.ensure_buffer_capacity(graph, fi, vertex_byte_len, index_byte_len)?;

        graph.write_buffer(self.vertex_bufs[fi], &self.vertices);
        graph.write_buffer(self.index_bufs[fi], &self.indices);

        let pipeline = self.pipeline;
        let sampler = self.sampler;
        let vertex_buf = self.vertex_bufs[fi];
        let index_buf = self.index_bufs[fi];
        let screen_size = [
            frame.extent.width as f32 / pixels_per_point,
            frame.extent.height as f32 / pixels_per_point,
        ];
        let extent = frame.extent;

        let mut tex_images: Vec<(egui::Rect, Image, u32, u32, i32)> = Vec::new();
        for dc in &self.draw_calls {
            if let Some(&image) = self.textures.get(&dc.texture_id) {
                tex_images.push((
                    dc.clip_rect,
                    image,
                    dc.index_count,
                    dc.first_index,
                    dc.vertex_offset,
                ));
            } else {
                eprintln!(
                    "[vrlgraph-egui] missing texture {:?}, skipping draw call",
                    dc.texture_id
                );
            }
        }

        graph
            .render_pass("egui")
            .write(WithLoadOp(
                frame.backbuffer,
                Access::ColorAttachment,
                LoadOp::Load,
            ))
            .execute(move |cmd, res| {
                cmd.bind_graphics_pipeline(res.pipeline(pipeline));

                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                cmd.set_viewport(viewport);

                cmd.set_color_blend_enable(0, &[vk::TRUE]);
                cmd.set_color_blend_equation(
                    0,
                    &[vk::ColorBlendEquationEXT {
                        src_color_blend_factor: vk::BlendFactor::ONE,
                        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        color_blend_op: vk::BlendOp::ADD,
                        src_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_DST_ALPHA,
                        dst_alpha_blend_factor: vk::BlendFactor::ONE,
                        alpha_blend_op: vk::BlendOp::ADD,
                    }],
                );

                cmd.bind_vertex_buffer(res.buffer(vertex_buf), 0);
                cmd.bind_index_buffer(res.buffer(index_buf), 0);

                let sampler_idx = res.sampler_index(sampler);
                let max_w = extent.width as f32;
                let max_h = extent.height as f32;

                for &(clip_rect, image, index_count, first_index, vertex_offset) in &tex_images {
                    let x = (clip_rect.min.x * pixels_per_point)
                        .round()
                        .clamp(0.0, max_w) as i32;
                    let y = (clip_rect.min.y * pixels_per_point)
                        .round()
                        .clamp(0.0, max_h) as i32;
                    let w = ((clip_rect.max.x - clip_rect.min.x) * pixels_per_point)
                        .round()
                        .clamp(0.0, max_w) as u32;
                    let h = ((clip_rect.max.y - clip_rect.min.y) * pixels_per_point)
                        .round()
                        .clamp(0.0, max_h) as u32;

                    let w = w.min(extent.width.saturating_sub(x as u32));
                    let h = h.min(extent.height.saturating_sub(y as u32));

                    if w == 0 || h == 0 {
                        continue;
                    }

                    cmd.set_scissor(vk::Rect2D {
                        offset: vk::Offset2D { x, y },
                        extent: vk::Extent2D {
                            width: w,
                            height: h,
                        },
                    });

                    cmd.push_shader(&PushConstants {
                        screen_size,
                        texture_index: res.sampled_index(image),
                        sampler_index: sampler_idx,
                    });

                    cmd.draw_indexed(index_count, 1, first_index, vertex_offset);
                }
            });

        Ok(())
    }

    pub fn destroy(mut self, graph: &mut Graph) {
        for frees in &mut self.pending_frees {
            for id in frees.drain(..) {
                if let Some(tex) = self.textures.remove(&id) {
                    graph.destroy_image(tex);
                }
            }
        }
        for (_, tex) in self.textures.drain() {
            graph.destroy_image(tex);
        }
        for &buf in &self.vertex_bufs {
            graph.destroy_buffer(buf);
        }
        for &buf in &self.index_bufs {
            graph.destroy_buffer(buf);
        }
        graph.destroy_pipeline(self.pipeline);
        graph.destroy_sampler(self.sampler);
    }

    fn apply_texture_delta(
        &mut self,
        graph: &mut Graph,
        id: egui::TextureId,
        delta: &egui::epaint::ImageDelta,
    ) -> Result<(), GraphError> {
        let pixels = Self::image_data_to_rgba(&delta.image);
        let [w, h] = delta.image.size();

        if let Some(pos) = delta.pos {
            graph.upload_to_image(
                self.textures[&id],
                &pixels,
                [pos[0] as u32, pos[1] as u32],
                [w as u32, h as u32],
            )?;
        } else {
            if let Some(old) = self.textures.remove(&id) {
                graph.destroy_image(old);
            }

            let image = graph
                .persistent_image(format!("egui_tex_{id:?}"))
                .format(vk::Format::R8G8B8A8_SRGB)
                .extent(w as u32, h as u32)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .build()?;

            graph.upload_to_image(image, &pixels, [0, 0], [w as u32, h as u32])?;

            self.textures.insert(id, image);
        }

        Ok(())
    }

    fn image_data_to_rgba(image: &egui::epaint::ImageData) -> Vec<u8> {
        match image {
            egui::epaint::ImageData::Color(color_image) => {
                let mut rgba = Vec::with_capacity(color_image.pixels.len() * 4);
                for pixel in &color_image.pixels {
                    rgba.extend_from_slice(&pixel.to_array());
                }
                rgba
            }
        }
    }

    fn tessellate(&mut self, primitives: &[egui::ClippedPrimitive]) {
        self.vertices.clear();
        self.indices.clear();
        self.draw_calls.clear();

        for clipped in primitives {
            let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive else {
                continue;
            };

            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }

            let vertex_offset = self.vertices.len() as i32;
            let first_index = self.indices.len() as u32;

            for v in &mesh.vertices {
                self.vertices.push(EguiVertex {
                    pos: [v.pos.x, v.pos.y],
                    uv: [v.uv.x, v.uv.y],
                    color: v.color.to_array(),
                });
            }

            self.indices.extend_from_slice(&mesh.indices);

            self.draw_calls.push(DrawCall {
                clip_rect: clipped.clip_rect,
                texture_id: mesh.texture_id,
                index_count: mesh.indices.len() as u32,
                first_index,
                vertex_offset,
            });
        }
    }

    fn ensure_buffer_capacity(
        &mut self,
        graph: &mut Graph,
        fi: usize,
        vertex_bytes: u64,
        index_bytes: u64,
    ) -> Result<(), GraphError> {
        if vertex_bytes > self.vertex_capacities[fi] {
            graph.destroy_buffer(self.vertex_bufs[fi]);
            let new_cap = vertex_bytes.next_power_of_two();
            self.vertex_bufs[fi] = graph.create_buffer(&BufferDesc {
                size: new_cap,
                usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                location: MemoryLocation::CpuToGpu,
                label: format!("egui_vertices_{fi}"),
            })?;
            self.vertex_capacities[fi] = new_cap;
        }

        if index_bytes > self.index_capacities[fi] {
            graph.destroy_buffer(self.index_bufs[fi]);
            let new_cap = index_bytes.next_power_of_two();
            self.index_bufs[fi] = graph.create_buffer(&BufferDesc {
                size: new_cap,
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                location: MemoryLocation::CpuToGpu,
                label: format!("egui_indices_{fi}"),
            })?;
            self.index_capacities[fi] = new_cap;
        }

        Ok(())
    }
}

struct DrawCall {
    clip_rect: egui::Rect,
    texture_id: egui::TextureId,
    index_count: u32,
    first_index: u32,
    vertex_offset: i32,
}
