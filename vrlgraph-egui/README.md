# vrlgraph-egui

An egui rendering backend for vrlgraph. Takes egui's tessellated output and draws it using vrlgraph's render graph, bindless descriptors, and dynamic rendering. Handles texture management, buffer resizing, and DPI scaling so you only deal with egui's immediate-mode API.

vrlgraph-egui is not a windowing or input integration. It renders egui primitives — nothing more. For window events and input handling, pair it with `egui-winit` or another platform backend.

---

## Prerequisites

- **Rust 1.85+** (edition 2024)
- **Vulkan SDK** with `glslc` on PATH (required at build time for shader compilation)

---

## Installation

```toml
[dependencies]
vrlgraph = { git = "https://github.com/raphvrl/vrlgraph" }
vrlgraph-egui = { git = "https://github.com/raphvrl/vrlgraph" }
egui = "0.31"
```

For windowing and input, add the platform integration as dev or regular dependencies:

```toml
winit = "0.30.13"
egui-winit = "0.31"
```

---

## Quick start

```rust,ignore
use vrlgraph::graph::WithClearColor;
use vrlgraph::prelude::*;
use vrlgraph_egui::EguiRenderer;

// Initialization
let mut graph = Graph::builder()
    .window(&window)
    .size(size.width, size.height)
    .build()?;

let mut egui_renderer = EguiRenderer::new(&mut graph)?;
let egui_ctx = egui::Context::default();

// Frame loop
loop {
    let input = egui_state.take_egui_input(&window);
    let ppp = egui_ctx.pixels_per_point();

    let output = egui_ctx.run(input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Hello from vrlgraph-egui");
        });
    });

    let primitives = egui_ctx.tessellate(output.shapes, ppp);

    // prepare() uploads textures — must be called BEFORE begin_frame
    egui_renderer.prepare(&mut graph, &output.textures_delta)?;

    let frame = graph.begin_frame()?;

    graph.render_pass("clear")
        .write(WithClearColor(
            frame.backbuffer,
            Access::ColorAttachment,
            [0.1, 0.1, 0.1, 1.0],
        ))
        .execute(|_, _| {});

    egui_renderer.paint(&mut graph, &frame, &primitives, ppp)?;

    graph.end_frame(frame)?;
}

// Cleanup
egui_renderer.destroy(&mut graph);
```

---

## API reference

### `EguiRenderer`

The sole public type. Owns the graphics pipeline, sampler, texture map, and dynamic vertex/index buffers needed to render egui output.

| Method | Description |
|---|---|
| `new(graph: &mut Graph) -> Result<Self, GraphError>` | Creates the graphics pipeline from embedded SPIR-V shaders, allocates a linear sampler, and provisions initial 64 KB vertex and index buffers. |
| `register_texture(&mut self, image: Image) -> egui::TextureId` | Registers a vrlgraph `Image` for use in egui widgets. The image must have `SAMPLED` usage. Returns a `TextureId::User` usable with `egui::Image` or `ui.image()`. Ownership stays with the caller. |
| `update_texture(&mut self, id: egui::TextureId, image: Image)` | Replaces the image behind an existing user texture ID (e.g. after recreating a render target on resize). The old image is not destroyed. |
| `unregister_texture(&mut self, id: egui::TextureId)` | Removes a user texture mapping. The image is not destroyed. |
| `prepare(&mut self, graph: &mut Graph, textures_delta: &egui::TexturesDelta) -> Result<(), GraphError>` | Processes texture uploads (full and partial) and deferred frees. Must be called **before** `Graph::begin_frame`. |
| `paint(&mut self, graph: &mut Graph, frame: &Frame, primitives: &[egui::ClippedPrimitive], pixels_per_point: f32) -> Result<(), GraphError>` | Tessellates primitives into vertices and indices, resizes buffers if needed, and records a render pass with scissor-clipped draw calls. Called between `begin_frame` and `end_frame`. |
| `destroy(self, graph: &mut Graph)` | Frees all GPU resources (managed textures, buffers, pipeline, sampler). User textures registered via `register_texture` are **not** destroyed — the caller owns them. |

---

## Call order

The per-frame sequence must follow this order:

1. `egui_state.take_egui_input(&window)` — collect platform input
2. `egui_ctx.run(input, |ctx| { ... })` — run UI logic, obtain `FullOutput`
3. `egui_ctx.tessellate(output.shapes, ppp)` — tessellate shapes into primitives
4. **`renderer.prepare(&mut graph, &output.textures_delta)`** — upload textures and process frees
5. `graph.begin_frame()` — acquire the swapchain image
6. Your own passes (e.g. a clear pass or scene rendering)
7. **`renderer.paint(&mut graph, &frame, &primitives, ppp)`** — record egui draw commands
8. `graph.end_frame(frame)` — submit

`prepare` must come before `begin_frame` because it issues transfer commands (image uploads) that need to complete before the frame's render passes reference those textures.

`paint` should generally be the last render pass so the UI is drawn on top of everything else.

---

## Resource management

### Textures

Textures are stored as persistent `R8G8B8A8_SRGB` images with `SAMPLED | TRANSFER_DST` usage. The renderer manages them automatically through egui's `TexturesDelta`:

- **Full uploads** create a new persistent image and upload the pixel data.
- **Partial updates** write a sub-region to an existing image via `upload_to_image`.
- **Deferred deletion:** texture IDs from `textures_delta.free` are queued during `prepare` and actually destroyed on the next `prepare` call. This avoids destroying an image that may still be in flight on the GPU.

### User textures

You can display your own vrlgraph images (render targets, loaded textures, etc.) inside egui widgets by registering them:

```rust,ignore
// Create an offscreen render target
let offscreen = graph.persistent_image("offscreen")
    .format(vk::Format::R8G8B8A8_SRGB)
    .extent(256, 256)
    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT)
    .build()?;

// Register it with the egui renderer
let texture_id = egui_renderer.register_texture(offscreen);

// Render to the offscreen image in your own pass
graph.render_pass("offscreen")
    .write(WithClearColor(offscreen, Access::ColorAttachment, [0.0, 0.0, 0.2, 1.0]))
    .execute(move |cmd, res| {
        cmd.bind_graphics_pipeline(res.pipeline(pipeline));
        cmd.set_viewport_scissor(offscreen_extent);
        cmd.draw(3, 1);
    });

// Display it in egui
ui.image(egui::load::SizedTexture::new(texture_id, [256.0, 256.0]));
```

The render graph handles layout transitions automatically (`COLOR_ATTACHMENT_OPTIMAL` → `SHADER_READ_ONLY_OPTIMAL`).

User textures are **not owned** by the renderer — `destroy()` and `unregister_texture()` will never free the underlying image. Use `update_texture()` if you need to swap the backing image (e.g. after a resize).

### Buffers

Vertex and index buffers use `CpuToGpu` mapped memory and are rewritten every frame.

| Buffer | Initial capacity | Growth strategy |
|---|---|---|
| Vertex | 64 KB | Next power of two when exceeded |
| Index | 64 KB | Next power of two when exceeded |

When data exceeds the current capacity, the old buffer is destroyed and a new one is allocated at the next power-of-two size.

### Cleanup

Call `destroy` before dropping the `Graph`. It frees all remaining textures (including any pending deletions), both buffers, the pipeline, and the sampler.

---

## Rendering details

### Shaders

Shaders are compiled from GLSL to SPIR-V at build time by `build.rs` using `glslc --target-env=vulkan1.2` and embedded in the binary via `include_bytes!`.

- **Vertex shader:** transforms egui screen-space positions to NDC: `ndc = 2.0 * pos / screen_size - 1.0`. Passes UV and color to the fragment stage.
- **Fragment shader:** samples the texture via bindless index (`GL_EXT_nonuniform_qualifier`), converts vertex color from sRGB to linear, and applies premultiplied alpha (`color.rgb *= color.a`).

### Push constants

| Field | Type | Description |
|---|---|---|
| `screen_size` | `[f32; 2]` | Logical screen dimensions (`extent / pixels_per_point`) |
| `texture_index` | `u32` | Bindless sampled image index (binding 0) |
| `sampler_index` | `u32` | Bindless sampler index (binding 2) |

### Blending

The egui pass uses premultiplied alpha blending:

| Parameter | Value |
|---|---|
| `src_color_blend_factor` | `ONE` |
| `dst_color_blend_factor` | `ONE_MINUS_SRC_ALPHA` |
| `color_blend_op` | `ADD` |
| `src_alpha_blend_factor` | `ONE_MINUS_DST_ALPHA` |
| `dst_alpha_blend_factor` | `ONE` |
| `alpha_blend_op` | `ADD` |

The pass writes to the backbuffer with `LoadOp::Load` to preserve whatever was rendered before it.

### Clipping

Each draw call sets a scissor rectangle derived from egui's `clip_rect`, scaled by `pixels_per_point` and clamped to the swapchain extent. Draw calls with zero-area scissor rects are skipped.

---

## Examples

### demo

Basic egui integration with text input, slider, and button:

```shell
cargo run -p vrlgraph-egui --example demo
```

### egui_texture

Renders a colored triangle to an offscreen image and displays it inside an egui panel using `register_texture`:

```shell
cargo run -p vrlgraph-egui --example egui_texture
```

---

## Build requirements

The `build.rs` script compiles the GLSL shaders (`egui.vert.glsl`, `egui.frag.glsl`) to SPIR-V. It searches for `glslc` in the following order:

1. System PATH
2. `$VULKAN_SDK/Bin/glslc` (or `glslc.exe` on Windows)

If `glslc` is not found, the build will fail because the embedded SPIR-V files will not be generated.
