# vrlgraph

A personal Vulkan render graph library for Rust, built to speed up future graphics projects without rewriting the same boilerplate every time. It handles pass ordering, image layout transitions, pipeline barriers, and swapchain management so the focus stays on what the shaders do rather than how to synchronize them.

vrlgraph is not a game engine, a scene graph, or a material system. It is a thin, explicit layer on top of raw Vulkan that automates the tedious parts: resource tracking, barrier insertion, and frame pacing.

---

## Prerequisites

- **Rust 1.85+** (edition 2024)
- **Vulkan SDK**

---

## Installation

```toml
[dependencies]
vrlgraph = { git = "https://github.com/raphvrl/vrlgraph" }
```

`ash` is re-exported from vrlgraph, so you do not need to declare it as a separate dependency.

### Optional features

| Feature | Description |
|---|---|
| `glam` | Adds `VertexAttribute` and `ShaderType` implementations for glam vector and matrix types |

To use the `glam` feature, enable it in your `Cargo.toml` and make sure glam is declared with the `bytemuck` feature, which is required for glam types to implement `Pod`:

```toml
[dependencies]
vrlgraph = { git = "https://github.com/raphvrl/vrlgraph", features = ["glam"] }
glam = { version = "0.32.1", features = ["bytemuck"] }
```

Shaders must be compiled to SPIR-V before being passed to the pipeline builders. vrlgraph loads them from the filesystem at the paths you provide.

---

## Quick start

```rust,ignore
use vrlgraph::prelude::*;

let mut graph = Graph::builder()
    .window(&window)
    .size(1280, 720)
    .build()?;

let vs = graph.shader_module("shaders/triangle.vert.spv", "main")?;
let fs = graph.shader_module("shaders/triangle.frag.spv", "main")?;

let pipeline = graph
    .graphics_pipeline("triangle")
    .vertex_shader(vs)
    .fragment_shader(fs)
    .build()?;

loop {
    let frame = match graph.begin_frame() {
        Ok(f) => f,
        Err(GraphError::SwapchainOutOfDate) => {
            let size = window.inner_size();
            graph.resize(size.width, size.height);
            continue;
        }
        Err(e) => return Err(e),
    };

    graph.render_pass("main")
        .write((frame.backbuffer, Access::ColorAttachment))
        .execute(move |cmd, res| {
            cmd.bind_graphics_pipeline(res.pipeline(pipeline));
            cmd.set_viewport_scissor(frame.extent);
            cmd.draw(3, 1);
        });

    graph.end_frame(frame)?;
}
```

---

## Overview

The central type is `Graph`. It owns the Vulkan device, the swapchain, all GPU resources, and the frame timeline. You interact with it by declaring passes, describing which images and buffers each pass reads and writes, and providing a closure that records GPU commands. The graph executes them in dependency order with the correct synchronization automatically inserted.

```rust,ignore
use vrlgraph::prelude::*;

let mut graph = Graph::builder()
    .window(&window)
    .size(1280, 720)
    .validation(true)
    .build()?;
```

---

## Frame loop

Each frame starts with `begin_frame` and ends with `end_frame`. Between those two calls you declare your passes. Nothing executes until `end_frame` is called, at which point the graph submits all recorded work in dependency order.

`begin_frame` returns a `Frame` that gives you the current backbuffer handle, the swapchain extent, the frame index, and a flag indicating whether the window was just resized.

```rust,ignore
let frame = graph.begin_frame()?;

// declare passes here

graph.end_frame(frame)?;
```

If the swapchain is out of date (e.g. the window was minimized and restored), `begin_frame` returns `GraphError::SwapchainOutOfDate`. The standard response is to call `graph.resize(width, height)` and skip the current frame.

```rust,ignore
match graph.begin_frame() {
    Ok(frame) => { /* record passes */ }
    Err(GraphError::SwapchainOutOfDate) => {
        let size = window.inner_size();
        graph.resize(size.width, size.height);
    }
    Err(e) => return Err(e),
}
```

### Frame fields

| Field | Type | Description |
|---|---|---|
| `backbuffer` | `Image` | The swapchain image for this frame |
| `extent` | `vk::Extent2D` | Current surface dimensions |
| `index` | `u32` | Swapchain image index |
| `resized` | `bool` | True on the first frame after a resize |

---

## Passes

A pass is a named unit of GPU work. You declare it with `render_pass` or `compute_pass`, describe its image and buffer accesses, and record its commands in a closure.

```rust,ignore
graph.render_pass("lighting")
    .read((gbuffer_color,   Access::ShaderRead))
    .read((gbuffer_normals, Access::ShaderRead))
    .write((hdr_output,     Access::ColorAttachment))
    .execute(|cmd, res| {
        // record commands
    });
```

The graph uses the declared accesses to determine pass order and insert the required pipeline barriers. You do not call `vkCmdPipelineBarrier` yourself.

### Pass builder methods

| Method | Description |
|---|---|
| `.read(impl ReadParam)` | Declare resource read access for this pass |
| `.write(impl WriteParam)` | Declare resource write access for this pass |
| `.multiview(u32)` | Enable multiview rendering with the given view mask |
| `.execute(FnOnce(&mut Cmd, &FrameResources))` | Provide the closure that records GPU commands |

### Render passes

`render_pass` is for fragment shader work. A pass that writes a color or depth attachment will have dynamic rendering (`VK_KHR_dynamic_rendering`) set up automatically for the images it writes.

```rust,ignore
graph.render_pass("shadow_map")
    .write((shadow_atlas, Access::DepthAttachment))
    .execute(move |cmd, res| {
        cmd.bind_graphics_pipeline(res.pipeline(shadow_pipeline));
        cmd.set_viewport_scissor(shadow_extent);
        cmd.set_depth_bias_enable(true);
        cmd.set_depth_bias(1.25, 0.0, 1.75);
        cmd.bind_vertex_buffer(res.buffer(vertex_buffer), 0);
        cmd.bind_index_buffer(res.buffer(index_buffer), 0);
        cmd.draw_indexed(index_count, 1, 0, 0);
    });
```

### Compute passes

`compute_pass` is for compute shader work. Dynamic rendering is not started for compute passes.

```rust,ignore
graph.compute_pass("blur")
    .read((hdr_output,   Access::ComputeRead))
    .write((blur_result, Access::ComputeWrite))
    .execute(move |cmd, res| {
        cmd.bind_compute_pipeline(res.pipeline(blur_pipeline));
        cmd.dispatch(width.div_ceil(8), height.div_ceil(8), 1);
    });
```

### Load operations

By default, attachments written with `Access::ColorAttachment` or `Access::DepthAttachment` are cleared at the start of the pass. You can override this with `LoadOp`.

```rust,ignore
// Clear the attachment (default)
.write((target, Access::ColorAttachment))

// Preserve existing contents, e.g. for accumulation passes
.write(WithLoadOp(target, Access::ColorAttachment, LoadOp::Load))

// Discard — fastest option when you will write every pixel
.write(WithLoadOp(target, Access::ColorAttachment, LoadOp::DontCare))
```

### Array image layers

To write a single layer of an array image, use `WithLayer` or `WithLayerLoadOp`.

```rust,ignore
// Write layer 2 of a cubemap face
.write(WithLayer(cubemap, Access::ColorAttachment, 2))

// Write layer 2 with an explicit load op
.write(WithLayerLoadOp(cubemap, Access::ColorAttachment, LoadOp::Load, 2))
```

### Multiview

For multiview rendering (e.g. VR), call `.multiview(view_mask)` before `.execute`.

```rust,ignore
graph.render_pass("stereo_geometry")
    .write((stereo_target, Access::ColorAttachment))
    .multiview(0b11) // views 0 and 1
    .execute(move |cmd, res| { /* ... */ });
```

---

## Access types

`Access` describes how a pass uses an image. The graph translates each variant to the correct `VkImageLayout`, `VkPipelineStageFlags2`, and `VkAccessFlags2`.

| Variant | Typical use |
|---|---|
| `ColorAttachment` | Writing a color render target |
| `DepthAttachment` | Writing a depth/stencil buffer |
| `DepthStencilAttachment` | Writing depth and stencil together |
| `DepthRead` | Depth buffer read in a depth test without writes |
| `ShaderRead` | Sampling in a fragment or vertex shader |
| `ComputeRead` | Reading from a compute shader |
| `ComputeWrite` | Writing from a compute shader |
| `TransferSrc` | Source of a copy or blit operation |
| `TransferDst` | Destination of a copy or blit operation |

`BufferUsage` serves the same purpose for buffers.

| Variant | Typical use |
|---|---|
| `UniformRead` | UBO read in any shader stage |
| `StorageRead` | SSBO read in a compute shader |
| `StorageWrite` | SSBO write in a compute shader |
| `VertexRead` | Vertex buffer |
| `IndexRead` | Index buffer |
| `IndirectRead` | Indirect draw/dispatch arguments |
| `TransferSrc` / `TransferDst` | Copy operations |

---

## Images

### Transient images

Transient images live for a single frame. The graph allocates and destroys them automatically. Use them for intermediate results that are not needed across frames. When no extent is specified, the swapchain extent is used.

```rust,ignore
let gbuffer_albedo = graph.transient_image()
    .format(vk::Format::R8G8B8A8_UNORM)
    .label("gbuffer_albedo")
    .build()?;
```

### Persistent images

Persistent images survive across frames and must be destroyed manually. Use them for render targets you allocate once at startup (shadow maps, lookup tables, etc.).

```rust,ignore
let shadow_atlas = graph.persistent_image("shadow_atlas")
    .format(vk::Format::D32_SFLOAT)
    .extent(4096, 4096)
    .build()?;

// later, when no longer needed — also frees any bindless slots
graph.destroy_image(shadow_atlas);
```

### Resizable images

Resizable images are persistent images that are automatically recreated when the window is resized. Call `.resizable()` on a persistent image builder. The extent defaults to the swapchain size.

```rust,ignore
let hdr_buffer = graph.persistent_image("hdr_buffer")
    .format(vk::Format::R16G16B16A16_SFLOAT)
    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
    .resizable()
    .build()?;
```

### Textures

`load_texture` creates a persistent GPU image from raw pixel data.
The caller is responsible for image decoding (e.g. via the `image` or `stb_image` crate).

```rust,ignore
// Automatic mipmap generation (blit-based)
let albedo = graph.load_texture("wood_albedo")
    .pixels(&rgba_pixels)
    .extent(width, height)
    .format(vk::Format::R8G8B8A8_SRGB)
    .build()?;

// Pre-computed mipmaps (required for block-compressed formats like BC7)
let albedo_bc7 = graph.load_texture("wood_albedo_bc7")
    .mip_data(&[&mip0, &mip1, &mip2])
    .extent(width, height)
    .format(vk::Format::BC7_SRGB_BLOCK)
    .build()?;
```

### Image builder methods

| Method | Required | Default | Description |
|---|---|---|---|
| `.format(vk::Format)` | yes | — | Pixel format |
| `.extent(w, h)` | no | swapchain extent | Width and height |
| `.extent_3d(w, h, d)` | no | swapchain extent, depth 1 | Width, height, and depth |
| `.mip_levels(u32)` | no | `1` | Number of mip levels |
| `.samples(SampleCount)` | no | `S1` | MSAA sample count |
| `.array_2d(layers)` | no | `Image2D` | 2D array image |
| `.cubemap()` | no | `Image2D` | Cubemap image |
| `.cubemap_array(count)` | no | `Image2D` | Cubemap array image |
| `.usage(vk::ImageUsageFlags)` | no | empty | Additional Vulkan usage flags |
| `.resizable()` | no | `false` | Auto-recreate on window resize (persistent only) |

The label is provided as the first argument to `persistent_image(label)` and `load_texture(label)`.

### Texture builder methods

| Method | Required | Default | Description |
|---|---|---|---|
| `.pixels(&[u8])` | yes* | — | Raw pixel data (mip 0); mipmaps generated via blit |
| `.mip_data(&[&[u8]])` | yes* | — | Pre-computed mip levels (one slice per level) |
| `.extent(w, h)` | yes | — | Width and height |
| `.format(vk::Format)` | yes | — | Pixel format |
| `.mip_levels(u32)` | no | auto | Number of mip levels (0 = auto from extent; ignored with `mip_data`) |

\* Exactly one of `.pixels()` or `.mip_data()` is required. Use `.mip_data()` for block-compressed formats (BC7, BC1, etc.) where GPU blit generation is not supported.

**Important with bindless:** set `SAMPLED` and/or `STORAGE` explicitly in `.usage()` if you need to access the image by bindless index. The graph infers other usage flags (attachment, transfer) from pass accesses, but `SAMPLED`/`STORAGE` must be declared upfront so the bindless slot is allocated at creation time. Transient images are an exception — their usage is inferred from passes before slot allocation.

### ImageKind

```rust,ignore
ImageKind::Image2D                        // standard 2D texture
ImageKind::Image2DArray { layers: 6 }     // array of 2D textures
ImageKind::Cubemap                        // 6-face cubemap
ImageKind::CubemapArray { count: 4 }      // array of cubemaps
```

---

## Buffers

### Buffer builders

Buffers are created through a builder pattern. They are empty by default — call `.data()` to provide initial contents.

```rust,ignore
// Storage buffer (CpuToGpu) — SHADER_DEVICE_ADDRESS included automatically
let params = graph.storage_buffer("params").data(&data).build()?;

// Uniform buffer (CpuToGpu) — update each frame with write_buffer
let ubo = graph.uniform_buffer("scene_ubo").data(&uniforms).build()?;
graph.write_buffer(ubo, &new_uniforms);

// Vertex / index buffers (GpuOnly) — data uploaded via the transfer queue
let verts   = graph.vertex_buffer("mesh_verts").data(&vertices).build()?;
let indices = graph.index_buffer("mesh_indices").data(&idx_data).build()?;

// Async vertex buffer — uploaded in the background, does not block
let chunk_verts = graph.vertex_buffer("chunk").data(&vertices).build_async()?;

// Vertex / index buffers (CpuToGpu) — no staging, directly writable for dynamic geometry
let dyn_verts = graph.vertex_buffer("dyn_verts").data(&vertices).dynamic().build()?;
graph.write_buffer_slice(dyn_verts, &new_vertices);

// Empty storage buffer (e.g. compute scratch space)
let scratch = graph.storage_buffer("scratch").size(1 << 20).build()?;

// Streaming buffer (one slot per frame in flight, auto-rotated)
let stream = graph.storage_buffer("per_frame").size(256).streaming()?;
```

### Host buffer builder methods (storage_buffer / uniform_buffer)

| Method | Required | Default | Description |
|---|---|---|---|
| `.size(vk::DeviceSize)` | yes* | — | Buffer size in bytes |
| `.data<T: ShaderType>(&T)` | yes* | — | Initial contents with shader-type padding |
| `.build()` | — | — | Create the buffer |
| `.streaming()` | — | — | Create a streaming buffer (one slot per frame in flight) |

\* Exactly one of `.size()` or `.data()` is required.

### GPU buffer builder methods (vertex_buffer / index_buffer)

| Method | Required | Default | Description |
|---|---|---|---|
| `.size(vk::DeviceSize)` | yes* | — | Buffer size in bytes |
| `.data<T: Pod>(&[T])` | yes* | — | Initial contents (uploaded via transfer queue) |
| `.dynamic()` | no | `false` | Use CpuToGpu memory instead of GpuOnly (no staging, directly writable) |
| `.build()` | — | — | Create the buffer (blocks until upload completes), returns `Buffer` |
| `.build_async()` | — | — | Create the buffer without blocking, returns `AsyncBuffer` |

\* Exactly one of `.size()` or `.data()` is required.

For cases that need custom usage flags or memory location, `create_buffer` remains available.

```rust,ignore
let buf = graph.create_buffer(&BufferDesc {
    size: 1024,
    usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
    location: gpu_allocator::MemoryLocation::GpuOnly,
    label: "indirect_args".into(),
})?;
```

### Streaming buffers

Streaming buffers maintain one slot per frame in flight, so you can write to the current frame's slot from the CPU while the GPU reads from the previous frame's slot — no explicit synchronization needed.

```rust,ignore
let per_frame_buf = graph.create_streaming_buffer(
    std::mem::size_of::<PerFrameData>() as u64,
    vk::BufferUsageFlags::UNIFORM_BUFFER,
    gpu_allocator::MemoryLocation::CpuToGpu,
    "per_frame_data",
)?;
```

Inside the frame loop, access the current slot through `FrameResources`:

```rust,ignore
.execute(move |cmd, res| {
    let buf = res.streaming_buffer(per_frame_buf);
    buf.write_slice(std::slice::from_ref(&per_frame_data));
    // bind buf.raw as a uniform buffer
});
```

### Async buffers

`build_async()` uploads buffer data via the transfer queue without blocking the calling thread. It returns an `AsyncBuffer` instead of a `Buffer` — the two are distinct types, enforced at compile time.

A dedicated transfer queue (DMA engine) is used when the GPU exposes one (common on AMD); otherwise the graphics queue is used transparently. The graph discovers the transfer queue automatically at initialization.

```rust,ignore
// Upload a chunk mesh in the background
let chunk_vb = graph.vertex_buffer("chunk_42").data(&vertices).build_async()?;
let chunk_ib = graph.index_buffer("chunk_42_idx").data(&indices).build_async()?;
```

Inside a pass, use `try_buffer()` to access an `AsyncBuffer`. It returns `None` while the transfer is still in progress:

```rust,ignore
graph.render_pass("draw_chunks")
    .read((chunk_vb, BufferUsage::VertexRead))
    .read((chunk_ib, BufferUsage::IndexRead))
    .execute(move |cmd, res| {
        let Some(vb) = res.try_buffer(chunk_vb) else { return };
        let Some(ib) = res.try_buffer(chunk_ib) else { return };
        cmd.bind_vertex_buffer(vb, 0);
        cmd.bind_index_buffer(ib, 0);
        cmd.draw_indexed(index_count, 1, 0, 0, 0);
    });
```

`Buffer` and `AsyncBuffer` cannot be mixed: `res.buffer()` only accepts `Buffer`, and `res.try_buffer()` only accepts `AsyncBuffer`. This prevents accidentally reading an in-flight buffer.

To destroy an async buffer, use `destroy_async_buffer()` which waits for any pending transfer to complete before freeing the resource.

---

## Pipelines

### Graphics pipelines

Derive `VertexInput` on your vertex struct. The macro generates the binding and attribute descriptions automatically from the field types, using `offset_of!` for offsets and inferring the Vulkan format from the Rust type.

```rust,ignore
#[repr(C)]
#[derive(VertexInput)]
struct Vertex {
    pos:    [f32; 3],
    normal: [f32; 3],
    uv:     [f32; 2],
}

let vs = graph.shader_module("shaders/mesh.vert.spv", "main")?;
let fs = graph.shader_module("shaders/pbr.frag.spv", "main")?;

let pipeline = graph
    .graphics_pipeline("mesh")
    .vertex_shader(vs)
    .fragment_shader(fs)
    .color_formats(&[vk::Format::R16G16B16A16_SFLOAT])
    .depth_format(vk::Format::D32_SFLOAT)
    .vertex_input::<Vertex>()
    .build()?;
```

### Graphics pipeline builder methods

| Method | Required | Default | Description |
|---|---|---|---|
| `.vertex_shader(ShaderModule)` | yes | — | Vertex shader module |
| `.fragment_shader(ShaderModule)` | yes | — | Fragment shader module |
| `.color_formats(&[vk::Format])` | no | inferred from pass | Color attachment formats |
| `.depth_format(vk::Format)` | no | — | Depth attachment format |
| `.vertex_input::<V: VertexInput>()` | no | — | Vertex layout from derive macro |
| `.vertex_input_raw(bindings, attributes)` | no | — | Raw Vulkan vertex input descriptors |
| `.view_mask(u32)` | no | — | View mask for multiview rendering |

Skip `vertex_input` entirely for shader-only draws (fullscreen triangles, compute-driven geometry, etc.).

#### Format inference

| Rust type | Vulkan format |
|---|---|
| `f32` | `R32_SFLOAT` |
| `[f32; 2]` | `R32G32_SFLOAT` |
| `[f32; 3]` | `R32G32B32_SFLOAT` |
| `[f32; 4]` | `R32G32B32A32_SFLOAT` |
| `u32` / `[u32; N]` | `R32_UINT` / `R32G32[B32[A32]]_UINT` |
| `i32` / `[i32; N]` | `R32_SINT` / `R32G32[B32[A32]]_SINT` |
| `[u8; 4]` | `R8G8B8A8_UNORM` |

Enable the `glam` feature to add automatic format inference for `glam::Vec2/3/4`, `UVec2/3/4`, `IVec2/3/4`.

#### Overrides

Use `#[format(FORMAT)]` on a field when the type cannot be inferred automatically:

```rust,ignore
#[repr(C)]
#[derive(VertexInput)]
struct Vertex {
    pos: glam::Vec3,                    // inferred if `glam` feature is enabled
    #[format(R16G16_SFLOAT)]
    uv: [u16; 2],                       // explicit override
}
```

Use `#[vertex_input(rate = instance)]` on the struct for per-instance data:

```rust,ignore
#[repr(C)]
#[derive(VertexInput)]
#[vertex_input(rate = instance)]
struct InstanceData {
    model_col0: [f32; 4],
    model_col1: [f32; 4],
    model_col2: [f32; 4],
    model_col3: [f32; 4],
}
```

For cases that cannot be expressed as a `VertexInput` impl, `vertex_input_raw(bindings, attributes)` accepts Vulkan descriptors directly.

You do not need to set `color_formats` or `depth_format` if your pass writes attachments — the graph infers them from the declared accesses. Set them explicitly only when the format cannot be inferred from context.

All pipelines share the single global pipeline layout (set 0 = bindless table, 256-byte push constant range). There is no per-pipeline layout to configure.

### Compute pipelines

```rust,ignore
let cs = graph.shader_module("shaders/tonemap.comp.spv", "main")?;

let pipeline = graph
    .compute_pipeline("tonemap")
    .shader(cs)
    .build()?;
```

### Compute pipeline builder methods

| Method | Required | Description |
|---|---|---|
| `.shader(ShaderModule)` | yes | Compute shader module |

### Pipeline caching

Pass a path to `pipeline_cache_path` on the builder to persist the Vulkan pipeline cache to disk. This reduces compilation time on subsequent runs.

```rust,ignore
let graph = Graph::builder()
    .window(&window)
    .size(1280, 720)
    .pipeline_cache_path("pipeline_cache.bin")
    .build()?;
```

---

## Bindless resources

vrlgraph uses a single global bindless descriptor set (set 0, `UPDATE_AFTER_BIND`) that holds all images and samplers for the entire application. There are no per-pass descriptor sets or descriptor pools to manage.

### Layout

| Binding | Type | Capacity | Accessor |
|---|---|---|---|
| 0 | `texture2D textures[]` | 4096 | `res.sampled_index(img)` → `BindlessIndex<Sampled>` |
| 1 | `image2D storage_images[]` | 1024 | `res.storage_index(img)` → `BindlessIndex<Storage>` |
| 2 | `sampler samplers[]` | 32 | `sampler.index` |
| 3 | `textureCube cube_textures[]` | 128 | `res.cubemap_index(img)` → `BindlessIndex<Cubemap>` |
| 4 | `texture2DArray array_textures[]` | 256 | `res.array_index(img)` → `BindlessIndex<Array2D>` |

### Automatic registration

Images are routed to the correct binding automatically based on `ImageKind` and `SAMPLED` usage:

| ImageKind | SAMPLED binding |
|---|---|
| `Image2D` (default) | 0 — `res.sampled_index()` |
| `Cubemap` / `CubemapArray` | 3 — `res.cubemap_index()` |
| `Image2DArray` | 4 — `res.array_index()` |

`STORAGE` images always go to binding 1 regardless of kind. On resize, all bindless slots are updated automatically.

```rust,ignore
.execute(move |cmd, res| {
    let idx: BindlessIndex<Sampled> = res.sampled_index(tex2d);   // binding 0
    let idx: BindlessIndex<Storage> = res.storage_index(target);  // binding 1
    let idx: BindlessIndex<Cubemap> = res.cubemap_index(skybox);  // binding 3
    let idx: BindlessIndex<Array2D> = res.array_index(atlas);     // binding 4
});
```

### Shaders

```glsl
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform texture2D          textures[];
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D storage_images[];
layout(set = 0, binding = 2) uniform sampler            samplers[];
layout(set = 0, binding = 3) uniform textureCube        cube_textures[];
layout(set = 0, binding = 4) uniform texture2DArray     array_textures[];

layout(push_constant) uniform PC {
    uint tex_idx;
    uint cube_idx;
    uint arr_idx;
    uint sampler_idx;
} pc;

void main() {
    vec4 c    = texture(sampler2D(textures[pc.tex_idx], samplers[pc.sampler_idx]), uv);
    vec4 cube = texture(samplerCube(cube_textures[pc.cube_idx], samplers[pc.sampler_idx]), dir);
    vec4 arr  = texture(sampler2DArray(array_textures[pc.arr_idx], samplers[pc.sampler_idx]), vec3(uv, layer));
}
```

### Buffers

Structured buffers are accessed via Buffer Device Address (BDA). All buffers carry a device address — retrieve it with `buffer_device_address` and pass it as a `uint64_t` in the push constants.

```rust,ignore
let buf = graph.storage_buffer("my_data").data(&data).build()?;

let addr = graph.buffer_device_address(buf);
```

```glsl
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, std430) readonly buffer MyData { vec4 items[]; };

layout(push_constant) uniform PC { uint64_t data_addr; } pc;

void main() {
    MyData data = MyData(pc.data_addr);
    vec4 item = data.items[gl_GlobalInvocationID.x];
}
```

---

## Commands

The `Cmd` type is the command recorder passed to every pass closure. It wraps the underlying `VkCommandBuffer` and exposes a typed API.

### Pipelines and state

```rust,ignore
cmd.bind_graphics_pipeline(res.pipeline(pipeline));
cmd.bind_compute_pipeline(res.pipeline(compute_pipeline));

cmd.set_viewport_scissor(frame.extent);
cmd.set_viewport(vk::Viewport { x: 0.0, y: 0.0, width: 1920.0, height: 1080.0, min_depth: 0.0, max_depth: 1.0 });
cmd.set_scissor(vk::Rect2D { offset: vk::Offset2D::default(), extent: frame.extent });
```

### Dynamic rasterizer state

The pipeline uses extended dynamic state. Values persist across pipeline binds (OpenGL-like model) and are reset to defaults once at the beginning of each pass via `reset_dynamic_state`. Defaults: no culling, no depth test/write, counter-clockwise winding, triangle list topology, fill mode, blending disabled, depth bias off.

```rust,ignore
cmd.set_cull_mode(vk::CullModeFlags::BACK);
cmd.set_front_face(vk::FrontFace::COUNTER_CLOCKWISE);
cmd.set_depth_test_enable(true);
cmd.set_depth_write_enable(true);
cmd.set_depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
cmd.set_polygon_mode(vk::PolygonMode::FILL);

cmd.set_depth_bias_enable(true);
cmd.set_depth_bias(1.25, 0.0, 1.75); // constant_factor, clamp, slope_factor
```

To set color blending for all attachments at once with additive defaults:

```rust,ignore
cmd.set_default_blend_state(attachment_count);
```

### Vertex and index buffers

Pass the `&GpuBuffer` reference from `FrameResources` directly — no `.raw` unwrapping needed.

```rust,ignore
cmd.bind_vertex_buffer(res.buffer(vertex_buf), 0);
cmd.bind_index_buffer(res.buffer(index_buf), 0);
```

### Push constants

Push constants are the sole mechanism to pass bindless indices, BDA pointers, and per-draw parameters to shaders. The shared pipeline layout exposes a single 256-byte range covering all stages.

Pass any `ShaderType` value directly — scalar-layout padding is applied automatically:

```rust,ignore
cmd.push_constants(&my_value);
```

For dynamic payloads assembled at runtime (e.g. a `Vec<u8>` slice), use `push_constants_raw(&[u8])`.

### Shader types

`#[derive(ShaderType)]` generates `Clone`, `Copy`, and a `write_padded` method that serializes the struct to GPU-compatible bytes with the correct padding for std140 (default) or std430. The struct itself is not modified — padding is applied at serialization time.

```rust,ignore
// std140 (default — suitable for uniform buffers)
#[derive(ShaderType)]
struct Camera {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    position: [f32; 3],
}

let cam = Camera { view, proj, position: [0.0, 1.0, 0.0] };
cmd.push_constants(&cam);

// std430 (suitable for storage buffers)
#[derive(ShaderType)]
#[shader_type(std430)]
struct Particle {
    position: [f32; 3],
    velocity: [f32; 3],
}
```

Dedicated API methods handle serialization transparently:

```rust,ignore
graph.uniform_buffer("camera").data(&cam).build()?; // allocate + write padded
graph.write_buffer(buf, &cam);               // update existing buffer
cmd.push_constants(&cam);                    // push constants with padding
```

**Supported types:** `f32`, `u32`, `i32`, `u64`, `[f32; 2..4]`, `[u32; 2..4]`, `[i32; 2..4]`, `[[f32; 4]; 4]` (mat4), `[[f32; 4]; 3]` (mat3). With the `glam` feature: `Vec2`, `Vec3`, `Vec3A`, `Vec4`, `UVec2`–`UVec4`, `IVec2`–`IVec4`, `Mat3`, `Mat4`.

For unsupported types, use the `#[align(N)]` attribute on the field to specify alignment manually.

### Draw and dispatch commands

```rust,ignore
cmd.draw(vertex_count, instance_count);
cmd.draw_indexed(index_count, instance_count, first_index, vertex_offset);
cmd.draw_indirect(res.buffer(indirect_buf), 0, draw_count, stride);
cmd.draw_indexed_indirect(res.buffer(indirect_buf), 0, draw_count, stride);

cmd.dispatch(groups_x, groups_y, groups_z);
cmd.dispatch_indirect(res.buffer(indirect_buf), 0);
```

### Debug markers

Debug markers appear in tools like RenderDoc and Nsight. They have no runtime cost in release builds when the validation layer is disabled.

```rust,ignore
cmd.begin_debug_group("shadow pass", [1.0, 0.5, 0.0, 1.0]);
// ... draw calls
cmd.end_debug_group();

cmd.insert_debug_label("barrier point", [0.0, 1.0, 0.0, 1.0]);
```

---

## Samplers

Samplers use a builder pattern matching the rest of the API. `create_sampler` returns a `SamplerBuilder` — configure it with method chaining and call `.build()` to get the `Sampler`. The sampler bundles the handle (for `destroy_sampler`) with the bindless index to pass to shaders via push constants.

```rust,ignore
let sampler = graph.create_sampler()
    .filter(Filter::LINEAR)
    .mipmap_mode(MipmapMode::LINEAR)
    .address_mode(AddressMode::REPEAT)
    .build()?;

// sampler.raw() -> u32, pass to shaders via push constants (binding 2)
graph.destroy_sampler(sampler);
```

### Sampler builder methods

| Method | Default | Description |
|---|---|---|
| `.mag_filter(Filter)` | `LINEAR` | Magnification filter |
| `.min_filter(Filter)` | `LINEAR` | Minification filter |
| `.filter(Filter)` | — | Set both mag and min filters |
| `.mipmap_mode(MipmapMode)` | `LINEAR` | Mipmap filtering mode |
| `.address_mode_u(AddressMode)` | — | U coordinate wrapping |
| `.address_mode_v(AddressMode)` | — | V coordinate wrapping |
| `.address_mode_w(AddressMode)` | — | W coordinate wrapping |
| `.address_mode(AddressMode)` | — | Set all three address modes |
| `.anisotropy(f32)` | disabled | Anisotropic filtering max ratio |
| `.compare_op(CompareOp)` | disabled | Comparison function for shadow maps |
| `.lod(min, max)` | — | LOD clamp range |
| `.mip_lod_bias(f32)` | `0.0` | Mipmap LOD bias |
| `.border_color(BorderColor)` | — | Border color for `CLAMP_TO_BORDER` |
| `.unnormalized_coordinates()` | `false` | Use pixel coordinates instead of \[0,1\] |

---

## Pass timing

The graph inserts GPU timestamp queries around each pass. After `end_frame` returns, `pass_timings` gives you the GPU execution time of every pass in the previous frame.

```rust,ignore
graph.end_frame(frame)?;

for timing in graph.pass_timings() {
    println!("{}: {:.2} us", timing.name, timing.gpu_ns as f64 / 1000.0);
}
```

`PassTiming` fields:

| Field | Type | Description |
|---|---|---|
| `name` | `&'static str` | Pass name as given to `render_pass` / `compute_pass` |
| `gpu_ns` | `u64` | GPU execution time in nanoseconds |

---

## Initialization options

`GraphBuilder` accepts the following options before calling `build`.

| Method | Required | Default | Description |
|---|---|---|---|
| `.window(&impl HasWindowHandle + HasDisplayHandle)` | yes | — | Window and display handles |
| `.size(width, height)` | yes | — | Initial swapchain dimensions |
| `.validation(bool)` | no | `false` | Enable Vulkan validation layers |
| `.present_mode(PresentMode)` | no | `Fifo` | Presentation mode (see table below) |
| `.gpu(GpuPreference)` | no | `HighPerformance` | GPU selection hint (see table below) |
| `.frames_in_flight(usize)` | no | `2` | Pipeline depth |
| `.pipeline_cache_path(impl Into<PathBuf>)` | no | none | Persist pipeline cache to disk |
| `.srgb(bool)` | no | `true` | Request sRGB swapchain format |

```rust,ignore
let graph = Graph::builder()
    .window(&window)                        // required: window handle
    .size(1280, 720)                        // required: initial surface size
    .validation(cfg!(debug_assertions))     // Vulkan validation layers
    .present_mode(PresentMode::Mailbox)     // presentation mode
    .gpu(GpuPreference::HighPerformance)    // GPU selection hint
    .frames_in_flight(2)                    // pipeline depth (default: 2)
    .pipeline_cache_path("cache.bin")       // persist pipeline cache
    .build()?;
```

### PresentMode

| Variant | Behaviour |
|---|---|
| `Fifo` | V-sync. Guaranteed to be available. |
| `Mailbox` | Submit as fast as possible, display latest frame. No tearing. |
| `Immediate` | No synchronization. May tear. Lowest latency. |

If the requested present mode is not supported by the hardware, the graph falls back to `Fifo`.

### GpuPreference

| Variant | Behaviour |
|---|---|
| `HighPerformance` | Prefer discrete GPU (default) |
| `LowPower` | Prefer integrated GPU |

---

## Window resize

Call `graph.resize(width, height)` when the window size changes. The graph recreates the swapchain on the next frame. Resizable images are recreated automatically and their bindless indices are updated in the global table — no manual intervention required.

---

## Hot shader reload

In debug builds, `reload_shaders` recompiles all pipelines from their source SPIR-V files on disk. Useful when combined with a file watcher to iterate on shaders without restarting the application.

```rust,ignore
#[cfg(debug_assertions)]
graph.reload_shaders()?;
```

---

## Error handling

All fallible operations return `Result<T, GraphError>`. The main variants you should handle at runtime are:

| Variant | When it occurs |
|---|---|
| `GraphError::SwapchainOutOfDate` | The surface was resized or invalidated |
| `GraphError::ShaderLoad(msg)` | SPIR-V file not found or invalid |
| `GraphError::ImageLoad(msg)` | Texture file not found or unsupported format |
| `GraphError::PassCycle(name)` | A cycle was detected in the pass dependency graph |

All other variants wrap lower-level errors (`DeviceError`, `ResourceError`, Vulkan result codes) and are generally fatal.
