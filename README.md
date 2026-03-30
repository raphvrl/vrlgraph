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

Shaders must be compiled to SPIR-V before being passed to the pipeline builders. vrlgraph loads them from the filesystem at the paths you provide.

---

## Quick start

```rust,ignore
use vrlgraph::prelude::*;

let mut graph = Graph::builder()
    .window(&window)
    .size(1280, 720)
    .build()?;

let pipeline = graph
    .graphics_pipeline()
    .vertex_shader("shaders/triangle.vert.spv")?
    .fragment_shader("shaders/triangle.frag.spv")?
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

### Render passes

`render_pass` is for fragment shader work. A pass that writes a color or depth attachment will have dynamic rendering (`VK_KHR_dynamic_rendering`) set up automatically for the images it writes.

```rust,ignore
graph.render_pass("shadow_map")
    .write((shadow_atlas, Access::DepthAttachment))
    .execute(move |cmd, res| {
        cmd.bind_graphics_pipeline(res.pipeline(shadow_pipeline));
        cmd.set_viewport_scissor(shadow_extent);
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
let shadow_atlas = graph.persistent_image()
    .format(vk::Format::D32_SFLOAT)
    .extent(4096, 4096)
    .label("shadow_atlas")
    .build()?;

// later, when no longer needed — also frees any bindless slots
graph.destroy_image(shadow_atlas);
```

### Resizable images

Resizable images are persistent images that are automatically recreated when the window is resized. Call `.resizable()` on a persistent image builder. The extent defaults to the swapchain size.

```rust,ignore
let hdr_buffer = graph.persistent_image()
    .format(vk::Format::R16G16B16A16_SFLOAT)
    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
    .label("hdr_buffer")
    .resizable()
    .build()?;
```

### Textures

`load_texture` decodes a PNG or JPEG file and uploads it to a GPU image. The image is persistent.

```rust,ignore
let albedo = graph.load_texture("assets/wood_albedo.png")?;
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
| `.label(impl Into<String>)` | no | `""` | Debug name |
| `.usage(vk::ImageUsageFlags)` | no | empty | Additional Vulkan usage flags |
| `.resizable()` | no | `false` | Auto-recreate on window resize (persistent only) |

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

### Convenience buffer methods

The typed methods handle usage flags and memory location automatically.

```rust,ignore
// Storage buffer (CpuToGpu) — SHADER_DEVICE_ADDRESS included automatically
let params = graph.storage_buffer("params", &data)?;

// Uniform buffer (CpuToGpu) — update each frame with write_buffer
let ubo = graph.uniform_buffer("scene_ubo", &[uniforms])?;
graph.write_buffer(ubo, &[new_uniforms]);

// Vertex / index buffers (GpuOnly) — data staged automatically via a one-shot transfer
let verts   = graph.vertex_buffer("mesh_verts", &vertices)?;
let indices = graph.index_buffer("mesh_indices", &idx_data)?;

// Vertex / index buffers (CpuToGpu) — no staging, directly writable for dynamic geometry
let chunk_verts   = graph.vertex_buffer_dynamic("chunk_verts", &vertices)?;
let chunk_indices = graph.index_buffer_dynamic("chunk_indices", &idx_data)?;
graph.write_buffer(chunk_verts, &new_vertices);

// Empty storage buffer (e.g. compute scratch space)
let scratch = graph.storage_buffer_empty("scratch", 1 << 20)?;
```

For cases that need custom usage flags or memory location, `create_buffer` and `upload_buffer` remain available.

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
    buf.write(std::slice::from_ref(&per_frame_data));
    // bind buf.raw as a uniform buffer
});
```

---

## Pipelines

### Graphics pipelines

Derive `VertexInput` on your vertex struct. The macro generates the binding and attribute descriptions automatically from the field types, using `offset_of!` for offsets and inferring the Vulkan format from the Rust type.

```rust,ignore
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, VertexInput)]
struct Vertex {
    pos:    [f32; 3],
    normal: [f32; 3],
    uv:     [f32; 2],
}

let pipeline = graph
    .graphics_pipeline()
    .vertex_shader("shaders/mesh.vert.spv")?
    .fragment_shader("shaders/pbr.frag.spv")?
    .color_formats(&[vk::Format::R16G16B16A16_SFLOAT])
    .depth_format(vk::Format::D32_SFLOAT)
    .vertex_input::<Vertex>()
    .build()?;
```

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
#[derive(Clone, Copy, Pod, Zeroable, VertexInput)]
struct Vertex {
    pos: glam::Vec3,                    // inferred if `glam` feature is enabled
    #[format(R16G16_SFLOAT)]
    uv: [u16; 2],                       // explicit override
}
```

Use `#[vertex_input(rate = instance)]` on the struct for per-instance data:

```rust,ignore
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, VertexInput)]
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
let pipeline = graph
    .compute_pipeline()
    .shader("shaders/tonemap.comp.spv")?
    .build()?;
```

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
let buf = graph.storage_buffer("my_data", &data)?;

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

The pipeline uses extended dynamic state. These values can change between draw calls without rebuilding the pipeline.

```rust,ignore
cmd.set_cull_mode(vk::CullModeFlags::BACK);
cmd.set_front_face(vk::FrontFace::COUNTER_CLOCKWISE);
cmd.set_depth_test_enable(true);
cmd.set_depth_write_enable(true);
cmd.set_depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
cmd.set_polygon_mode(vk::PolygonMode::FILL);
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

Pass any `Pod` value directly — no manual byte conversion needed:

```rust,ignore
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawPush {
    sampled_idx: u32,
    sampler_idx: u32,
    _pad: [u32; 2],
}

cmd.push_constants(&DrawPush { sampled_idx: res.sampled_index(my_image).0, sampler_idx, _pad: [0; 2] });
```

For dynamic payloads assembled at runtime (e.g. a `Vec<u8>` slice), use `push_constants_raw(&[u8])`.

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

// sampler.raw()  -> u32, pass to shaders via push constants (binding 2)
// sampler.handle -> for destroy_sampler
graph.destroy_sampler(sampler);
```

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
