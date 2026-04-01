# vrlgraph

A personal Vulkan render graph workspace for Rust, built to speed up future graphics projects without rewriting the same boilerplate every time. It handles pass ordering, image layout transitions, pipeline barriers, and swapchain management so the focus stays on what the shaders do rather than how to synchronize them.

vrlgraph is not a game engine, a scene graph, or a material system. It is a thin, explicit layer on top of raw Vulkan that automates the tedious parts: resource tracking, barrier insertion, and frame pacing.

---

## Crates

| Crate | Description |
|---|---|
| [`vrlgraph`](crates/vrlgraph) | Core render graph library — `Graph`, passes, resources, pipelines, bindless descriptors |
| [`vrlgraph-egui`](crates/vrlgraph-egui) | egui rendering backend built on top of vrlgraph |

---

## Prerequisites

- **Rust 1.85+** (edition 2024)
- **Vulkan SDK** (with `glslc` on PATH for `vrlgraph-egui`)

---

## Workspace layout

```
vrlgraph/          # core library
  derive/          # proc-macro crate (VertexInput, ShaderType)
vrlgraph-egui/     # egui backend
```

---

## Quick start

Add the core crate to your project:

```toml
[dependencies]
vrlgraph = { git = "https://github.com/raphvrl/vrlgraph" }
```

To also render an egui UI on top:

```toml
[dependencies]
vrlgraph      = { git = "https://github.com/raphvrl/vrlgraph" }
vrlgraph-egui = { git = "https://github.com/raphvrl/vrlgraph" }
egui          = "0.31"
```

See each crate's README for the full installation guide, usage, and API reference.

---

## Design principles

**Declare, don't manage.** You describe what each pass reads and writes. The graph computes the execution order, inserts the required `vkCmdPipelineBarrier2` calls, and transitions image layouts — you never touch synchronization primitives directly.

**Bindless by default.** A single global descriptor set (set 0, `UPDATE_AFTER_BIND`) holds all images and samplers. There are no per-pass descriptor sets or descriptor pool management.

**Push constants only.** All per-draw data — bindless indices, Buffer Device Address pointers, shader parameters — flows through a shared 256-byte push constant range. No uniform buffer juggling.

**Typed over raw.** Proc-macro derives (`VertexInput`, `ShaderType`) generate Vulkan binding descriptions and GPU-layout serialization from plain Rust structs. `ash` types remain accessible when you need them.
