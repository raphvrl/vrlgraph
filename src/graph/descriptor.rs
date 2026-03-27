use ash::vk;

use crate::resource::{BufferHandle, ResourcePool, SamplerHandle};

use super::image::{GraphImage, ImageEntry};
use super::{Graph, GraphError};

#[derive(Clone)]
enum DescriptorKind {
    StorageImage {
        image: GraphImage,
    },
    SampledImage {
        image: GraphImage,
    },
    CombinedSampler {
        sampler: SamplerHandle,
        image: GraphImage,
    },
    UniformBuffer {
        buffer: BufferHandle,
    },
    StorageBuffer {
        buffer: BufferHandle,
    },
}

impl DescriptorKind {
    fn descriptor_type(&self) -> vk::DescriptorType {
        match self {
            Self::StorageImage { .. } => vk::DescriptorType::STORAGE_IMAGE,
            Self::SampledImage { .. } => vk::DescriptorType::SAMPLED_IMAGE,
            Self::CombinedSampler { .. } => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            Self::UniformBuffer { .. } => vk::DescriptorType::UNIFORM_BUFFER,
            Self::StorageBuffer { .. } => vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

#[derive(Clone)]
pub struct DescriptorWrite {
    stage: vk::ShaderStageFlags,
    kind: DescriptorKind,
}

impl DescriptorWrite {
    pub fn storage_image(stage: vk::ShaderStageFlags, image: GraphImage) -> Self {
        Self {
            stage,
            kind: DescriptorKind::StorageImage { image },
        }
    }

    pub fn sampled_image(stage: vk::ShaderStageFlags, image: GraphImage) -> Self {
        Self {
            stage,
            kind: DescriptorKind::SampledImage { image },
        }
    }

    pub fn combined_image_sampler(
        stage: vk::ShaderStageFlags,
        sampler: SamplerHandle,
        image: GraphImage,
    ) -> Self {
        Self {
            stage,
            kind: DescriptorKind::CombinedSampler { sampler, image },
        }
    }

    pub fn uniform_buffer(stage: vk::ShaderStageFlags, buffer: BufferHandle) -> Self {
        Self {
            stage,
            kind: DescriptorKind::UniformBuffer { buffer },
        }
    }

    pub fn storage_buffer(stage: vk::ShaderStageFlags, buffer: BufferHandle) -> Self {
        Self {
            stage,
            kind: DescriptorKind::StorageBuffer { buffer },
        }
    }
}

pub struct PushDescriptor {
    kind: DescriptorKind,
}

impl PushDescriptor {
    pub fn storage_image(image: GraphImage) -> Self {
        Self {
            kind: DescriptorKind::StorageImage { image },
        }
    }

    pub fn sampled_image(image: GraphImage) -> Self {
        Self {
            kind: DescriptorKind::SampledImage { image },
        }
    }

    pub fn combined_image_sampler(sampler: SamplerHandle, image: GraphImage) -> Self {
        Self {
            kind: DescriptorKind::CombinedSampler { sampler, image },
        }
    }

    pub fn uniform_buffer(buffer: BufferHandle) -> Self {
        Self {
            kind: DescriptorKind::UniformBuffer { buffer },
        }
    }

    pub fn storage_buffer(buffer: BufferHandle) -> Self {
        Self {
            kind: DescriptorKind::StorageBuffer { buffer },
        }
    }
}

pub(crate) struct OwnedDescriptorResources {
    pub pool: vk::DescriptorPool,
    pub layout: vk::DescriptorSetLayout,
}

/// Builder for a descriptor set.
///
/// Obtained from [`Graph::descriptor_set`]. Add bindings in order, then call
/// [`build`](DescriptorSetBuilder::build) for a static set or
/// [`build_dynamic`](DescriptorSetBuilder::build_dynamic) for a set that can
/// be updated after a window resize.
///
/// Bindings are assigned sequentially starting at 0 in the order they are added.
pub struct DescriptorSetBuilder<'g> {
    graph: &'g mut Graph,
    writes: Vec<DescriptorWrite>,
}

impl<'g> DescriptorSetBuilder<'g> {
    pub(super) fn new(graph: &'g mut Graph) -> Self {
        Self {
            graph,
            writes: Vec::new(),
        }
    }

    pub fn storage_image(mut self, stage: vk::ShaderStageFlags, image: GraphImage) -> Self {
        self.writes
            .push(DescriptorWrite::storage_image(stage, image));
        self
    }

    pub fn sampled_image(mut self, stage: vk::ShaderStageFlags, image: GraphImage) -> Self {
        self.writes
            .push(DescriptorWrite::sampled_image(stage, image));
        self
    }

    pub fn combined_image_sampler(
        mut self,
        stage: vk::ShaderStageFlags,
        sampler: SamplerHandle,
        image: GraphImage,
    ) -> Self {
        self.writes.push(DescriptorWrite::combined_image_sampler(
            stage, sampler, image,
        ));
        self
    }

    pub fn uniform_buffer(mut self, stage: vk::ShaderStageFlags, buffer: BufferHandle) -> Self {
        self.writes
            .push(DescriptorWrite::uniform_buffer(stage, buffer));
        self
    }

    pub fn storage_buffer(mut self, stage: vk::ShaderStageFlags, buffer: BufferHandle) -> Self {
        self.writes
            .push(DescriptorWrite::storage_buffer(stage, buffer));
        self
    }

    /// Allocates and writes a static descriptor set. The returned layout and set
    /// are owned by the graph and freed when the graph is dropped. Use this when
    /// the bound images and buffers never change.
    pub fn build(self) -> Result<(vk::DescriptorSetLayout, vk::DescriptorSet), GraphError> {
        let device = self.graph.ash_device().clone();
        let (layout, pool, set) = alloc_descriptor_set(&device, &self.writes)?;
        apply_writes(
            &device,
            set,
            &self.writes,
            self.graph.images_slice(),
            self.graph.resources_ref(),
        );
        self.graph
            .push_owned_desc(OwnedDescriptorResources { pool, layout });
        Ok((layout, set))
    }

    /// Allocates and writes a dynamic descriptor set. Unlike [`build`](DescriptorSetBuilder::build),
    /// the returned [`DynamicDescriptorSet`] can be re-written after a resize by
    /// calling [`DynamicDescriptorSet::update`]. Use this for any set that references
    /// resizable images.
    pub fn build_dynamic(self) -> Result<DynamicDescriptorSet, GraphError> {
        let device = self.graph.ash_device().clone();
        let (layout, pool, set) = alloc_descriptor_set(&device, &self.writes)?;
        apply_writes(
            &device,
            set,
            &self.writes,
            self.graph.images_slice(),
            self.graph.resources_ref(),
        );
        self.graph
            .push_owned_desc(OwnedDescriptorResources { pool, layout });
        Ok(DynamicDescriptorSet {
            layout,
            set,
            writes: self.writes,
        })
    }
}

/// A descriptor set that can be re-written after a window resize.
///
/// Created by [`DescriptorSetBuilder::build_dynamic`]. The `layout` and `set`
/// fields are public so you can pass them to pipeline builders and
/// [`Cmd::bind_descriptor_sets`](super::command::Cmd::bind_descriptor_sets).
pub struct DynamicDescriptorSet {
    /// The `VkDescriptorSetLayout` for this set. Pass to
    /// [`PipelineBuilder::descriptor_set_layouts`](super::pipeline::PipelineBuilder::descriptor_set_layouts).
    pub layout: vk::DescriptorSetLayout,
    /// The `VkDescriptorSet`. Pass to
    /// [`Cmd::bind_descriptor_sets`](super::command::Cmd::bind_descriptor_sets).
    pub set: vk::DescriptorSet,
    writes: Vec<DescriptorWrite>,
}

impl DynamicDescriptorSet {
    /// Re-writes the descriptor set with the current image views from the graph.
    ///
    /// Call this when `frame.resized` is `true`, before recording any pass that
    /// uses this set. This is necessary because resizable images get new
    /// `VkImageView` handles when the swapchain is recreated.
    pub fn update(&self, graph: &Graph) {
        apply_writes(
            graph.ash_device(),
            self.set,
            &self.writes,
            graph.images_slice(),
            graph.resources_ref(),
        );
    }
}

fn create_layout(
    device: &ash::Device,
    writes: &[DescriptorWrite],
) -> Result<vk::DescriptorSetLayout, GraphError> {
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = writes
        .iter()
        .enumerate()
        .map(|(i, w)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i as u32)
                .descriptor_type(w.kind.descriptor_type())
                .descriptor_count(1)
                .stage_flags(w.stage)
        })
        .collect();

    Ok(unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?
    })
}

fn create_pool(
    device: &ash::Device,
    writes: &[DescriptorWrite],
) -> Result<vk::DescriptorPool, GraphError> {
    let mut counts = std::collections::HashMap::<vk::DescriptorType, u32>::new();
    for w in writes {
        *counts.entry(w.kind.descriptor_type()).or_insert(0) += 1;
    }

    let pool_sizes: Vec<vk::DescriptorPoolSize> = counts
        .into_iter()
        .map(|(ty, descriptor_count)| vk::DescriptorPoolSize {
            ty,
            descriptor_count,
        })
        .collect();

    Ok(unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )?
    })
}

fn alloc_descriptor_set(
    device: &ash::Device,
    writes: &[DescriptorWrite],
) -> Result<
    (
        vk::DescriptorSetLayout,
        vk::DescriptorPool,
        vk::DescriptorSet,
    ),
    GraphError,
> {
    let layout = create_layout(device, writes)?;

    let pool = create_pool(device, writes).inspect_err(|_| {
        unsafe { device.destroy_descriptor_set_layout(layout, None) };
    })?;

    let sets = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&[layout]),
        )
    }
    .map_err(|e| {
        unsafe {
            device.destroy_descriptor_set_layout(layout, None);
            device.destroy_descriptor_pool(pool, None);
        }
        GraphError::from(e)
    })?;

    Ok((layout, pool, sets[0]))
}

fn resolve_infos<'a>(
    kinds: impl Iterator<Item = &'a DescriptorKind>,
    images: &[ImageEntry],
    pool: &ResourcePool,
) -> (Vec<vk::DescriptorImageInfo>, Vec<vk::DescriptorBufferInfo>) {
    let mut image_infos = Vec::new();
    let mut buffer_infos = Vec::new();

    for kind in kinds {
        match kind {
            DescriptorKind::StorageImage { image } => {
                image_infos.push(
                    vk::DescriptorImageInfo::default()
                        .image_view(images[image.0 as usize].view(pool))
                        .image_layout(vk::ImageLayout::GENERAL),
                );
            }
            DescriptorKind::SampledImage { image } => {
                image_infos.push(
                    vk::DescriptorImageInfo::default()
                        .image_view(images[image.0 as usize].view(pool))
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                );
            }
            DescriptorKind::CombinedSampler { sampler, image } => {
                let raw = pool
                    .get_sampler(*sampler)
                    .expect("sampler handle stale — destroyed before descriptor update");
                image_infos.push(
                    vk::DescriptorImageInfo::default()
                        .sampler(raw)
                        .image_view(images[image.0 as usize].view(pool))
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                );
            }
            DescriptorKind::UniformBuffer { buffer } => {
                let raw = pool
                    .get_buffer(*buffer)
                    .expect("uniform buffer handle stale — destroyed before descriptor update")
                    .raw;
                buffer_infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(raw)
                        .offset(0)
                        .range(vk::WHOLE_SIZE),
                );
            }
            DescriptorKind::StorageBuffer { buffer } => {
                let raw = pool
                    .get_buffer(*buffer)
                    .expect("storage buffer handle stale — destroyed before descriptor update")
                    .raw;
                buffer_infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(raw)
                        .offset(0)
                        .range(vk::WHOLE_SIZE),
                );
            }
        }
    }

    (image_infos, buffer_infos)
}

fn build_vk_writes<'img, 'k>(
    dst_set: vk::DescriptorSet,
    kinds: impl Iterator<Item = &'k DescriptorKind>,
    image_infos: &'img [vk::DescriptorImageInfo],
    buffer_infos: &'img [vk::DescriptorBufferInfo],
) -> Vec<vk::WriteDescriptorSet<'img>> {
    let mut img_i = 0usize;
    let mut buf_i = 0usize;

    kinds
        .enumerate()
        .map(|(binding, kind)| {
            let base = vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(binding as u32)
                .descriptor_type(kind.descriptor_type());
            match kind {
                DescriptorKind::StorageImage { .. }
                | DescriptorKind::SampledImage { .. }
                | DescriptorKind::CombinedSampler { .. } => {
                    let r = base.image_info(std::slice::from_ref(&image_infos[img_i]));
                    img_i += 1;
                    r
                }
                DescriptorKind::UniformBuffer { .. } | DescriptorKind::StorageBuffer { .. } => {
                    let r = base.buffer_info(std::slice::from_ref(&buffer_infos[buf_i]));
                    buf_i += 1;
                    r
                }
            }
        })
        .collect()
}

pub(crate) fn apply_writes(
    device: &ash::Device,
    set: vk::DescriptorSet,
    writes: &[DescriptorWrite],
    images: &[ImageEntry],
    pool: &ResourcePool,
) {
    let (image_infos, buffer_infos) = resolve_infos(writes.iter().map(|w| &w.kind), images, pool);
    let vk_writes = build_vk_writes(
        set,
        writes.iter().map(|w| &w.kind),
        &image_infos,
        &buffer_infos,
    );
    unsafe { device.update_descriptor_sets(&vk_writes, &[]) };
}

pub(crate) fn apply_push_writes(
    writes: &[PushDescriptor],
    images: &[ImageEntry],
    pool: &ResourcePool,
    call: impl for<'a> FnOnce(&'a [vk::WriteDescriptorSet<'a>]),
) {
    let (image_infos, buffer_infos) = resolve_infos(writes.iter().map(|w| &w.kind), images, pool);
    let vk_writes = build_vk_writes(
        vk::DescriptorSet::null(),
        writes.iter().map(|w| &w.kind),
        &image_infos,
        &buffer_infos,
    );
    call(&vk_writes);
}
