use std::marker::PhantomData;

use ash::vk;

use super::GraphError;

const MAX_SAMPLED_IMAGES: u32 = 4096;
const MAX_STORAGE_IMAGES: u32 = 1024;
const MAX_CUBEMAP_IMAGES: u32 = 128;
const MAX_ARRAY_IMAGES: u32 = 256;
const MAX_SAMPLERS: u32 = 32;

/// Marker type for 2D sampled images (binding 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sampled;
/// Marker type for storage images (binding 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Storage;
/// Marker type for cubemap sampled images (binding 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cubemap;
/// Marker type for 2D array sampled images (binding 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Array2D;

/// Type-safe index into the global bindless descriptor table.
///
/// The phantom type `K` encodes which binding the index belongs to. Returned
/// by `FrameResources::sampled_index` and friends as an internal representation;
/// use those methods directly to get the `u32` for push constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindlessIndex<K>(u32, PhantomData<K>);

impl<K> BindlessIndex<K> {
    fn new(index: u32) -> Self {
        Self(index, PhantomData)
    }

    pub fn raw(self) -> u32 {
        self.0
    }
}

impl<K> From<BindlessIndex<K>> for u32 {
    fn from(idx: BindlessIndex<K>) -> u32 {
        idx.0
    }
}

/// A sampler registered in the global bindless table.
///
/// Returned by [`Graph::create_sampler`](crate::graph::Graph::create_sampler). Pass it to
/// [`FrameResources::sampler_index`](crate::graph::FrameResources::sampler_index) inside a
/// pass closure to get the `u32` index for push constants. Pass it to
/// [`Graph::destroy_sampler`](crate::graph::Graph::destroy_sampler) to release it.
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    pub(crate) handle: crate::resource::SamplerHandle,
    index: u32,
}

impl Sampler {
    pub(crate) fn new(handle: crate::resource::SamplerHandle, index: u32) -> Self {
        Self { handle, index }
    }

    pub fn raw(&self) -> u32 {
        self.index
    }
}

fn allocate_slot(free: &mut Vec<u32>, count: &mut u32, max: u32, msg: &str) -> u32 {
    free.pop().unwrap_or_else(|| {
        let i = *count;
        *count += 1;
        assert!(i < max, "{}", msg);
        i
    })
}

/// A single, global descriptor set holding all sampled images, storage images,
/// and samplers for the entire frame.
///
/// Resources are registered once and accessed by index through push constants.
/// The set uses `UPDATE_AFTER_BIND` so descriptors can be written while the set
/// is bound to an in-flight command buffer.
///
/// Layout:
/// - binding 0: `texture2D     textures[]`      (2D sampled images)
/// - binding 1: `image2D       storage_imgs[]`  (storage images)
/// - binding 2: `sampler       samplers[]`
/// - binding 3: `textureCube   cube_textures[]` (cubemap sampled images)
/// - binding 4: `texture2DArray array_textures[]` (2D array sampled images)
pub struct BindlessDescriptorTable {
    device: ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,

    sampled_free: Vec<u32>,
    sampled_count: u32,

    storage_free: Vec<u32>,
    storage_count: u32,

    cubemap_free: Vec<u32>,
    cubemap_count: u32,

    array_free: Vec<u32>,
    array_count: u32,

    sampler_count: u32,
}

impl BindlessDescriptorTable {
    pub fn new(device: &ash::Device, push_constant_size: u32) -> Result<Self, GraphError> {
        let uab = vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::PARTIALLY_BOUND;
        let binding_flags = [uab, uab, uab, uab, uab];

        let mut binding_flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(MAX_SAMPLED_IMAGES)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(MAX_STORAGE_IMAGES)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(MAX_SAMPLERS)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(MAX_CUBEMAP_IMAGES)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(MAX_ARRAY_IMAGES)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&bindings)
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut binding_flags_info),
                None,
            )?
        };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: MAX_SAMPLED_IMAGES + MAX_CUBEMAP_IMAGES + MAX_ARRAY_IMAGES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: MAX_STORAGE_IMAGES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: MAX_SAMPLERS,
            },
        ];

        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes)
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                None,
            )
        }
        .inspect_err(|_| unsafe {
            device.destroy_descriptor_set_layout(layout, None);
        })?;

        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&[layout]),
            )
        }
        .map(|sets| sets[0])
        .inspect_err(|_| unsafe {
            device.destroy_descriptor_pool(pool, None);
            device.destroy_descriptor_set_layout(layout, None);
        })?;

        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::ALL,
            offset: 0,
            size: push_constant_size,
        };

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[layout])
                    .push_constant_ranges(&[push_range]),
                None,
            )
        }
        .inspect_err(|_| unsafe {
            device.destroy_descriptor_pool(pool, None);
            device.destroy_descriptor_set_layout(layout, None);
        })?;

        Ok(Self {
            device: device.clone(),
            pool,
            layout,
            set,
            pipeline_layout,
            sampled_free: Vec::new(),
            sampled_count: 0,
            storage_free: Vec::new(),
            storage_count: 0,
            cubemap_free: Vec::new(),
            cubemap_count: 0,
            array_free: Vec::new(),
            array_count: 0,
            sampler_count: 0,
        })
    }

    /// Registers a sampled image and returns its bindless index.
    pub fn allocate_sampled_image(
        &mut self,
        view: vk::ImageView,
        image_layout: vk::ImageLayout,
    ) -> BindlessIndex<Sampled> {
        let index = allocate_slot(
            &mut self.sampled_free,
            &mut self.sampled_count,
            MAX_SAMPLED_IMAGES,
            "bindless sampled image table full",
        );
        self.write_sampled(index, view, image_layout);
        BindlessIndex::new(index)
    }

    /// Registers a storage image and returns its bindless index.
    pub fn allocate_storage_image(&mut self, view: vk::ImageView) -> BindlessIndex<Storage> {
        let index = allocate_slot(
            &mut self.storage_free,
            &mut self.storage_count,
            MAX_STORAGE_IMAGES,
            "bindless storage image table full",
        );
        self.write_storage(index, view);
        BindlessIndex::new(index)
    }

    /// Updates an existing sampled image slot (e.g. after a resize).
    pub fn update_sampled_image(
        &self,
        index: BindlessIndex<Sampled>,
        view: vk::ImageView,
        image_layout: vk::ImageLayout,
    ) {
        self.write_sampled(index.raw(), view, image_layout);
    }

    /// Updates an existing storage image slot (e.g. after a resize).
    pub fn update_storage_image(&self, index: BindlessIndex<Storage>, view: vk::ImageView) {
        self.write_storage(index.raw(), view);
    }

    /// Writes a sampler into the table and returns its index.
    pub fn write_sampler(&mut self, sampler: vk::Sampler) -> u32 {
        let index = self.sampler_count;
        self.sampler_count += 1;
        assert!(index < MAX_SAMPLERS, "bindless sampler table full");

        let info = vk::DescriptorImageInfo::default().sampler(sampler);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(2)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(std::slice::from_ref(&info));
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        index
    }

    /// Registers a cubemap image (binding 3) and returns its bindless index.
    pub fn allocate_cubemap_image(
        &mut self,
        view: vk::ImageView,
        image_layout: vk::ImageLayout,
    ) -> BindlessIndex<Cubemap> {
        let index = allocate_slot(
            &mut self.cubemap_free,
            &mut self.cubemap_count,
            MAX_CUBEMAP_IMAGES,
            "bindless cubemap table full",
        );
        self.write_cubemap(index, view, image_layout);
        BindlessIndex::new(index)
    }

    /// Registers a 2D array image (binding 4) and returns its bindless index.
    pub fn allocate_array_image(
        &mut self,
        view: vk::ImageView,
        image_layout: vk::ImageLayout,
    ) -> BindlessIndex<Array2D> {
        let index = allocate_slot(
            &mut self.array_free,
            &mut self.array_count,
            MAX_ARRAY_IMAGES,
            "bindless array texture table full",
        );
        self.write_array(index, view, image_layout);
        BindlessIndex::new(index)
    }

    /// Updates an existing cubemap slot (e.g. after a resize).
    pub fn update_cubemap_image(
        &self,
        index: BindlessIndex<Cubemap>,
        view: vk::ImageView,
        image_layout: vk::ImageLayout,
    ) {
        self.write_cubemap(index.raw(), view, image_layout);
    }

    /// Updates an existing 2D array slot (e.g. after a resize).
    pub fn update_array_image(
        &self,
        index: BindlessIndex<Array2D>,
        view: vk::ImageView,
        image_layout: vk::ImageLayout,
    ) {
        self.write_array(index.raw(), view, image_layout);
    }

    /// Releases a sampled image slot for reuse.
    pub fn free_sampled(&mut self, index: BindlessIndex<Sampled>) {
        self.sampled_free.push(index.raw());
    }

    /// Releases a storage image slot for reuse.
    pub fn free_storage(&mut self, index: BindlessIndex<Storage>) {
        self.storage_free.push(index.raw());
    }

    /// Releases a cubemap slot for reuse.
    pub fn free_cubemap(&mut self, index: BindlessIndex<Cubemap>) {
        self.cubemap_free.push(index.raw());
    }

    /// Releases a 2D array slot for reuse.
    pub fn free_array(&mut self, index: BindlessIndex<Array2D>) {
        self.array_free.push(index.raw());
    }

    pub fn set(&self) -> vk::DescriptorSet {
        self.set
    }

    #[allow(dead_code)]
    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn destroy(&self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }

    fn write_image_descriptor(
        &self,
        binding: u32,
        index: u32,
        view: vk::ImageView,
        layout: vk::ImageLayout,
        descriptor_type: vk::DescriptorType,
    ) {
        let info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(layout);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(binding)
            .dst_array_element(index)
            .descriptor_type(descriptor_type)
            .image_info(std::slice::from_ref(&info));
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
    }

    fn write_sampled(&self, index: u32, view: vk::ImageView, layout: vk::ImageLayout) {
        self.write_image_descriptor(0, index, view, layout, vk::DescriptorType::SAMPLED_IMAGE);
    }

    fn write_storage(&self, index: u32, view: vk::ImageView) {
        self.write_image_descriptor(
            1,
            index,
            view,
            vk::ImageLayout::GENERAL,
            vk::DescriptorType::STORAGE_IMAGE,
        );
    }

    fn write_cubemap(&self, index: u32, view: vk::ImageView, layout: vk::ImageLayout) {
        self.write_image_descriptor(3, index, view, layout, vk::DescriptorType::SAMPLED_IMAGE);
    }

    fn write_array(&self, index: u32, view: vk::ImageView, layout: vk::ImageLayout) {
        self.write_image_descriptor(4, index, view, layout, vk::DescriptorType::SAMPLED_IMAGE);
    }
}
