use ash::vk;

use crate::graph::bindless::BindlessDescriptorTable;
use crate::graph::image::ImageEntry;
use crate::resource::ImageKind;

pub(in crate::graph) fn register_bindless(
    entry: &mut ImageEntry,
    bindless: &mut BindlessDescriptorTable,
    view: vk::ImageView,
) {
    if entry.usage.contains(vk::ImageUsageFlags::SAMPLED) {
        match entry.desc.kind {
            ImageKind::Cubemap | ImageKind::CubemapArray { .. } => {
                entry.cubemap_index = Some(
                    bindless
                        .allocate_cubemap_image(view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                );
            }
            ImageKind::Image2DArray { .. } => {
                entry.array_index = Some(
                    bindless.allocate_array_image(view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                );
            }
            ImageKind::Image2D => {
                entry.sampled_index = Some(
                    bindless
                        .allocate_sampled_image(view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                );
            }
        }
    }
    if entry.usage.contains(vk::ImageUsageFlags::STORAGE) {
        entry.storage_index = Some(bindless.allocate_storage_image(view));
    }
}

pub(in crate::graph) fn free_bindless(
    entry: &mut ImageEntry,
    bindless: &mut BindlessDescriptorTable,
) {
    if let Some(idx) = entry.sampled_index.take() {
        bindless.free_sampled(idx);
    }
    if let Some(idx) = entry.storage_index.take() {
        bindless.free_storage(idx);
    }
    if let Some(idx) = entry.cubemap_index.take() {
        bindless.free_cubemap(idx);
    }
    if let Some(idx) = entry.array_index.take() {
        bindless.free_array(idx);
    }
}

pub(in crate::graph) fn update_bindless(
    entry: &ImageEntry,
    bindless: &BindlessDescriptorTable,
    view: vk::ImageView,
) {
    if let Some(si) = entry.sampled_index {
        bindless.update_sampled_image(si, view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
    if let Some(si) = entry.storage_index {
        bindless.update_storage_image(si, view);
    }
    if let Some(si) = entry.cubemap_index {
        bindless.update_cubemap_image(si, view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
    if let Some(si) = entry.array_index {
        bindless.update_array_image(si, view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
}
