use ash::vk;

use super::bindless_ops::free_bindless;
use crate::graph::image::Image;
use crate::graph::transfer::{ImageTransferDesc, TransferId};
use crate::graph::{Graph, GraphError};
use crate::resource::ImageHandle;

impl Graph {
    pub fn destroy_image(&mut self, handle: Image) {
        assert!(
            !self.frame_active,
            "destroy_image() must be called outside the frame loop"
        );

        let entry = &mut self.images[handle.0 as usize];
        free_bindless(entry, &mut self.bindless);

        if let Some(h) = entry.handle.take() {
            let device = self.device.ash_device().clone();
            self.resources
                .destroy_image(&device, self.device.allocator_mut(), h);
        }
    }

    pub fn upload_to_image(
        &mut self,
        image: Image,
        data: &[u8],
        offset: [u32; 2],
        extent: [u32; 2],
    ) -> Result<(), GraphError> {
        let entry = &self.images[image.0 as usize];
        let handle = entry.handle.expect("upload_to_image: image not allocated");
        let old_layout = entry.layout;
        let old_stage = entry.stage;
        let old_access = entry.access;
        let vk_img = self
            .resources
            .get_image(handle)
            .expect("upload_to_image: image handle stale")
            .raw;

        let staging = self.create_staging(data, "staging_upload_sub")?;
        let stg_buf = self
            .resources
            .get_buffer(staging)
            .expect("buffer just created")
            .raw;

        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D {
                x: offset[0] as i32,
                y: offset[1] as i32,
                z: 0,
            })
            .image_extent(vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: 1,
            });

        self.one_shot_submit(|cmd| {
            cmd.pipeline_barrier2(&[vk::ImageMemoryBarrier2::default()
                .src_stage_mask(old_stage)
                .src_access_mask(old_access)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(vk_img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })]);

            cmd.copy_buffer_to_image_region(stg_buf, vk_img, &[region]);

            cmd.pipeline_barrier2(&[vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(vk_img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })]);
        })?;

        self.destroy_staging(staging);

        let entry = &mut self.images[image.0 as usize];
        entry.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        entry.stage = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        entry.access = vk::AccessFlags2::SHADER_READ;

        Ok(())
    }

    pub(in crate::graph) fn upload_image_data_with_mips(
        &mut self,
        dst: ImageHandle,
        mip_data: &[&[u8]],
        base_extent: vk::Extent3D,
    ) -> Result<(), GraphError> {
        let mut packed = Vec::new();
        let mut offsets = Vec::with_capacity(mip_data.len());
        for level in mip_data {
            offsets.push(packed.len());
            packed.extend_from_slice(level);
        }

        let staging = self.create_staging(&packed, "staging_upload_mips")?;

        let vk_img = self
            .resources
            .get_image(dst)
            .expect("image just created")
            .raw;
        let stg_buf = self
            .resources
            .get_buffer(staging)
            .expect("buffer just created")
            .raw;

        let mip_count = mip_data.len() as u32;

        let regions: Vec<vk::BufferImageCopy> = (0..mip_data.len())
            .map(|i| {
                vk::BufferImageCopy::default()
                    .buffer_offset(offsets[i] as vk::DeviceSize)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: i as u32,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: (base_extent.width >> i).max(1),
                        height: (base_extent.height >> i).max(1),
                        depth: 1,
                    })
            })
            .collect();

        self.one_shot_submit(|cmd| {
            cmd.pipeline_barrier2(&[vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(vk_img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: 1,
                })]);

            cmd.copy_buffer_to_image_region(stg_buf, vk_img, &regions);

            cmd.pipeline_barrier2(&[vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(vk_img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_count,
                    base_array_layer: 0,
                    layer_count: 1,
                })]);
        })?;

        self.destroy_staging(staging);
        Ok(())
    }

    pub(in crate::graph) fn upload_image_data_async(
        &mut self,
        dst: ImageHandle,
        pixels: &[u8],
        extent: vk::Extent3D,
        mip_levels: u32,
    ) -> Result<TransferId, GraphError> {
        let staging = self.create_staging(pixels, "staging_upload_async")?;

        let vk_img = self
            .resources
            .get_image(dst)
            .expect("image just created")
            .raw;
        let stg_buf = self
            .resources
            .get_buffer(staging)
            .expect("buffer just created")
            .raw;

        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(extent);

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: if mip_levels > 1 {
                vk::REMAINING_MIP_LEVELS
            } else {
                1
            },
            base_array_layer: 0,
            layer_count: 1,
        };

        let mip_gen = if mip_levels > 1 {
            Some((extent, mip_levels))
        } else {
            None
        };

        let id = self.transfer.enqueue_image(ImageTransferDesc {
            src: stg_buf,
            dst: vk_img,
            regions: vec![region],
            staging_handle: staging,
            subresource_range,
            dst_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            mip_gen,
        });
        self.transfer.submit_pending()?;

        Ok(id)
    }

    pub(crate) fn wait_for_transfer(&self, id: TransferId) -> Result<(), GraphError> {
        self.transfer.wait_for(id)
    }
}
