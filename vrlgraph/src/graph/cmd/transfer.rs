use ash::vk;

use super::Cmd;

impl<'a> Cmd<'a> {
    pub(crate) fn copy_buffer_to_image_region(
        &self,
        buffer: vk::Buffer,
        image: vk::Image,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.raw,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                regions,
            )
        };
    }

    pub(crate) fn pipeline_barrier2(&self, image_barriers: &[vk::ImageMemoryBarrier2]) {
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(image_barriers);
        unsafe { self.device.cmd_pipeline_barrier2(self.raw, &dep_info) };
    }

    pub(crate) fn pipeline_barrier2_mixed(
        &self,
        image_barriers: &[vk::ImageMemoryBarrier2],
        buffer_barriers: &[vk::BufferMemoryBarrier2],
    ) {
        let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(image_barriers)
            .buffer_memory_barriers(buffer_barriers);
        unsafe { self.device.cmd_pipeline_barrier2(self.raw, &dep_info) };
    }

    pub(crate) fn generate_mipmaps(&self, image: vk::Image, extent: vk::Extent3D, mip_levels: u32) {
        assert!(mip_levels >= 2, "generate_mipmaps: mip_levels must be >= 2");

        let mut mip_w =
            i32::try_from(extent.width).expect("generate_mipmaps: image width exceeds i32::MAX");
        let mut mip_h =
            i32::try_from(extent.height).expect("generate_mipmaps: image height exceeds i32::MAX");

        for i in 1..mip_levels {
            self.pipeline_barrier2(&[vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })]);

            let next_w = (mip_w / 2).max(1);
            let next_h = (mip_h / 2).max(1);

            let blit = vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_w,
                        y: mip_h,
                        z: 1,
                    },
                ])
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: next_w,
                        y: next_h,
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            unsafe {
                self.device.cmd_blit_image(
                    self.raw,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }

            mip_w = next_w;
            mip_h = next_h;
        }

        let barriers = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip_levels - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        ];

        self.pipeline_barrier2(&barriers);
    }
}
