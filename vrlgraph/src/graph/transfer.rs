use ash::vk;

use rustc_hash::FxHashMap;

use crate::resource::BufferHandle;

use super::GraphError;
use super::cmd::CommandPool;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TransferId(u64);

pub(crate) enum PendingTransfer {
    BufferToBuffer {
        src: vk::Buffer,
        dst: vk::Buffer,
        size: vk::DeviceSize,
        staging_handle: BufferHandle,
    },
    BufferToImage {
        src: vk::Buffer,
        dst: vk::Image,
        regions: Vec<vk::BufferImageCopy>,
        staging_handle: BufferHandle,
        subresource_range: vk::ImageSubresourceRange,
    },
}

pub(crate) struct AcquireInfo {
    pub timeline_value: u64,
    pub kind: AcquireKind,
}

pub(crate) enum AcquireKind {
    Buffer {
        raw: vk::Buffer,
        size: vk::DeviceSize,
    },
    Image {
        raw: vk::Image,
        dst_layout: vk::ImageLayout,
        subresource_range: vk::ImageSubresourceRange,
    },
}

pub(crate) struct ImageTransferDesc {
    pub src: vk::Buffer,
    pub dst: vk::Image,
    pub regions: Vec<vk::BufferImageCopy>,
    pub staging_handle: BufferHandle,
    pub subresource_range: vk::ImageSubresourceRange,
    pub dst_layout: vk::ImageLayout,
    pub mip_gen: Option<(vk::Extent3D, u32)>,
}

pub(crate) struct DeferredMipGen {
    pub timeline_value: u64,
    pub image: vk::Image,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
}

pub(crate) struct TransferManager {
    queue: vk::Queue,
    transfer_family: u32,
    graphics_family: u32,
    is_dedicated: bool,
    timeline: vk::Semaphore,
    next_value: u64,
    last_completed: u64,
    pool: CommandPool,
    pending: Vec<PendingTransfer>,
    deferred_staging: Vec<(u64, BufferHandle)>,
    pending_acquires: Vec<AcquireInfo>,
    pub(crate) deferred_mip_gens: Vec<DeferredMipGen>,
    pending_buffers: FxHashMap<BufferHandle, TransferId>,
    device: ash::Device,
}

impl TransferManager {
    pub fn new(
        device: ash::Device,
        queue: vk::Queue,
        transfer_family: u32,
        graphics_family: u32,
    ) -> Result<Self, GraphError> {
        let is_dedicated = transfer_family != graphics_family;

        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let timeline = unsafe { device.create_semaphore(&create_info, None)? };

        let pool = CommandPool::new(&device, transfer_family)?;

        Ok(Self {
            queue,
            transfer_family,
            graphics_family,
            is_dedicated,
            timeline,
            next_value: 1,
            last_completed: 0,
            pool,
            pending: Vec::new(),
            deferred_staging: Vec::new(),
            pending_acquires: Vec::new(),
            deferred_mip_gens: Vec::new(),
            pending_buffers: FxHashMap::default(),
            device,
        })
    }

    pub fn timeline_semaphore(&self) -> vk::Semaphore {
        self.timeline
    }

    pub fn transfer_family(&self) -> u32 {
        self.transfer_family
    }

    pub fn enqueue_buffer(
        &mut self,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: vk::DeviceSize,
        staging_handle: BufferHandle,
    ) -> TransferId {
        let id = TransferId(self.next_value);

        if self.is_dedicated {
            self.pending_acquires.push(AcquireInfo {
                timeline_value: self.next_value,
                kind: AcquireKind::Buffer { raw: dst, size },
            });
        }

        self.pending.push(PendingTransfer::BufferToBuffer {
            src,
            dst,
            size,
            staging_handle,
        });

        id
    }

    pub fn enqueue_image(&mut self, desc: ImageTransferDesc) -> TransferId {
        let id = TransferId(self.next_value);

        if self.is_dedicated {
            self.pending_acquires.push(AcquireInfo {
                timeline_value: self.next_value,
                kind: AcquireKind::Image {
                    raw: desc.dst,
                    dst_layout: desc.dst_layout,
                    subresource_range: desc.subresource_range,
                },
            });
        }

        if let Some((extent, mip_levels)) = desc.mip_gen {
            self.deferred_mip_gens.push(DeferredMipGen {
                timeline_value: self.next_value,
                image: desc.dst,
                extent,
                mip_levels,
            });
        }

        self.pending.push(PendingTransfer::BufferToImage {
            src: desc.src,
            dst: desc.dst,
            regions: desc.regions,
            staging_handle: desc.staging_handle,
            subresource_range: desc.subresource_range,
        });

        id
    }

    pub fn submit_pending(&mut self) -> Result<u64, GraphError> {
        if self.pending.is_empty() {
            return Ok(self.last_completed);
        }

        let cb = self.pool.reset_and_begin()?;

        for op in &self.pending {
            match op {
                PendingTransfer::BufferToBuffer { src, dst, size, .. } => {
                    let region = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: *size,
                    };
                    unsafe { self.device.cmd_copy_buffer(cb, *src, *dst, &[region]) };
                }
                PendingTransfer::BufferToImage {
                    src,
                    dst,
                    regions,
                    subresource_range,
                    ..
                } => {
                    let barrier = vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(*dst)
                        .subresource_range(*subresource_range);

                    let dep_info = vk::DependencyInfo::default()
                        .image_memory_barriers(std::slice::from_ref(&barrier));
                    unsafe { self.device.cmd_pipeline_barrier2(cb, &dep_info) };

                    unsafe {
                        self.device.cmd_copy_buffer_to_image(
                            cb,
                            *src,
                            *dst,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            regions,
                        );
                    }
                }
            }
        }

        if self.is_dedicated {
            self.record_release_barriers(cb);
        } else {
            self.record_final_transitions(cb);
        }

        unsafe { self.device.end_command_buffer(cb)? };

        let signal_value = self.next_value;
        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline)
            .value(signal_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER);

        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cb);

        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_info));

        unsafe {
            self.device
                .queue_submit2(self.queue, &[submit], vk::Fence::null())?
        };

        for op in self.pending.drain(..) {
            let staging = match op {
                PendingTransfer::BufferToBuffer { staging_handle, .. } => staging_handle,
                PendingTransfer::BufferToImage { staging_handle, .. } => staging_handle,
            };
            self.deferred_staging.push((signal_value, staging));
        }

        self.next_value += 1;

        Ok(signal_value)
    }

    fn record_release_barriers(&self, cb: vk::CommandBuffer) {
        let mut buffer_barriers = Vec::new();
        let mut image_barriers = Vec::new();

        for op in &self.pending {
            match op {
                PendingTransfer::BufferToBuffer { dst, size, .. } => {
                    buffer_barriers.push(
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                            .dst_access_mask(vk::AccessFlags2::NONE)
                            .src_queue_family_index(self.transfer_family)
                            .dst_queue_family_index(self.graphics_family)
                            .buffer(*dst)
                            .offset(0)
                            .size(*size),
                    );
                }
                PendingTransfer::BufferToImage {
                    dst,
                    subresource_range,
                    ..
                } => {
                    image_barriers.push(
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                            .dst_access_mask(vk::AccessFlags2::NONE)
                            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .src_queue_family_index(self.transfer_family)
                            .dst_queue_family_index(self.graphics_family)
                            .image(*dst)
                            .subresource_range(*subresource_range),
                    );
                }
            }
        }

        if !buffer_barriers.is_empty() || !image_barriers.is_empty() {
            let dep_info = vk::DependencyInfo::default()
                .buffer_memory_barriers(&buffer_barriers)
                .image_memory_barriers(&image_barriers);
            unsafe { self.device.cmd_pipeline_barrier2(cb, &dep_info) };
        }
    }

    fn record_final_transitions(&self, cb: vk::CommandBuffer) {
        let mut image_barriers = Vec::new();

        for op in &self.pending {
            if let PendingTransfer::BufferToImage {
                dst,
                subresource_range,
                ..
            } = op
            {
                let has_deferred_mips = self.deferred_mip_gens.iter().any(|m| m.image == *dst);

                if !has_deferred_mips {
                    image_barriers.push(
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(*dst)
                            .subresource_range(*subresource_range),
                    );
                }
            }
        }

        if !image_barriers.is_empty() {
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
            unsafe { self.device.cmd_pipeline_barrier2(cb, &dep_info) };
        }
    }

    pub fn poll_completed(&mut self) -> u64 {
        let val = unsafe {
            self.device
                .get_semaphore_counter_value(self.timeline)
                .unwrap_or(self.last_completed)
        };
        self.last_completed = val;
        val
    }

    pub fn wait_for(&self, id: TransferId) -> Result<(), GraphError> {
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&self.timeline))
            .values(std::slice::from_ref(&id.0));
        unsafe { self.device.wait_semaphores(&wait_info, u64::MAX)? };
        Ok(())
    }

    pub fn wait_idle(&mut self) -> Result<(), GraphError> {
        if self.next_value <= 1 {
            return Ok(());
        }
        let target = self.next_value - 1;
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&self.timeline))
            .values(std::slice::from_ref(&target));
        unsafe { self.device.wait_semaphores(&wait_info, u64::MAX)? };
        self.last_completed = target;
        Ok(())
    }

    pub fn take_completed_acquires(&mut self) -> Vec<AcquireInfo> {
        self.poll_completed();
        let completed = self.last_completed;
        let mut taken = Vec::new();
        let mut remaining = Vec::new();
        for a in self.pending_acquires.drain(..) {
            if a.timeline_value <= completed {
                taken.push(a);
            } else {
                remaining.push(a);
            }
        }
        self.pending_acquires = remaining;
        taken
    }

    pub fn cleanup_staging(
        &mut self,
        resources: &mut crate::resource::ResourcePool,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        self.poll_completed();
        let completed = self.last_completed;
        self.deferred_staging.retain(|(val, handle)| {
            if *val <= completed {
                resources.destroy_buffer(device, allocator, *handle);
                false
            } else {
                true
            }
        });
    }

    pub fn take_completed_mip_gens(&mut self) -> Vec<DeferredMipGen> {
        let completed = self.last_completed;
        let mut taken = Vec::new();
        let mut remaining = Vec::new();
        for m in self.deferred_mip_gens.drain(..) {
            if m.timeline_value <= completed {
                taken.push(m);
            } else {
                remaining.push(m);
            }
        }
        self.deferred_mip_gens = remaining;
        taken
    }

    pub fn wait_for_buffer(&mut self, handle: BufferHandle) {
        if let Some(id) = self.pending_buffers.remove(&handle) {
            let _ = self.wait_for(id);
        }
    }

    pub fn track_buffer(&mut self, handle: BufferHandle, id: TransferId) {
        self.pending_buffers.insert(handle, id);
    }

    pub fn is_buffer_ready_peek(&self, handle: BufferHandle) -> bool {
        match self.pending_buffers.get(&handle) {
            None => true,
            Some(id) => id.0 <= self.last_completed,
        }
    }
}

impl Drop for TransferManager {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_semaphore(self.timeline, None);
        }
    }
}
