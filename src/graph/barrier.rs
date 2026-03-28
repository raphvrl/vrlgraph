use std::collections::HashMap;

use ash::vk;
use smallvec::SmallVec;

use crate::resource::BufferHandle;

use super::image::{GraphImage, ImageEntry};
use super::pass::{BufferAccess, PassAccess};

#[derive(Clone)]
pub(super) struct BarrierState {
    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

impl BarrierState {
    pub(super) fn from_entry(e: &ImageEntry) -> Self {
        Self {
            layout: e.layout,
            stage: e.stage,
            access: e.access,
        }
    }
}

#[derive(Clone)]
pub(crate) struct BufferBarrierState {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

impl Default for BufferBarrierState {
    fn default() -> Self {
        Self {
            stage: vk::PipelineStageFlags2::NONE,
            access: vk::AccessFlags2::NONE,
        }
    }
}

pub(super) struct BarrierInfo {
    pub image: GraphImage,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub src_stage: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    pub dst_stage: vk::PipelineStageFlags2,
    pub dst_access: vk::AccessFlags2,
    pub layer: Option<u32>,
}

pub(super) struct BufferBarrierInfo {
    pub handle: BufferHandle,
    pub src_stage: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    pub dst_stage: vk::PipelineStageFlags2,
    pub dst_access: vk::AccessFlags2,
}

const WRITE_ACCESS: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
    vk::AccessFlags2::SHADER_WRITE.as_raw()
        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
        | vk::AccessFlags2::TRANSFER_WRITE.as_raw()
        | vk::AccessFlags2::MEMORY_WRITE.as_raw(),
);

fn needs_image_barrier(state: &BarrierState, next: &PassAccess) -> bool {
    state.layout != next.layout
        || state.access.intersects(WRITE_ACCESS)
        || next.access.intersects(WRITE_ACCESS)
}

pub(super) fn compute_barriers(
    reads: &[PassAccess],
    writes: &[PassAccess],
    states: &mut [BarrierState],
) -> Option<SmallVec<[BarrierInfo; 8]>> {
    let mut infos: SmallVec<[BarrierInfo; 8]> = SmallVec::new();

    for access in reads.iter().chain(writes.iter()) {
        let idx = access.image.0 as usize;
        let state = &states[idx];

        if needs_image_barrier(state, access) {
            infos.push(BarrierInfo {
                image: access.image,
                old_layout: state.layout,
                new_layout: access.layout,
                src_stage: state.stage,
                src_access: state.access,
                dst_stage: access.stage,
                dst_access: access.access,
                layer: access.layer,
            });
        }

        states[idx] = BarrierState {
            layout: access.layout,
            stage: access.stage,
            access: access.access,
        };
    }

    if infos.is_empty() { None } else { Some(infos) }
}

pub(super) fn compute_buffer_barriers(
    reads: &[BufferAccess],
    writes: &[BufferAccess],
    states: &mut HashMap<BufferHandle, BufferBarrierState>,
) -> Option<SmallVec<[BufferBarrierInfo; 4]>> {
    let mut infos: SmallVec<[BufferBarrierInfo; 4]> = SmallVec::new();

    for access in reads.iter().chain(writes.iter()) {
        let prev = states.entry(access.handle).or_default().clone();

        let needs_barrier =
            prev.access.intersects(WRITE_ACCESS) || access.access.intersects(WRITE_ACCESS);

        if needs_barrier {
            infos.push(BufferBarrierInfo {
                handle: access.handle,
                src_stage: prev.stage,
                src_access: prev.access,
                dst_stage: access.stage,
                dst_access: access.access,
            });
        }

        states.insert(
            access.handle,
            BufferBarrierState {
                stage: access.stage,
                access: access.access,
            },
        );
    }

    if infos.is_empty() { None } else { Some(infos) }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ash::vk;

    use super::*;
    use crate::graph::access::LoadOp;
    use crate::graph::image::GraphImage;
    use crate::graph::pass::{BufferAccess, PassAccess};
    use crate::resource::BufferHandle;

    fn img_state(
        layout: vk::ImageLayout,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> BarrierState {
        BarrierState {
            layout,
            stage,
            access,
        }
    }

    fn img_access(
        id: u32,
        layout: vk::ImageLayout,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> PassAccess {
        PassAccess {
            image: GraphImage(id),
            layout,
            stage,
            access,
            is_color: false,
            is_depth: false,
            load_op: LoadOp::Auto,
            layer: None,
        }
    }

    #[test]
    fn no_barrier_for_same_layout_read_only() {
        let layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let stage = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        let access = vk::AccessFlags2::SHADER_READ;
        let mut states = vec![img_state(layout, stage, access)];
        let reads = vec![img_access(0, layout, stage, access)];
        assert!(compute_barriers(&reads, &[], &mut states).is_none());
    }

    #[test]
    fn barrier_on_layout_change() {
        let mut states = vec![img_state(
            vk::ImageLayout::UNDEFINED,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
        )];
        let writes = vec![img_access(
            0,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        )];
        let result = compute_barriers(&[], &writes, &mut states).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(
            result[0].new_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
    }

    #[test]
    fn barrier_when_prev_state_had_write() {
        let mut states = vec![img_state(
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        )];
        let reads = vec![img_access(
            0,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
        )];
        let result = compute_barriers(&reads, &[], &mut states).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].src_access,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
        );
    }

    #[test]
    fn buffer_barrier_after_write() {
        let mut states = HashMap::new();
        let buf = BufferHandle::default();

        let writes = vec![BufferAccess {
            handle: buf,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_WRITE,
        }];
        let first = compute_buffer_barriers(&[], &writes, &mut states).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].src_access, vk::AccessFlags2::NONE);

        let reads = vec![BufferAccess {
            handle: buf,
            stage: vk::PipelineStageFlags2::VERTEX_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
        }];
        let second = compute_buffer_barriers(&reads, &[], &mut states).unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].src_access, vk::AccessFlags2::SHADER_WRITE);
    }
}
