use ash::vk;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::resource::BufferHandle;

use super::image::{Image, ImageEntry};
use super::pass::{BufferAccess, PassAccess};

#[derive(Clone)]
pub(super) struct LayerState {
    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

#[derive(Clone)]
pub(super) struct BarrierState {
    pub layers: SmallVec<[LayerState; 1]>,
}

impl BarrierState {
    pub(super) fn from_entry(e: &ImageEntry) -> Self {
        let layer_count = e.layer_count();
        let base = LayerState {
            layout: e.layout,
            stage: e.stage,
            access: e.access,
        };
        Self {
            layers: smallvec::smallvec![base; layer_count as usize],
        }
    }

    pub(super) fn representative(&self) -> LayerState {
        let first = &self.layers[0];
        let uniform = self.layers.iter().all(|l| {
            l.layout == first.layout && l.stage == first.stage && l.access == first.access
        });
        if uniform {
            first.clone()
        } else {
            LayerState {
                layout: vk::ImageLayout::UNDEFINED,
                stage: vk::PipelineStageFlags2::NONE,
                access: vk::AccessFlags2::NONE,
            }
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
    pub image: Image,
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

fn needs_layer_barrier(state: &LayerState, next: &PassAccess) -> bool {
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

        match access.layer {
            Some(l) => {
                let ls = &state.layers[l as usize];
                if needs_layer_barrier(ls, access) {
                    infos.push(BarrierInfo {
                        image: access.image,
                        old_layout: ls.layout,
                        new_layout: access.layout,
                        src_stage: ls.stage,
                        src_access: ls.access,
                        dst_stage: access.stage,
                        dst_access: access.access,
                        layer: Some(l),
                    });
                }
                states[idx].layers[l as usize] = LayerState {
                    layout: access.layout,
                    stage: access.stage,
                    access: access.access,
                };
            }
            None => {
                let all_same = state.layers.iter().all(|ls| {
                    ls.layout == state.layers[0].layout
                        && ls.stage == state.layers[0].stage
                        && ls.access == state.layers[0].access
                });

                if all_same {
                    if needs_layer_barrier(&state.layers[0], access) {
                        infos.push(BarrierInfo {
                            image: access.image,
                            old_layout: state.layers[0].layout,
                            new_layout: access.layout,
                            src_stage: state.layers[0].stage,
                            src_access: state.layers[0].access,
                            dst_stage: access.stage,
                            dst_access: access.access,
                            layer: None,
                        });
                    }
                } else {
                    for (l, ls) in state.layers.iter().enumerate() {
                        if needs_layer_barrier(ls, access) {
                            infos.push(BarrierInfo {
                                image: access.image,
                                old_layout: ls.layout,
                                new_layout: access.layout,
                                src_stage: ls.stage,
                                src_access: ls.access,
                                dst_stage: access.stage,
                                dst_access: access.access,
                                layer: Some(l as u32),
                            });
                        }
                    }
                }

                let new_ls = LayerState {
                    layout: access.layout,
                    stage: access.stage,
                    access: access.access,
                };
                for ls in &mut states[idx].layers {
                    *ls = new_ls.clone();
                }
            }
        }
    }

    if infos.is_empty() { None } else { Some(infos) }
}

pub(super) fn compute_buffer_barriers(
    reads: &[BufferAccess],
    writes: &[BufferAccess],
    states: &mut FxHashMap<BufferHandle, BufferBarrierState>,
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
    use ash::vk;
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::graph::access::LoadOp;
    use crate::graph::image::Image;
    use crate::graph::pass::{BufferAccess, PassAccess};
    use crate::resource::BufferHandle;

    fn img_state(
        layout: vk::ImageLayout,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> BarrierState {
        BarrierState {
            layers: smallvec::smallvec![LayerState { layout, stage, access }],
        }
    }

    fn img_state_layers(
        layer_count: u32,
        layout: vk::ImageLayout,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> BarrierState {
        BarrierState {
            layers: smallvec::smallvec![LayerState { layout, stage, access }; layer_count as usize],
        }
    }

    fn img_access(
        id: u32,
        layout: vk::ImageLayout,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> PassAccess {
        PassAccess {
            image: Image(id),
            layout,
            stage,
            access,
            is_color: false,
            is_depth: false,
            load_op: LoadOp::Auto,
            layer: None,
            clear_color: None,
        }
    }

    fn img_access_layer(
        id: u32,
        layout: vk::ImageLayout,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        layer: u32,
    ) -> PassAccess {
        PassAccess {
            image: Image(id),
            layout,
            stage,
            access,
            is_color: false,
            is_depth: false,
            load_op: LoadOp::Auto,
            layer: Some(layer),
            clear_color: None,
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
        let result = compute_barriers(&[], &writes, &mut states).expect("test invariant");
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
        let result = compute_barriers(&reads, &[], &mut states).expect("test invariant");
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].src_access,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
        );
    }

    #[test]
    fn per_layer_writes_use_correct_old_layout() {
        let mut states = vec![img_state_layers(
            4,
            vk::ImageLayout::UNDEFINED,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
        )];

        let depth_layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        let depth_stage = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS;
        let depth_access = vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;

        for layer in 0..4u32 {
            let writes = vec![img_access_layer(0, depth_layout, depth_stage, depth_access, layer)];
            let result = compute_barriers(&[], &writes, &mut states).expect("barrier expected");
            assert_eq!(result.len(), 1);
            assert_eq!(result[0].old_layout, vk::ImageLayout::UNDEFINED);
            assert_eq!(result[0].new_layout, depth_layout);
            assert_eq!(result[0].layer, Some(layer));
        }
    }

    #[test]
    fn whole_image_read_after_per_layer_writes() {
        let mut states = vec![img_state_layers(
            4,
            vk::ImageLayout::UNDEFINED,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
        )];

        let depth_layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        let depth_stage = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS;
        let depth_access = vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;

        for layer in 0..4u32 {
            let writes = vec![img_access_layer(0, depth_layout, depth_stage, depth_access, layer)];
            compute_barriers(&[], &writes, &mut states);
        }

        let read_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let reads = vec![img_access(
            0,
            read_layout,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
        )];
        let result = compute_barriers(&reads, &[], &mut states).expect("barrier expected");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].old_layout, depth_layout);
        assert_eq!(result[0].new_layout, read_layout);
        assert!(result[0].layer.is_none());
    }

    #[test]
    fn whole_image_read_after_partial_layer_writes() {
        let mut states = vec![img_state_layers(
            4,
            vk::ImageLayout::UNDEFINED,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
        )];

        let depth_layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        let depth_stage = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS;
        let depth_access = vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;

        for layer in 0..2u32 {
            let writes = vec![img_access_layer(0, depth_layout, depth_stage, depth_access, layer)];
            compute_barriers(&[], &writes, &mut states);
        }

        let read_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let reads = vec![img_access(
            0,
            read_layout,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
        )];
        let result = compute_barriers(&reads, &[], &mut states).expect("barrier expected");
        assert_eq!(result.len(), 4);
        for info in &result {
            assert_eq!(info.new_layout, read_layout);
            assert!(info.layer.is_some());
        }
        assert_eq!(result[0].old_layout, depth_layout);
        assert_eq!(result[1].old_layout, depth_layout);
        assert_eq!(result[2].old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(result[3].old_layout, vk::ImageLayout::UNDEFINED);
    }

    #[test]
    fn buffer_barrier_after_write() {
        let mut states = FxHashMap::default();
        let buf = BufferHandle::default();

        let writes = vec![BufferAccess {
            handle: buf,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_WRITE,
        }];
        let first = compute_buffer_barriers(&[], &writes, &mut states).expect("test invariant");
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].src_access, vk::AccessFlags2::NONE);

        let reads = vec![BufferAccess {
            handle: buf,
            stage: vk::PipelineStageFlags2::VERTEX_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
        }];
        let second = compute_buffer_barriers(&reads, &[], &mut states).expect("test invariant");
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].src_access, vk::AccessFlags2::SHADER_WRITE);
    }
}
