use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::resource::{ImageDesc, ImageHandle, ResourceError, ResourcePool};

use super::image::{ImageEntry, ImageOrigin};
use super::pass::RecordedPass;

fn compute_lifetimes(
    passes: &[RecordedPass],
    images: &[ImageEntry],
    persistent_count: usize,
) -> Vec<Option<(usize, usize)>> {
    let mut lifetimes = vec![None::<(usize, usize)>; images.len()];

    for (pass_idx, pass) in passes.iter().enumerate() {
        for access in pass.reads.iter().chain(pass.writes.iter()) {
            let i = access.image.0 as usize;
            if i < persistent_count {
                continue;
            }
            if images[i].origin == ImageOrigin::External {
                continue;
            }
            lifetimes[i] = Some(match lifetimes[i] {
                None => (pass_idx, pass_idx),
                Some((lo, hi)) => (lo.min(pass_idx), hi.max(pass_idx)),
            });
        }
    }

    lifetimes
}

fn assign_slots(
    lifetimes: &[Option<(usize, usize)>],
    images: &[ImageEntry],
) -> (Vec<Option<usize>>, usize) {
    let mut assignments = vec![None::<usize>; lifetimes.len()];
    let mut slot_end: Vec<(usize, vk::Format, vk::ImageAspectFlags)> = Vec::new();

    let mut order: Vec<usize> = lifetimes
        .iter()
        .enumerate()
        .filter_map(|(i, lt)| lt.map(|_| i))
        .collect();
    order.sort_by_key(|&i| lifetimes[i].expect("filtered to non-None above").0);

    for i in order {
        let (first, last) = lifetimes[i].expect("filtered to non-None above");
        let fmt = images[i].desc.format;
        let asp = images[i].aspect;
        let slot = slot_end
            .iter()
            .position(|&(end, f, a)| end < first && f == fmt && a == asp);
        let slot = match slot {
            Some(s) => {
                slot_end[s].0 = last;
                s
            }
            None => {
                slot_end.push((last, fmt, asp));
                slot_end.len() - 1
            }
        };
        assignments[i] = Some(slot);
    }

    (assignments, slot_end.len())
}

struct CachedSlot {
    handle: ImageHandle,
    desc: ImageDesc,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
}

impl CachedSlot {
    fn is_compatible(
        &self,
        desc: &ImageDesc,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> bool {
        self.desc.format == desc.format
            && self.desc.kind == desc.kind
            && self.desc.samples == desc.samples
            && self.desc.mip_levels >= desc.mip_levels
            && self.desc.extent.width >= desc.extent.width
            && self.desc.extent.height >= desc.extent.height
            && self.desc.extent.depth >= desc.extent.depth
            && self.aspect == aspect
            && self.usage.contains(usage)
    }
}

pub(crate) struct TransientCache {
    slots: Vec<Option<CachedSlot>>,
}

impl TransientCache {
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    pub fn allocate(
        &mut self,
        images: &mut [ImageEntry],
        passes: &[RecordedPass],
        persistent_count: usize,
        resources: &mut ResourcePool,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) -> Result<(), ResourceError> {
        let (assignments, slot_count) = {
            let lifetimes = compute_lifetimes(passes, images, persistent_count);
            assign_slots(&lifetimes, images)
        };

        while self.slots.len() < slot_count {
            self.slots.push(None);
        }

        let mut slot_specs: Vec<Option<(ImageDesc, vk::ImageUsageFlags, vk::ImageAspectFlags)>> =
            vec![None; slot_count];

        for (i, entry) in images.iter().enumerate().skip(persistent_count) {
            let Some(slot) = assignments[i] else { continue };
            let usage = entry.usage | vk::ImageUsageFlags::TRANSFER_DST;
            slot_specs[slot] = Some(match slot_specs[slot].take() {
                None => (entry.desc.clone(), usage, entry.aspect),
                Some((mut desc, prev_u, prev_a)) => {
                    desc.extent.width = desc.extent.width.max(entry.desc.extent.width);
                    desc.extent.height = desc.extent.height.max(entry.desc.extent.height);
                    desc.extent.depth = desc.extent.depth.max(entry.desc.extent.depth);
                    desc.mip_levels = desc.mip_levels.max(entry.desc.mip_levels);
                    (desc, prev_u | usage, prev_a)
                }
            });
        }

        for (slot, spec) in slot_specs.iter().enumerate() {
            let Some((desc, usage, aspect)) = spec else {
                continue;
            };

            let compatible = self.slots[slot]
                .as_ref()
                .map(|s| s.is_compatible(desc, *usage, *aspect))
                .unwrap_or(false);

            if !compatible {
                if let Some(old) = self.slots[slot].take() {
                    resources.destroy_image(device, allocator, old.handle);
                }
                let handle = resources.create_image(device, allocator, desc, *usage, *aspect)?;
                self.slots[slot] = Some(CachedSlot {
                    handle,
                    desc: desc.clone(),
                    usage: *usage,
                    aspect: *aspect,
                });
            }
        }

        for (i, entry) in images.iter_mut().enumerate().skip(persistent_count) {
            let Some(slot) = assignments[i] else { continue };
            entry.handle = Some(
                self.slots[slot]
                    .as_ref()
                    .expect("slot assigned above")
                    .handle,
            );
        }

        Ok(())
    }

    pub fn clear(
        &mut self,
        resources: &mut ResourcePool,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) {
        for cached in self.slots.drain(..).flatten() {
            resources.destroy_image(device, allocator, cached.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::*;
    use crate::graph::access::LoadOp;
    use crate::graph::image::{Image, ImageEntry};
    use crate::graph::pass::{PassAccess, RecordedPass};

    fn img_access(id: u32) -> PassAccess {
        PassAccess {
            image: Image(id),
            layout: vk::ImageLayout::UNDEFINED,
            stage: vk::PipelineStageFlags2::empty(),
            access: vk::AccessFlags2::empty(),
            is_color: false,
            is_depth: false,
            load_op: LoadOp::Auto,
            layer: None,
            clear_color: None,
        }
    }

    fn make_pass(name: &'static str, write_ids: &[u32], read_ids: &[u32]) -> RecordedPass {
        RecordedPass {
            name,
            reads: read_ids.iter().copied().map(img_access).collect(),
            writes: write_ids.iter().copied().map(img_access).collect(),
            buffer_reads: vec![],
            buffer_writes: vec![],
            view_mask: 0,
            execute: Box::new(|_, _| {}),
        }
    }

    fn transient_entry() -> ImageEntry {
        ImageEntry::transient(ImageDesc::default())
    }

    fn persistent_entry() -> ImageEntry {
        ImageEntry::persistent(ImageDesc::default())
    }

    fn transient_entry_fmt(format: vk::Format) -> ImageEntry {
        let mut desc = ImageDesc::default();
        desc.format = format;
        ImageEntry::transient(desc)
    }

    fn external_entry() -> ImageEntry {
        ImageEntry::external(
            vk::Image::null(),
            vk::ImageView::null(),
            vk::Extent2D { width: 1, height: 1 },
        )
    }

    #[test]
    fn lifetimes_basic() {
        let images = vec![
            persistent_entry(),
            transient_entry(),
            transient_entry(),
            transient_entry(),
        ];
        let passes = vec![
            make_pass("geo", &[1, 2, 3], &[]),
            make_pass("light", &[], &[1, 2, 3]),
        ];
        let lt = compute_lifetimes(&passes, &images, 1);
        assert_eq!(lt[0], None);
        assert_eq!(lt[1], Some((0, 1)));
        assert_eq!(lt[2], Some((0, 1)));
        assert_eq!(lt[3], Some((0, 1)));
    }

    #[test]
    fn lifetimes_skips_persistent() {
        let images = vec![persistent_entry(), transient_entry()];
        let passes = vec![make_pass("p", &[0, 1], &[])];
        let lt = compute_lifetimes(&passes, &images, 1);
        assert_eq!(lt[0], None);
        assert_eq!(lt[1], Some((0, 0)));
    }

    #[test]
    fn lifetimes_skips_external() {
        let images = vec![persistent_entry(), external_entry()];
        let passes = vec![make_pass("p", &[1], &[])];
        let lt = compute_lifetimes(&passes, &images, 0);
        assert_eq!(lt[1], None);
    }

    #[test]
    fn lifetimes_unused_transient() {
        let images = vec![persistent_entry(), transient_entry(), transient_entry()];
        let passes = vec![make_pass("p", &[1], &[])];
        let lt = compute_lifetimes(&passes, &images, 1);
        assert_eq!(lt[1], Some((0, 0)));
        assert_eq!(lt[2], None);
    }

    #[test]
    fn slots_overlapping_separate() {
        let images = vec![persistent_entry(), transient_entry(), transient_entry()];
        let lifetimes = vec![None, Some((0, 1)), Some((0, 1))];
        let (assignments, count) = assign_slots(&lifetimes, &images);
        assert_eq!(count, 2);
        assert_ne!(assignments[1], assignments[2]);
    }

    #[test]
    fn slots_non_overlapping_share() {
        let images = vec![persistent_entry(), transient_entry(), transient_entry()];
        let lifetimes = vec![None, Some((0, 0)), Some((1, 1))];
        let (assignments, count) = assign_slots(&lifetimes, &images);
        assert_eq!(count, 1);
        assert_eq!(assignments[1], Some(0));
        assert_eq!(assignments[2], Some(0));
    }

    #[test]
    fn slots_chain_reuses() {
        let images = vec![
            persistent_entry(),
            transient_entry(),
            transient_entry(),
            transient_entry(),
        ];
        let lifetimes = vec![None, Some((0, 0)), Some((1, 1)), Some((2, 2))];
        let (assignments, count) = assign_slots(&lifetimes, &images);
        assert_eq!(count, 1);
        assert_eq!(assignments[1], Some(0));
        assert_eq!(assignments[2], Some(0));
        assert_eq!(assignments[3], Some(0));
    }

    #[test]
    fn slots_incompatible_formats_separate() {
        let images = vec![
            persistent_entry(),
            transient_entry_fmt(vk::Format::D32_SFLOAT),
            transient_entry_fmt(vk::Format::R16G16B16A16_SFLOAT),
        ];
        let lifetimes = vec![None, Some((0, 0)), Some((1, 1))];
        let (assignments, count) = assign_slots(&lifetimes, &images);
        assert_eq!(count, 2);
        assert_ne!(assignments[1], assignments[2]);
    }
}
