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

fn assign_slots(lifetimes: &[Option<(usize, usize)>]) -> (Vec<Option<usize>>, usize) {
    let mut assignments = vec![None::<usize>; lifetimes.len()];
    let mut slot_end: Vec<usize> = Vec::new();

    let mut order: Vec<usize> = lifetimes
        .iter()
        .enumerate()
        .filter_map(|(i, lt)| lt.map(|_| i))
        .collect();
    order.sort_by_key(|&i| lifetimes[i].unwrap().0);

    for i in order {
        let (first, last) = lifetimes[i].unwrap();
        let slot = slot_end.iter().position(|&end| end < first);
        let slot = match slot {
            Some(s) => {
                slot_end[s] = last;
                s
            }
            None => {
                slot_end.push(last);
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
            assign_slots(&lifetimes)
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
                Some((desc, prev_u, prev_a)) => (desc, prev_u | usage, prev_a),
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
            entry.handle = Some(self.slots[slot].as_ref().unwrap().handle);
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
