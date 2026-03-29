use ash::vk;

pub struct Queue {
    raw: vk::Queue,
}

impl Queue {
    pub(crate) fn new(device: &ash::Device, family_index: u32) -> Self {
        let raw = unsafe { device.get_device_queue(family_index, 0) };
        Self { raw }
    }

    pub fn raw(&self) -> vk::Queue {
        self.raw
    }
}
