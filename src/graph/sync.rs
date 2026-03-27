use ash::vk;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SyncError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

struct PerFrameData {
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
}

pub struct FrameSync {
    frames: Vec<PerFrameData>,
    render_finished: Vec<vk::Semaphore>,
    device: ash::Device,
}

impl FrameSync {
    pub fn new(
        device: &ash::Device,
        frames_in_flight: usize,
        image_count: usize,
    ) -> Result<Self, SyncError> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let frames = (0..frames_in_flight)
            .map(|_| unsafe {
                Ok(PerFrameData {
                    image_available: device.create_semaphore(&semaphore_info, None)?,
                    in_flight: device.create_fence(&fence_info, None)?,
                })
            })
            .collect::<Result<Vec<_>, vk::Result>>()?;

        let render_finished = (0..image_count)
            .map(|_| unsafe { device.create_semaphore(&semaphore_info, None) })
            .collect::<Result<Vec<_>, vk::Result>>()?;

        Ok(Self {
            frames,
            render_finished,
            device: device.clone(),
        })
    }

    pub fn wait(&self, frame_index: usize) -> Result<(), SyncError> {
        unsafe {
            self.device
                .wait_for_fences(&[self.frames[frame_index].in_flight], true, u64::MAX)?;
        }
        Ok(())
    }

    pub fn reset(&self, frame_index: usize) -> Result<(), SyncError> {
        unsafe {
            self.device
                .reset_fences(&[self.frames[frame_index].in_flight])?
        };
        Ok(())
    }

    pub fn image_available(&self, frame_index: usize) -> vk::Semaphore {
        self.frames[frame_index].image_available
    }

    pub fn render_finished(&self, image_index: usize) -> vk::Semaphore {
        self.render_finished[image_index]
    }

    pub fn in_flight_fence(&self, frame_index: usize) -> vk::Fence {
        self.frames[frame_index].in_flight
    }

    #[allow(dead_code)]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

impl Drop for FrameSync {
    fn drop(&mut self) {
        unsafe {
            for frame in &self.frames {
                self.device.destroy_semaphore(frame.image_available, None);
                self.device.destroy_fence(frame.in_flight, None);
            }
            for &sem in &self.render_finished {
                self.device.destroy_semaphore(sem, None);
            }
        }
    }
}
