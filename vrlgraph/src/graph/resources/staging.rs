use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::graph::cmd::{Cmd, CommandPool};
use crate::graph::{Graph, GraphError};
use crate::resource::{BufferDesc, BufferHandle};

impl Graph {
    pub(super) fn one_shot_submit(&mut self, f: impl FnOnce(&Cmd)) -> Result<(), GraphError> {
        let device = self.device.ash_device().clone();
        let pool = CommandPool::new(&device, self.device.graphics_family())?;
        let raw_cb = pool.reset_and_begin()?;
        let cmd = Cmd::new_one_shot(
            raw_cb,
            device.clone(),
            self.device.ext_dynamic_state3().clone(),
            None,
        );
        f(&cmd);
        let buffer = cmd.finish()?;
        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(buffer);
        let submit =
            vk::SubmitInfo2::default().command_buffer_infos(std::slice::from_ref(&cmd_info));
        unsafe {
            device.queue_submit2(self.device.queue().raw(), &[submit], vk::Fence::null())?;
            device.queue_wait_idle(self.device.queue().raw())?;
        }
        Ok(())
    }

    pub(super) fn create_staging(
        &mut self,
        data: &[u8],
        label: &str,
    ) -> Result<BufferHandle, GraphError> {
        let device = self.device.ash_device().clone();
        let handle = self.resources.create_buffer(
            &device,
            self.device.allocator_mut(),
            &BufferDesc {
                size: data.len() as vk::DeviceSize,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                location: MemoryLocation::CpuToGpu,
                label: label.to_string(),
            },
        )?;
        let buf = self
            .resources
            .get_buffer(handle)
            .expect("buffer just created");
        let ptr = buf.mapped_ptr().expect("staging buffer not host visible");
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
        Ok(handle)
    }

    pub(super) fn destroy_staging(&mut self, handle: BufferHandle) {
        let device = self.device.ash_device().clone();
        self.resources
            .destroy_buffer(&device, self.device.allocator_mut(), handle);
    }
}
