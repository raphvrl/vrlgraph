use ash::vk;

pub struct GpuPipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
}

impl GpuPipeline {
    pub(super) fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            // Pipeline layout is shared (owned by BindlessDescriptorTable) — do not destroy.
        }
    }
}
