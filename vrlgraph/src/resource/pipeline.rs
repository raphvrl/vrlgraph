use ash::vk;

#[cfg(debug_assertions)]
use crate::graph::pipeline::validate::ReflectedPushConstants;

pub struct GpuPipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    #[cfg(debug_assertions)]
    pub(crate) reflected_pc: Option<ReflectedPushConstants>,
}

impl GpuPipeline {
    pub(super) fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            // Pipeline layout is shared (owned by BindlessDescriptorTable) — do not destroy.
        }
    }
}
