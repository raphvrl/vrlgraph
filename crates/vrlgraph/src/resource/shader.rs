use std::ffi::CString;

use ash::vk;

pub(crate) struct GpuShaderModule {
    pub(crate) module: vk::ShaderModule,
    pub(crate) entry: CString,
}
