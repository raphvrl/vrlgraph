use std::ffi::CString;

use ash::vk;

use super::Cmd;

impl<'a> Cmd<'a> {
    /// Opens a debug group visible in RenderDoc and Nsight. No-op if the
    /// debug utils extension is not enabled.
    pub fn begin_debug_group(&self, name: &str, color: [f32; 4]) {
        let Some(du) = &self.debug_utils else { return };
        let name_c = CString::new(name).unwrap_or_else(|_| c"<invalid>".to_owned());
        let label = vk::DebugUtilsLabelEXT::default()
            .label_name(&name_c)
            .color(color);
        unsafe { du.cmd_begin_debug_utils_label(self.raw, &label) };
    }

    /// Closes the current debug group opened with [`begin_debug_group`](Cmd::begin_debug_group).
    pub fn end_debug_group(&self) {
        let Some(du) = &self.debug_utils else { return };
        unsafe { du.cmd_end_debug_utils_label(self.raw) };
    }

    /// Inserts a single debug label at the current command position.
    pub fn insert_debug_label(&self, name: &str, color: [f32; 4]) {
        let Some(du) = &self.debug_utils else { return };
        let name_c = CString::new(name).unwrap_or_else(|_| c"<invalid>".to_owned());
        let label = vk::DebugUtilsLabelEXT::default()
            .label_name(&name_c)
            .color(color);
        unsafe { du.cmd_insert_debug_utils_label(self.raw, &label) };
    }
}
