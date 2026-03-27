use ash::{ext::debug_utils, vk};
use std::ffi::{CStr, CString};
use thiserror::Error;
use tracing::{error, info, warn};

#[derive(Debug, Error)]
pub enum InstanceError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
    #[error("Missing validation layer: {0}")]
    MissingLayer(String),
    #[error("Failed to load Vulkan entry")]
    EntryLoad(#[from] ash::LoadingError),
}

pub struct Instance {
    entry: ash::Entry,
    inner: ash::Instance,
    debug: Option<DebugUtils>,
}

struct DebugUtils {
    instance: debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl Instance {
    pub fn new(validation: bool) -> Result<Self, InstanceError> {
        let entry = unsafe { ash::Entry::load()? };

        if validation {
            Self::check_layer_support(&entry, Self::required_layers())?;
        }

        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);

        let layers = Self::required_layers();
        let layer_ptrs: Vec<*const i8> = if validation {
            layers.iter().map(|s| s.as_ptr()).collect()
        } else {
            vec![]
        };

        let extensions = Self::required_extensions(validation);
        let extension_ptrs: Vec<*const i8> = extensions.iter().map(|s| s.as_ptr()).collect();

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_ptrs)
            .enabled_extension_names(&extension_ptrs);

        let inner = unsafe { entry.create_instance(&create_info, None)? };

        let debug_utils = if validation {
            Some(Self::create_debug_messenger(&entry, &inner)?)
        } else {
            None
        };

        info!("Vulkan instance created (validation={})", validation);

        Ok(Self {
            entry,
            inner,
            debug: debug_utils,
        })
    }

    pub fn raw(&self) -> &ash::Instance {
        &self.inner
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    fn required_layers() -> Vec<CString> {
        vec![CString::new("VK_LAYER_KHRONOS_validation").unwrap()]
    }

    fn required_extensions(debug: bool) -> Vec<CString> {
        let mut exts = vec![
            CString::new("VK_KHR_surface").unwrap(),
            #[cfg(target_os = "windows")]
            CString::new("VK_KHR_win32_surface").unwrap(),
            #[cfg(target_os = "linux")]
            CString::new("VK_KHR_xcb_surface").unwrap(),
            #[cfg(target_os = "macos")]
            CString::new("VK_EXT_metal_surface").unwrap(),
        ];

        if debug {
            exts.push(CString::new("VK_EXT_debug_utils").unwrap());
        }

        exts
    }

    fn check_layer_support(entry: &ash::Entry, layers: Vec<CString>) -> Result<(), InstanceError> {
        let available = unsafe { entry.enumerate_instance_layer_properties()? };

        for layer in layers {
            let found = available.iter().any(|p| {
                let name = unsafe { CStr::from_ptr(p.layer_name.as_ptr()) };
                name == layer.as_c_str()
            });

            if !found {
                return Err(InstanceError::MissingLayer(
                    layer.to_string_lossy().into_owned(),
                ));
            }
        }

        Ok(())
    }

    fn create_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<DebugUtils, InstanceError> {
        let loader = debug_utils::Instance::new(entry, instance);

        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback));

        let messenger = unsafe { loader.create_debug_utils_messenger(&create_info, None)? };

        Ok(DebugUtils {
            instance: loader,
            messenger,
        })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            if let Some(debug) = &self.debug {
                debug
                    .instance
                    .destroy_debug_utils_messenger(debug.messenger, None);
            }
            self.inner.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();

    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => error!("[Vulkan] {}", message),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => warn!("[Vulkan] {}", message),
        _ => tracing::debug!("[Vulkan] {}", message),
    }

    vk::FALSE
}
