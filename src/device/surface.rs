use ash::{khr, vk};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Error)]
pub enum SurfaceError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

pub struct Surface {
    loader: khr::surface::Instance,
    raw: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<Self, SurfaceError> {
        let loader = khr::surface::Instance::new(entry, instance);

        let raw = unsafe {
            ash_window::create_surface(entry, instance, display_handle, window_handle, None)?
        };

        info!("Vulkan surface created");

        Ok(Self { loader, raw })
    }

    pub fn raw(&self) -> vk::SurfaceKHR {
        self.raw
    }

    pub fn capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<vk::SurfaceCapabilitiesKHR, SurfaceError> {
        unsafe {
            self.loader
                .get_physical_device_surface_capabilities(physical_device, self.raw)
                .map_err(Into::into)
        }
    }

    pub fn formats(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::SurfaceFormatKHR>, SurfaceError> {
        unsafe {
            self.loader
                .get_physical_device_surface_formats(physical_device, self.raw)
                .map_err(Into::into)
        }
    }

    pub fn present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::PresentModeKHR>, SurfaceError> {
        unsafe {
            self.loader
                .get_physical_device_surface_present_modes(physical_device, self.raw)
                .map_err(Into::into)
        }
    }

    pub fn supports_present(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
    ) -> Result<bool, SurfaceError> {
        unsafe {
            self.loader
                .get_physical_device_surface_support(physical_device, queue_family, self.raw)
                .map_err(Into::into)
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.raw, None) };
    }
}
