mod instance;
mod queue;
pub(crate) mod surface;
pub(crate) mod swapchain;

use std::ffi::CStr;
use std::ops::Deref;

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use instance::{Instance, InstanceError};
use queue::Queue;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use surface::{Surface, SurfaceError};
use swapchain::{Swapchain, SwapchainConfig, SwapchainError};
use thiserror::Error;
use tracing::info;

const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[
    ash::khr::swapchain::NAME,
    ash::ext::extended_dynamic_state3::NAME,
];

#[derive(Debug, Error)]
pub enum DeviceError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
    #[error("Instance error: {0}")]
    Instance(#[from] InstanceError),
    #[error("Surface error: {0}")]
    Surface(#[from] SurfaceError),
    #[error("Swapchain error: {0}")]
    Swapchain(#[from] SwapchainError),
    #[error("No suitable GPU found")]
    NoSuitableDevice,
    #[error("Allocator error: {0}")]
    Allocator(#[from] gpu_allocator::AllocationError),
}

struct OwnedDevice(ash::Device);

impl Deref for OwnedDevice {
    type Target = ash::Device;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for OwnedDevice {
    fn drop(&mut self) {
        unsafe { self.0.destroy_device(None) };
    }
}

pub struct GpuDevice {
    swapchain: Swapchain,
    surface: Surface,
    queue: Queue,
    graphics_family: u32,
    present_family: u32,
    properties: vk::PhysicalDeviceProperties,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    physical_device: vk::PhysicalDevice,
    prefer_srgb: bool,
    allocator: Allocator,
    ext_dynamic_state3: ash::ext::extended_dynamic_state3::Device,
    debug_utils: Option<ash::ext::debug_utils::Device>,
    device: OwnedDevice,
    #[allow(dead_code)]
    instance: Instance,
}

impl GpuDevice {
    pub(crate) fn new(
        validation: bool,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        window_size: (u32, u32),
        preferred_present: vk::PresentModeKHR,
        prefer_integrated: bool,
        prefer_srgb: bool,
    ) -> Result<Self, DeviceError> {
        let instance = Instance::new(validation, display_handle)?;

        let surface = Surface::new(
            instance.entry(),
            instance.raw(),
            display_handle,
            window_handle,
        )?;

        let (physical_device, graphics_family, present_family) =
            Self::pick_physical_device(&instance, &surface, prefer_integrated)?;

        let properties = unsafe {
            instance
                .raw()
                .get_physical_device_properties(physical_device)
        };
        let memory_properties = unsafe {
            instance
                .raw()
                .get_physical_device_memory_properties(physical_device)
        };

        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
        info!("Selected GPU: {:?}", name);

        let device = Self::create_logical_device(
            &instance,
            physical_device,
            graphics_family,
            present_family,
        )?;

        let queue = Queue::new(&device, graphics_family);

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.raw().clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        let swapchain = Swapchain::new(
            instance.raw(),
            &device,
            &surface,
            SwapchainConfig {
                physical_device,
                graphics_family,
                present_family,
                window_size,
                preferred_present,
                prefer_srgb,
            },
        )?;

        let ext_dynamic_state3 =
            ash::ext::extended_dynamic_state3::Device::new(instance.raw(), &device);
        let debug_utils = if validation {
            Some(ash::ext::debug_utils::Device::new(instance.raw(), &device))
        } else {
            None
        };

        Ok(Self {
            swapchain,
            surface,
            graphics_family,
            present_family,
            queue,
            properties,
            memory_properties,
            physical_device,
            prefer_srgb,
            allocator,
            ext_dynamic_state3,
            debug_utils,
            device: OwnedDevice(device),
            instance,
        })
    }

    pub(crate) fn recreate_swapchain(
        &mut self,
        window_size: (u32, u32),
        preferred_present: vk::PresentModeKHR,
    ) -> Result<(), DeviceError> {
        unsafe { self.device.device_wait_idle()? };

        self.swapchain
            .recreate(
                &self.surface,
                SwapchainConfig {
                    physical_device: self.physical_device,
                    graphics_family: self.graphics_family,
                    present_family: self.present_family,
                    window_size,
                    preferred_present,
                    prefer_srgb: self.prefer_srgb,
                },
            )
            .map_err(Into::into)
    }

    pub fn ash_device(&self) -> &ash::Device {
        &self.device.0
    }
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }
    pub(crate) fn queue(&self) -> &Queue {
        &self.queue
    }
    pub(crate) fn swapchain(&self) -> &Swapchain {
        &self.swapchain
    }
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.properties
    }
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }
    pub fn graphics_family(&self) -> u32 {
        self.graphics_family
    }
    pub fn present_family(&self) -> u32 {
        self.present_family
    }
    pub(crate) fn allocator_mut(&mut self) -> &mut gpu_allocator::vulkan::Allocator {
        &mut self.allocator
    }

    pub(crate) fn ext_dynamic_state3(&self) -> &ash::ext::extended_dynamic_state3::Device {
        &self.ext_dynamic_state3
    }

    pub fn debug_utils(&self) -> Option<&ash::ext::debug_utils::Device> {
        self.debug_utils.as_ref()
    }

    fn pick_physical_device(
        instance: &Instance,
        surface: &Surface,
        prefer_integrated: bool,
    ) -> Result<(vk::PhysicalDevice, u32, u32), DeviceError> {
        let devices = unsafe { instance.raw().enumerate_physical_devices()? };

        devices
            .into_iter()
            .filter_map(|device| {
                Self::check_device_suitability(instance, surface, device)
                    .map(|(g, p)| (device, g, p))
            })
            .max_by_key(|(device, _, _)| Self::score_device(instance, *device, prefer_integrated))
            .ok_or(DeviceError::NoSuitableDevice)
    }

    fn check_device_suitability(
        instance: &Instance,
        surface: &Surface,
        device: vk::PhysicalDevice,
    ) -> Option<(u32, u32)> {
        if !Self::supports_required_extensions(instance, device) {
            return None;
        }
        Self::find_queue_families(instance, surface, device)
    }

    fn supports_required_extensions(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        let available = unsafe {
            instance
                .raw()
                .enumerate_device_extension_properties(device)
                .unwrap_or_default()
        };

        REQUIRED_DEVICE_EXTENSIONS.iter().all(|required| {
            available.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                name == *required
            })
        })
    }

    fn find_queue_families(
        instance: &Instance,
        surface: &Surface,
        device: vk::PhysicalDevice,
    ) -> Option<(u32, u32)> {
        let families = unsafe {
            instance
                .raw()
                .get_physical_device_queue_family_properties(device)
        };

        let required =
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;

        let graphics = families
            .iter()
            .position(|f| f.queue_flags.contains(required))
            .map(|i| i as u32)?;

        if surface.supports_present(device, graphics).unwrap_or(false) {
            return Some((graphics, graphics));
        }

        let present = (0..families.len() as u32)
            .find(|&i| surface.supports_present(device, i).unwrap_or(false))?;

        Some((graphics, present))
    }

    fn score_device(
        instance: &Instance,
        device: vk::PhysicalDevice,
        prefer_integrated: bool,
    ) -> u32 {
        let props = unsafe { instance.raw().get_physical_device_properties(device) };

        let (discrete, integrated) = if prefer_integrated {
            (100, 1000)
        } else {
            (1000, 100)
        };

        match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => discrete,
            vk::PhysicalDeviceType::INTEGRATED_GPU => integrated,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
            _ => 1,
        }
    }

    fn create_logical_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        graphics_family: u32,
        present_family: u32,
    ) -> Result<ash::Device, DeviceError> {
        let queue_priorities = [1.0f32];

        let mut queue_create_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(graphics_family)
                .queue_priorities(&queue_priorities),
        ];

        if present_family != graphics_family {
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(present_family)
                    .queue_priorities(&queue_priorities),
            );
        }

        let extension_ptrs: Vec<*const i8> = REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .map(|e| e.as_ptr())
            .collect();

        let mut dynamic_rendering =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

        let mut synchronization2 =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

        let mut extended_dynamic_state =
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default()
                .extended_dynamic_state(true);

        let mut extended_dynamic_state2 =
            vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT::default()
                .extended_dynamic_state2(true);

        let mut extended_dynamic_state3 =
            vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default()
                .extended_dynamic_state3_depth_clamp_enable(true)
                .extended_dynamic_state3_polygon_mode(true)
                .extended_dynamic_state3_color_blend_enable(true)
                .extended_dynamic_state3_color_blend_equation(true)
                .extended_dynamic_state3_color_write_mask(true);

        let mut multiview = vk::PhysicalDeviceMultiviewFeatures::default().multiview(true);

        let mut buffer_device_address =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);

        let mut descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .runtime_descriptor_array(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true);

        let core_features = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .depth_clamp(true);

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_ptrs)
            .enabled_features(&core_features)
            .push_next(&mut dynamic_rendering)
            .push_next(&mut synchronization2)
            .push_next(&mut extended_dynamic_state)
            .push_next(&mut extended_dynamic_state2)
            .push_next(&mut extended_dynamic_state3)
            .push_next(&mut multiview)
            .push_next(&mut buffer_device_address)
            .push_next(&mut descriptor_indexing);

        let device = unsafe {
            instance
                .raw()
                .create_device(physical_device, &create_info, None)?
        };

        Ok(device)
    }
}
