use ash::{khr, vk};
use thiserror::Error;
use tracing::info;

use super::surface::{Surface, SurfaceError};

#[derive(Debug, Error)]
pub enum SwapchainError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
    #[error("Surface error: {0}")]
    Surface(#[from] SurfaceError),
    #[error("No suitable surface format found")]
    NoSuitableFormat,
}

pub struct SwapchainConfig {
    pub physical_device: vk::PhysicalDevice,
    pub graphics_family: u32,
    pub present_family: u32,
    pub window_size: (u32, u32),
    pub preferred_present: vk::PresentModeKHR,
}

pub struct Swapchain {
    loader: khr::swapchain::Device,
    raw: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    format: vk::Format,
    extent: vk::Extent2D,
    device: ash::Device,
}

impl Swapchain {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        surface: &Surface,
        config: SwapchainConfig,
    ) -> Result<Self, SwapchainError> {
        let loader = khr::swapchain::Device::new(instance, device);

        let (raw, images, format, extent) =
            Self::create_internal(&loader, surface, &config, vk::SwapchainKHR::null())?;

        let image_views = Self::create_image_views(device, &images, format)?;

        info!(
            "Swapchain created: {}x{} ({:?}), {} images",
            extent.width,
            extent.height,
            format,
            images.len()
        );

        Ok(Self {
            loader,
            raw,
            images,
            image_views,
            format,
            extent,
            device: device.clone(),
        })
    }

    pub fn recreate(
        &mut self,
        surface: &Surface,
        config: SwapchainConfig,
    ) -> Result<(), SwapchainError> {
        if config.window_size.0 == 0 || config.window_size.1 == 0 {
            return Ok(());
        }

        self.destroy_image_views();
        let old = self.raw;

        let (raw, images, format, extent) =
            Self::create_internal(&self.loader, surface, &config, old)?;

        unsafe { self.loader.destroy_swapchain(old, None) };

        let image_views = Self::create_image_views(&self.device, &images, format)?;

        info!(
            "Swapchain recreated: {}x{} ({:?}), {} images",
            extent.width,
            extent.height,
            format,
            images.len()
        );

        self.raw = raw;
        self.images = images;
        self.image_views = image_views;
        self.format = format;
        self.extent = extent;

        Ok(())
    }

    pub fn acquire_next_image(&self, semaphore: vk::Semaphore) -> Result<(u32, bool), vk::Result> {
        unsafe {
            self.loader
                .acquire_next_image(self.raw, u64::MAX, semaphore, vk::Fence::null())
        }
    }

    pub fn present(
        &self,
        queue: vk::Queue,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
    ) -> Result<bool, vk::Result> {
        let swapchains = [self.raw];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe { self.loader.queue_present(queue, &present_info) }
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }
    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    fn create_internal(
        loader: &khr::swapchain::Device,
        surface: &Surface,
        config: &SwapchainConfig,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D), SwapchainError> {
        let SwapchainConfig {
            physical_device,
            graphics_family,
            present_family,
            window_size,
            preferred_present,
        } = config;

        let capabilities = surface.capabilities(*physical_device)?;
        let format = Self::choose_format(&surface.formats(*physical_device)?)?;
        let present_mode = Self::choose_present_mode(
            &surface.present_modes(*physical_device)?,
            *preferred_present,
        );
        let extent = Self::choose_extent(&capabilities, *window_size);

        let image_count = {
            let desired = capabilities.min_image_count + 1;
            match capabilities.max_image_count {
                0 => desired,
                max => desired.min(max),
            }
        };

        let (sharing_mode, queue_families): (vk::SharingMode, Vec<u32>) =
            if graphics_family != present_family {
                (
                    vk::SharingMode::CONCURRENT,
                    vec![*graphics_family, *present_family],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, vec![])
            };

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.raw())
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(&queue_families)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let raw = unsafe { loader.create_swapchain(&create_info, None)? };
        let images = unsafe { loader.get_swapchain_images(raw)? };

        Ok((raw, images, format.format, extent))
    }

    fn choose_format(
        formats: &[vk::SurfaceFormatKHR],
    ) -> Result<vk::SurfaceFormatKHR, SwapchainError> {
        formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first())
            .copied()
            .ok_or(SwapchainError::NoSuitableFormat)
    }

    fn choose_present_mode(
        modes: &[vk::PresentModeKHR],
        preferred: vk::PresentModeKHR,
    ) -> vk::PresentModeKHR {
        if modes.contains(&preferred) {
            preferred
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    fn choose_extent(
        caps: &vk::SurfaceCapabilitiesKHR,
        (width, height): (u32, u32),
    ) -> vk::Extent2D {
        if caps.current_extent.width != u32::MAX {
            return caps.current_extent;
        }

        vk::Extent2D {
            width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    }

    fn create_image_views(
        device: &ash::Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Result<Vec<vk::ImageView>, vk::Result> {
        images
            .iter()
            .map(|&image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                unsafe { device.create_image_view(&create_info, None) }
            })
            .collect()
    }

    fn destroy_image_views(&mut self) {
        for &view in &self.image_views {
            unsafe { self.device.destroy_image_view(view, None) };
        }
        self.image_views.clear();
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.destroy_image_views();
        unsafe { self.loader.destroy_swapchain(self.raw, None) };
    }
}
