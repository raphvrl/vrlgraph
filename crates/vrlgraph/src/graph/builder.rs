use std::path::PathBuf;

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use crate::device::GpuDevice;

use super::{Graph, GraphError};

#[derive(Clone, Copy, Debug, Default)]
pub enum PresentMode {
    #[default]
    Fifo,

    Mailbox,

    Immediate,
}

impl PresentMode {
    pub(crate) fn to_vk(self) -> vk::PresentModeKHR {
        match self {
            Self::Fifo => vk::PresentModeKHR::FIFO,
            Self::Mailbox => vk::PresentModeKHR::MAILBOX,
            Self::Immediate => vk::PresentModeKHR::IMMEDIATE,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum GpuPreference {
    #[default]
    HighPerformance,

    LowPower,
}

pub struct GraphBuilder {
    display_handle: Option<RawDisplayHandle>,
    window_handle: Option<RawWindowHandle>,
    window_handle_error: bool,
    window_size: (u32, u32),
    validation: bool,
    present_mode: PresentMode,
    gpu_preference: GpuPreference,
    frames_in_flight: usize,
    pipeline_cache_path: Option<PathBuf>,
    srgb: bool,
}

impl GraphBuilder {
    pub(super) fn new() -> Self {
        Self {
            display_handle: None,
            window_handle: None,
            window_handle_error: false,
            window_size: (800, 600),
            validation: false,
            present_mode: PresentMode::default(),
            gpu_preference: GpuPreference::default(),
            frames_in_flight: 2,
            pipeline_cache_path: None,
            srgb: true,
        }
    }

    pub fn window(mut self, w: &(impl HasWindowHandle + HasDisplayHandle)) -> Self {
        match (w.display_handle(), w.window_handle()) {
            (Ok(d), Ok(wh)) => {
                self.display_handle = Some(d.as_raw());
                self.window_handle = Some(wh.as_raw());
            }
            _ => self.window_handle_error = true,
        }
        self
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.window_size = (width, height);
        self
    }

    pub fn validation(mut self, enabled: bool) -> Self {
        self.validation = enabled;
        self
    }

    pub fn present_mode(mut self, mode: PresentMode) -> Self {
        self.present_mode = mode;
        self
    }

    pub fn gpu(mut self, preference: GpuPreference) -> Self {
        self.gpu_preference = preference;
        self
    }

    pub fn frames_in_flight(mut self, count: usize) -> Self {
        self.frames_in_flight = count;
        self
    }

    pub fn pipeline_cache_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.pipeline_cache_path = Some(path.into());
        self
    }

    pub fn srgb(mut self, enabled: bool) -> Self {
        self.srgb = enabled;
        self
    }

    pub fn build(self) -> Result<Graph, GraphError> {
        if self.window_handle_error {
            return Err(GraphError::WindowHandle);
        }
        let display = self
            .display_handle
            .expect("GraphBuilder: .window() is required");
        let window = self
            .window_handle
            .expect("GraphBuilder: .window() is required");

        let device = GpuDevice::new(
            self.validation,
            display,
            window,
            self.window_size,
            self.present_mode.to_vk(),
            matches!(self.gpu_preference, GpuPreference::LowPower),
            self.srgb,
        )?;

        Graph::from_device(
            device,
            self.frames_in_flight,
            self.present_mode.to_vk(),
            self.pipeline_cache_path,
        )
    }
}
