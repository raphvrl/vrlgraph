use ash::vk;

use super::FrameBuilder;
use crate::graph::image::{Image, ImageEntry};
use crate::graph::query::PassTiming;
use crate::graph::{Graph, GraphError};

impl Graph {
    pub fn begin_frame(&mut self) -> Result<FrameBuilder<'_>, GraphError> {
        debug_assert!(
            !self.frame_active,
            "begin_frame called while a frame is already active (missing end_frame?)"
        );

        let resized = if let Some((w, h)) = self.pending_resize.take() {
            self.apply_resize(w, h)?
        } else {
            false
        };

        let idx = self.current;
        self.sync.wait(idx)?;
        self.frames[idx]
            .deferred_frees
            .drain_into(&mut self.bindless);

        {
            let device = self.device.ash_device().clone();
            let alloc = self.device.allocator_mut();
            self.transfer
                .cleanup_staging(&mut self.resources, &device, alloc);
        }

        self.cleanup_frame();

        self.timestamps.last_timings.clear();
        if self.timestamps.is_enabled() && self.timestamps.written[idx] {
            let n = self.timestamps.names[idx].len() as u32;
            if n > 0 {
                let mut results = vec![0u64; (n * 2) as usize];
                let ok = unsafe {
                    self.device.ash_device().get_query_pool_results(
                        self.timestamps.pools[idx].raw(),
                        0,
                        &mut results,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                };
                if ok.is_ok() {
                    let period = self.timestamps.period;
                    for (i, &name) in self.timestamps.names[idx].iter().enumerate() {
                        let begin = results[i * 2];
                        let end = results[i * 2 + 1];
                        if end >= begin {
                            let gpu_ns = ((end - begin) as f64 * period) as u64;
                            self.timestamps
                                .last_timings
                                .push(PassTiming { name, gpu_ns });
                            tracing::debug!(gpu_ns, pass = name, "gpu_pass_timing");
                        }
                    }
                }
            }
            self.timestamps.written[idx] = false;
        }

        let image_index = match self
            .device
            .swapchain()
            .acquire_next_image(self.sync.image_available(idx))
        {
            Ok((i, suboptimal)) => {
                if suboptimal {
                    return Err(GraphError::SwapchainOutOfDate);
                }
                i
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Err(GraphError::SwapchainOutOfDate),
            Err(e) => return Err(GraphError::Vulkan(e)),
        };

        self.sync.reset(idx)?;

        self.frame_active = true;
        self.image_index = image_index;
        self.frame_index = idx;

        let extent = self.device.swapchain().extent();
        debug_assert!(
            (image_index as usize) < self.device.swapchain().image_count(),
            "swapchain image_index {} out of range (count = {})",
            image_index,
            self.device.swapchain().image_count(),
        );
        let raw_img = self.device.swapchain().images()[image_index as usize];
        let raw_view = self.device.swapchain().image_views()[image_index as usize];

        let backbuffer = Image(self.images.len() as u32);
        self.images
            .push(ImageEntry::external(raw_img, raw_view, extent));
        self.sc_graph_image = Some(backbuffer);

        Ok(FrameBuilder {
            graph: self,
            backbuffer,
            extent,
            index: idx as u32,
            resized,
            pending_passes: Vec::new(),
        })
    }
}
