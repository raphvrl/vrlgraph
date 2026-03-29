use ash::vk;

pub(super) const MAX_TIMESTAMP_PASSES: u32 = 64;

#[derive(Debug, Clone)]
pub struct PassTiming {
    pub name: &'static str,
    pub gpu_ns: u64,
}

pub(crate) struct TimestampQueryPool {
    pool: vk::QueryPool,
    device: ash::Device,
}

impl TimestampQueryPool {
    pub fn new(device: &ash::Device) -> Result<Self, vk::Result> {
        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(MAX_TIMESTAMP_PASSES * 2);

        let pool = unsafe { device.create_query_pool(&create_info, None)? };

        Ok(Self {
            pool,
            device: device.clone(),
        })
    }

    pub fn raw(&self) -> vk::QueryPool {
        self.pool
    }
}

impl Drop for TimestampQueryPool {
    fn drop(&mut self) {
        unsafe { self.device.destroy_query_pool(self.pool, None) };
    }
}
