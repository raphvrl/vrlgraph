use ash::vk;

pub(super) const MAX_TIMESTAMP_PASSES: u32 = 64;

#[derive(Debug, Clone)]
pub struct PassTiming {
    pub name: &'static str,
    pub gpu_ns: u64,
}

pub(crate) struct TimestampState {
    pub pools: Vec<TimestampQueryPool>,
    pub names: Vec<Vec<&'static str>>,
    pub written: Vec<bool>,
    pub period: f64,
    pub last_timings: Vec<PassTiming>,
}

impl TimestampState {
    pub fn new(device: &ash::Device, frames_count: usize, period: f64) -> Result<Self, vk::Result> {
        let pools = if period > 0.0 {
            (0..frames_count)
                .map(|_| TimestampQueryPool::new(device))
                .collect::<Result<Vec<_>, vk::Result>>()?
        } else {
            tracing::warn!(
                "GPU timestamp queries not supported on this device — profiling disabled"
            );
            Vec::new()
        };
        Ok(Self {
            pools,
            names: vec![Vec::new(); frames_count],
            written: vec![false; frames_count],
            period,
            last_timings: Vec::new(),
        })
    }

    pub fn is_enabled(&self) -> bool {
        !self.pools.is_empty()
    }
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
