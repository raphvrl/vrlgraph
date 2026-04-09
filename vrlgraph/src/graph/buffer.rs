use ash::vk;
use gpu_allocator::MemoryLocation;

use super::{Graph, GraphError};
use crate::resource::{Buffer, BufferDesc, StreamingBufferHandle};

pub struct HostBufferBuilder<'g> {
    graph: &'g mut Graph,
    label: String,
    usage: vk::BufferUsageFlags,
    size: Option<vk::DeviceSize>,
    data: Option<Vec<u8>>,
}

impl<'g> HostBufferBuilder<'g> {
    pub(super) fn new(
        graph: &'g mut Graph,
        label: String,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        Self {
            graph,
            label,
            usage,
            size: None,
            data: None,
        }
    }

    pub fn size(mut self, size: vk::DeviceSize) -> Self {
        self.size = Some(size);
        self
    }

    pub fn data<T: crate::ShaderType>(mut self, value: &T) -> Self {
        let padded = T::PADDED_SIZE;
        let mut buf = vec![0u8; padded];
        value.write_padded(&mut buf);
        self.size = Some(padded as vk::DeviceSize);
        self.data = Some(buf);
        self
    }

    pub fn build(self) -> Result<Buffer, GraphError> {
        let size = self
            .size
            .expect("HostBufferBuilder: call size() or data() before build()");
        let buf = self.graph.host_buffer(&self.label, size, self.usage)?;
        if let Some(bytes) = &self.data {
            self.graph
                .resources
                .get_buffer(buf.0)
                .expect("buffer just created")
                .write_bytes(bytes);
        }
        Ok(buf)
    }

    pub fn streaming(self) -> Result<StreamingBufferHandle, GraphError> {
        let size = self
            .size
            .expect("HostBufferBuilder: call size() or data() before streaming()");
        let usage = self.usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        Ok(self.graph.create_streaming_buffer(
            size,
            usage,
            MemoryLocation::CpuToGpu,
            &self.label,
        )?)
    }
}

pub struct GpuBufferBuilder<'g> {
    graph: &'g mut Graph,
    label: String,
    usage: vk::BufferUsageFlags,
    size: Option<vk::DeviceSize>,
    data: Option<Vec<u8>>,
    dynamic: bool,
}

impl<'g> GpuBufferBuilder<'g> {
    pub(super) fn new(
        graph: &'g mut Graph,
        label: String,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        Self {
            graph,
            label,
            usage,
            size: None,
            data: None,
            dynamic: false,
        }
    }

    pub fn size(mut self, size: vk::DeviceSize) -> Self {
        self.size = Some(size);
        self
    }

    pub fn data<T: bytemuck::Pod>(mut self, data: &[T]) -> Self {
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        self.size = Some(bytes.len() as vk::DeviceSize);
        self.data = Some(bytes.to_vec());
        self
    }

    pub fn dynamic(mut self) -> Self {
        self.dynamic = true;
        self
    }

    pub fn build(self) -> Result<Buffer, GraphError> {
        let size = self
            .size
            .expect("GpuBufferBuilder: call size() or data() before build()");

        if self.dynamic {
            let buf = self.graph.host_buffer(&self.label, size, self.usage)?;
            if let Some(bytes) = &self.data {
                self.graph
                    .resources
                    .get_buffer(buf.0)
                    .expect("buffer just created")
                    .write_bytes(bytes);
            }
            Ok(buf)
        } else {
            match &self.data {
                Some(bytes) => self.graph.upload_buffer_labeled(
                    bytes,
                    self.usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    &self.label,
                ),
                None => Ok(self.graph.create_buffer(&BufferDesc {
                    size,
                    usage: self.usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    location: MemoryLocation::GpuOnly,
                    label: self.label,
                })?),
            }
        }
    }
}

impl Graph {
    pub fn storage_buffer(&mut self, label: impl Into<String>) -> HostBufferBuilder<'_> {
        HostBufferBuilder::new(self, label.into(), vk::BufferUsageFlags::STORAGE_BUFFER)
    }

    pub fn uniform_buffer(&mut self, label: impl Into<String>) -> HostBufferBuilder<'_> {
        HostBufferBuilder::new(self, label.into(), vk::BufferUsageFlags::UNIFORM_BUFFER)
    }

    pub fn vertex_buffer(&mut self, label: impl Into<String>) -> GpuBufferBuilder<'_> {
        GpuBufferBuilder::new(self, label.into(), vk::BufferUsageFlags::VERTEX_BUFFER)
    }

    pub fn index_buffer(&mut self, label: impl Into<String>) -> GpuBufferBuilder<'_> {
        GpuBufferBuilder::new(self, label.into(), vk::BufferUsageFlags::INDEX_BUFFER)
    }
}
