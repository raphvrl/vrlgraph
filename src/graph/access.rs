use ash::vk;

pub(crate) struct AccessInfo {
    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub usage: vk::ImageUsageFlags,
}

/// How a pass accesses an image.
///
/// This is the primary way you tell the graph what an image is used for in a
/// given pass. The graph maps each variant to the correct `VkImageLayout`,
/// pipeline stage, and access flags, and inserts barriers between passes that
/// use the same image with incompatible accesses.
///
/// Pass an `(image, Access)` tuple to [`PassSetup::read`](crate::graph::PassSetup::read) or [`PassSetup::write`](crate::graph::PassSetup::write).
#[derive(Clone, Copy, Debug)]
pub enum Access {
    /// Write to a color render target. Sets layout to `COLOR_ATTACHMENT_OPTIMAL`.
    ColorAttachment,
    /// Write to a depth attachment (depth only). Sets layout to `DEPTH_ATTACHMENT_OPTIMAL`.
    DepthAttachment,
    /// Read and write depth and stencil. Sets layout to `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`.
    DepthStencilAttachment,
    /// Read depth in a depth test without writing. Sets layout to
    /// `DEPTH_STENCIL_READ_ONLY_OPTIMAL`. Use this for shadow map sampling in a
    /// fragment shader with an active depth test.
    DepthRead,
    /// Sample the image in a fragment or vertex shader. Sets layout to
    /// `SHADER_READ_ONLY_OPTIMAL`. This is the standard access for textures and
    /// render targets consumed by a later pass.
    ShaderRead,
    /// Read from a compute shader via a storage image binding. Sets layout to `GENERAL`.
    ComputeRead,
    /// Write from a compute shader via a storage image binding. Sets layout to `GENERAL`.
    ComputeWrite,
    /// Source of a copy or blit operation.
    TransferSrc,
    /// Destination of a copy or blit operation.
    TransferDst,
}

impl Access {
    pub(crate) fn info(self) -> AccessInfo {
        use vk::{
            AccessFlags2 as A, ImageLayout as L, ImageUsageFlags as U, PipelineStageFlags2 as S,
        };
        match self {
            Self::ColorAttachment => AccessInfo {
                layout: L::COLOR_ATTACHMENT_OPTIMAL,
                stage: S::COLOR_ATTACHMENT_OUTPUT,
                access: A::COLOR_ATTACHMENT_WRITE,
                usage: U::COLOR_ATTACHMENT,
            },
            Self::DepthAttachment => AccessInfo {
                layout: L::DEPTH_ATTACHMENT_OPTIMAL,
                stage: S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                access: A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                usage: U::DEPTH_STENCIL_ATTACHMENT,
            },
            Self::DepthStencilAttachment => AccessInfo {
                layout: L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                stage: S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                access: A::DEPTH_STENCIL_ATTACHMENT_WRITE | A::DEPTH_STENCIL_ATTACHMENT_READ,
                usage: U::DEPTH_STENCIL_ATTACHMENT,
            },
            Self::DepthRead => AccessInfo {
                layout: L::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                stage: S::EARLY_FRAGMENT_TESTS,
                access: A::DEPTH_STENCIL_ATTACHMENT_READ,
                usage: U::SAMPLED,
            },
            Self::ShaderRead => AccessInfo {
                layout: L::SHADER_READ_ONLY_OPTIMAL,
                stage: S::FRAGMENT_SHADER,
                access: A::SHADER_READ,
                usage: U::SAMPLED,
            },
            Self::ComputeRead => AccessInfo {
                layout: L::GENERAL,
                stage: S::COMPUTE_SHADER,
                access: A::SHADER_READ,
                usage: U::STORAGE,
            },
            Self::ComputeWrite => AccessInfo {
                layout: L::GENERAL,
                stage: S::COMPUTE_SHADER,
                access: A::SHADER_WRITE,
                usage: U::STORAGE,
            },
            Self::TransferSrc => AccessInfo {
                layout: L::TRANSFER_SRC_OPTIMAL,
                stage: S::TRANSFER,
                access: A::TRANSFER_READ,
                usage: U::TRANSFER_SRC,
            },
            Self::TransferDst => AccessInfo {
                layout: L::TRANSFER_DST_OPTIMAL,
                stage: S::TRANSFER,
                access: A::TRANSFER_WRITE,
                usage: U::TRANSFER_DST,
            },
        }
    }

    pub(crate) fn layout(self) -> vk::ImageLayout {
        self.info().layout
    }
    pub(crate) fn stage(self) -> vk::PipelineStageFlags2 {
        self.info().stage
    }
    pub(crate) fn flags(self) -> vk::AccessFlags2 {
        self.info().access
    }
    pub(crate) fn usage_flags(self) -> vk::ImageUsageFlags {
        self.info().usage
    }

    pub(crate) fn is_color_attachment(self) -> bool {
        matches!(self, Self::ColorAttachment)
    }

    pub(crate) fn is_depth_attachment(self) -> bool {
        matches!(self, Self::DepthAttachment | Self::DepthStencilAttachment)
    }
}

/// Load operation applied to a render target at the start of a pass.
///
/// Used with [`WithLoadOp`](crate::graph::WithLoadOp) when declaring a write
/// on a color or depth attachment.
#[derive(Clone, Copy, Debug, Default)]
pub enum LoadOp {
    /// The graph decides: `Clear` on first write, `Load` on subsequent writes
    /// within the same frame. This is the default behavior when using a plain
    /// `(image, Access)` write declaration.
    #[default]
    Auto,
    /// Clear the attachment to its default value (black / 0.0 depth) at the
    /// start of the pass.
    Clear,
    /// Preserve the existing contents of the attachment.
    Load,
    /// The existing contents are undefined. Slightly faster than `Clear` when
    /// every pixel will be written by the pass.
    DontCare,
}

pub(crate) struct BufferAccessInfo {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

/// How a pass accesses a buffer.
///
/// The buffer equivalent of [`Access`]. Pass a `(buffer, BufferUsage)` tuple
/// to [`PassSetup::read`](crate::graph::PassSetup::read) or [`PassSetup::write`](crate::graph::PassSetup::write) to declare buffer dependencies
/// between passes.
#[derive(Clone, Copy, Debug)]
pub enum BufferUsage {
    /// Read as a uniform buffer (`layout(binding = N) uniform`) in vertex or
    /// fragment shaders.
    UniformRead,
    /// Read as a storage buffer in any shader stage.
    StorageRead,
    /// Written by a compute shader as a storage buffer.
    StorageWrite,
    /// Bound as a vertex buffer.
    VertexRead,
    /// Bound as an index buffer.
    IndexRead,
    /// Read as indirect draw or dispatch arguments.
    IndirectRead,
    /// Source of a buffer copy.
    TransferSrc,
    /// Destination of a buffer copy.
    TransferDst,
}

impl BufferUsage {
    pub(crate) fn info(self) -> BufferAccessInfo {
        use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
        match self {
            Self::UniformRead => BufferAccessInfo {
                stage: S::VERTEX_SHADER | S::FRAGMENT_SHADER,
                access: A::UNIFORM_READ,
            },
            Self::StorageRead => BufferAccessInfo {
                stage: S::VERTEX_SHADER | S::FRAGMENT_SHADER | S::COMPUTE_SHADER,
                access: A::SHADER_READ,
            },
            Self::StorageWrite => BufferAccessInfo {
                stage: S::COMPUTE_SHADER,
                access: A::SHADER_WRITE,
            },
            Self::VertexRead => BufferAccessInfo {
                stage: S::VERTEX_ATTRIBUTE_INPUT,
                access: A::VERTEX_ATTRIBUTE_READ,
            },
            Self::IndexRead => BufferAccessInfo {
                stage: S::INDEX_INPUT,
                access: A::INDEX_READ,
            },
            Self::IndirectRead => BufferAccessInfo {
                stage: S::DRAW_INDIRECT,
                access: A::INDIRECT_COMMAND_READ,
            },
            Self::TransferSrc => BufferAccessInfo {
                stage: S::TRANSFER,
                access: A::TRANSFER_READ,
            },
            Self::TransferDst => BufferAccessInfo {
                stage: S::TRANSFER,
                access: A::TRANSFER_WRITE,
            },
        }
    }

    pub(crate) fn stage(self) -> vk::PipelineStageFlags2 {
        self.info().stage
    }
    pub(crate) fn flags(self) -> vk::AccessFlags2 {
        self.info().access
    }
}
