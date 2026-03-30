use ash::vk;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CullMode(pub(crate) vk::CullModeFlags);

impl CullMode {
    pub const NONE: Self = Self(vk::CullModeFlags::NONE);
    pub const FRONT: Self = Self(vk::CullModeFlags::FRONT);
    pub const BACK: Self = Self(vk::CullModeFlags::BACK);
    pub const FRONT_AND_BACK: Self = Self(vk::CullModeFlags::FRONT_AND_BACK);
}

impl Default for CullMode {
    fn default() -> Self {
        Self::NONE
    }
}

impl From<CullMode> for vk::CullModeFlags {
    fn from(m: CullMode) -> Self {
        m.0
    }
}

impl From<vk::CullModeFlags> for CullMode {
    fn from(f: vk::CullModeFlags) -> Self {
        Self(f)
    }
}

impl std::ops::BitOr for CullMode {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorWriteMask(pub(crate) vk::ColorComponentFlags);

impl ColorWriteMask {
    pub const NONE: Self = Self(vk::ColorComponentFlags::empty());
    pub const R: Self = Self(vk::ColorComponentFlags::R);
    pub const G: Self = Self(vk::ColorComponentFlags::G);
    pub const B: Self = Self(vk::ColorComponentFlags::B);
    pub const A: Self = Self(vk::ColorComponentFlags::A);
    pub const RGBA: Self = Self(vk::ColorComponentFlags::RGBA);
}

impl Default for ColorWriteMask {
    fn default() -> Self {
        Self::RGBA
    }
}

impl From<ColorWriteMask> for vk::ColorComponentFlags {
    fn from(m: ColorWriteMask) -> Self {
        m.0
    }
}

impl From<vk::ColorComponentFlags> for ColorWriteMask {
    fn from(f: vk::ColorComponentFlags) -> Self {
        Self(f)
    }
}

impl std::ops::BitOr for ColorWriteMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SampleCount(pub(crate) vk::SampleCountFlags);

impl SampleCount {
    pub const S1: Self = Self(vk::SampleCountFlags::TYPE_1);
    pub const S2: Self = Self(vk::SampleCountFlags::TYPE_2);
    pub const S4: Self = Self(vk::SampleCountFlags::TYPE_4);
    pub const S8: Self = Self(vk::SampleCountFlags::TYPE_8);
    pub const S16: Self = Self(vk::SampleCountFlags::TYPE_16);
    pub const S32: Self = Self(vk::SampleCountFlags::TYPE_32);
    pub const S64: Self = Self(vk::SampleCountFlags::TYPE_64);
}

impl Default for SampleCount {
    fn default() -> Self {
        Self::S1
    }
}

impl From<SampleCount> for vk::SampleCountFlags {
    fn from(s: SampleCount) -> Self {
        s.0
    }
}

impl From<vk::SampleCountFlags> for SampleCount {
    fn from(f: vk::SampleCountFlags) -> Self {
        Self(f)
    }
}

impl std::ops::BitOr for SampleCount {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum FrontFace {
    #[default]
    CounterClockwise,
    Clockwise,
}

impl From<FrontFace> for vk::FrontFace {
    fn from(f: FrontFace) -> Self {
        match f {
            FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
            FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
        }
    }
}

impl From<vk::FrontFace> for FrontFace {
    fn from(f: vk::FrontFace) -> Self {
        match f {
            vk::FrontFace::CLOCKWISE => Self::Clockwise,
            _ => Self::CounterClockwise,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Topology {
    #[default]
    TriangleList,
    TriangleStrip,
    TriangleFan,
    LineList,
    LineStrip,
    PointList,
}

impl From<Topology> for vk::PrimitiveTopology {
    fn from(t: Topology) -> Self {
        match t {
            Topology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            Topology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
            Topology::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
            Topology::LineList => vk::PrimitiveTopology::LINE_LIST,
            Topology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            Topology::PointList => vk::PrimitiveTopology::POINT_LIST,
        }
    }
}

impl From<vk::PrimitiveTopology> for Topology {
    fn from(t: vk::PrimitiveTopology) -> Self {
        match t {
            vk::PrimitiveTopology::TRIANGLE_STRIP => Self::TriangleStrip,
            vk::PrimitiveTopology::TRIANGLE_FAN => Self::TriangleFan,
            vk::PrimitiveTopology::LINE_LIST => Self::LineList,
            vk::PrimitiveTopology::LINE_STRIP => Self::LineStrip,
            vk::PrimitiveTopology::POINT_LIST => Self::PointList,
            _ => Self::TriangleList,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum CompareOp {
    Never,
    Less,
    Equal,
    #[default]
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
}

impl From<CompareOp> for vk::CompareOp {
    fn from(op: CompareOp) -> Self {
        match op {
            CompareOp::Never => vk::CompareOp::NEVER,
            CompareOp::Less => vk::CompareOp::LESS,
            CompareOp::Equal => vk::CompareOp::EQUAL,
            CompareOp::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
            CompareOp::Greater => vk::CompareOp::GREATER,
            CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
            CompareOp::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
            CompareOp::Always => vk::CompareOp::ALWAYS,
        }
    }
}

impl From<vk::CompareOp> for CompareOp {
    fn from(op: vk::CompareOp) -> Self {
        match op {
            vk::CompareOp::NEVER => Self::Never,
            vk::CompareOp::LESS => Self::Less,
            vk::CompareOp::EQUAL => Self::Equal,
            vk::CompareOp::GREATER => Self::Greater,
            vk::CompareOp::NOT_EQUAL => Self::NotEqual,
            vk::CompareOp::GREATER_OR_EQUAL => Self::GreaterOrEqual,
            vk::CompareOp::ALWAYS => Self::Always,
            _ => Self::LessOrEqual,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PolygonMode {
    #[default]
    Fill,
    Line,
    Point,
}

impl From<PolygonMode> for vk::PolygonMode {
    fn from(m: PolygonMode) -> Self {
        match m {
            PolygonMode::Fill => vk::PolygonMode::FILL,
            PolygonMode::Line => vk::PolygonMode::LINE,
            PolygonMode::Point => vk::PolygonMode::POINT,
        }
    }
}

impl From<vk::PolygonMode> for PolygonMode {
    fn from(m: vk::PolygonMode) -> Self {
        match m {
            vk::PolygonMode::LINE => Self::Line,
            vk::PolygonMode::POINT => Self::Point,
            _ => Self::Fill,
        }
    }
}
