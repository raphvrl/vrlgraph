use ash::vk;

use super::macros::vk_flags_newtype;

vk_flags_newtype! {
    pub struct CullMode(vk::CullModeFlags);
    default = NONE;
    bitor;
    const NONE = vk::CullModeFlags::NONE;
    const FRONT = vk::CullModeFlags::FRONT;
    const BACK = vk::CullModeFlags::BACK;
    const FRONT_AND_BACK = vk::CullModeFlags::FRONT_AND_BACK;
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
