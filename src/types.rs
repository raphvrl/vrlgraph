use ash::vk;

macro_rules! vk_flags_newtype {
    (
        pub struct $Name:ident($vk_ty:ty);
        default = $default:ident;
        bitor;
        $(const $CONST:ident = $val:expr;)*
    ) => {
        vk_flags_newtype! {
            pub struct $Name($vk_ty);
            default = $default;
            $(const $CONST = $val;)*
        }

        impl std::ops::BitOr for $Name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }
    };

    (
        pub struct $Name:ident($vk_ty:ty);
        default = $default:ident;
        $(const $CONST:ident = $val:expr;)*
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $Name(pub(crate) $vk_ty);

        impl $Name {
            $(pub const $CONST: Self = Self($val);)*
        }

        impl Default for $Name {
            fn default() -> Self {
                Self::$default
            }
        }

        impl From<$Name> for $vk_ty {
            fn from(v: $Name) -> Self {
                v.0
            }
        }

        impl From<$vk_ty> for $Name {
            fn from(v: $vk_ty) -> Self {
                Self(v)
            }
        }
    };
}

vk_flags_newtype! {
    pub struct CullMode(vk::CullModeFlags);
    default = NONE;
    bitor;
    const NONE = vk::CullModeFlags::NONE;
    const FRONT = vk::CullModeFlags::FRONT;
    const BACK = vk::CullModeFlags::BACK;
    const FRONT_AND_BACK = vk::CullModeFlags::FRONT_AND_BACK;
}

vk_flags_newtype! {
    pub struct ColorWriteMask(vk::ColorComponentFlags);
    default = RGBA;
    bitor;
    const NONE = vk::ColorComponentFlags::empty();
    const R = vk::ColorComponentFlags::R;
    const G = vk::ColorComponentFlags::G;
    const B = vk::ColorComponentFlags::B;
    const A = vk::ColorComponentFlags::A;
    const RGBA = vk::ColorComponentFlags::RGBA;
}

vk_flags_newtype! {
    pub struct SampleCount(vk::SampleCountFlags);
    default = S1;
    bitor;
    const S1 = vk::SampleCountFlags::TYPE_1;
    const S2 = vk::SampleCountFlags::TYPE_2;
    const S4 = vk::SampleCountFlags::TYPE_4;
    const S8 = vk::SampleCountFlags::TYPE_8;
    const S16 = vk::SampleCountFlags::TYPE_16;
    const S32 = vk::SampleCountFlags::TYPE_32;
    const S64 = vk::SampleCountFlags::TYPE_64;
}

vk_flags_newtype! {
    pub struct Filter(vk::Filter);
    default = LINEAR;
    const NEAREST = vk::Filter::NEAREST;
    const LINEAR = vk::Filter::LINEAR;
}

vk_flags_newtype! {
    pub struct MipmapMode(vk::SamplerMipmapMode);
    default = LINEAR;
    const NEAREST = vk::SamplerMipmapMode::NEAREST;
    const LINEAR = vk::SamplerMipmapMode::LINEAR;
}

vk_flags_newtype! {
    pub struct AddressMode(vk::SamplerAddressMode);
    default = REPEAT;
    const REPEAT = vk::SamplerAddressMode::REPEAT;
    const MIRRORED_REPEAT = vk::SamplerAddressMode::MIRRORED_REPEAT;
    const CLAMP_TO_EDGE = vk::SamplerAddressMode::CLAMP_TO_EDGE;
    const CLAMP_TO_BORDER = vk::SamplerAddressMode::CLAMP_TO_BORDER;
}

vk_flags_newtype! {
    pub struct BorderColor(vk::BorderColor);
    default = FLOAT_TRANSPARENT_BLACK;
    const FLOAT_TRANSPARENT_BLACK = vk::BorderColor::FLOAT_TRANSPARENT_BLACK;
    const INT_TRANSPARENT_BLACK = vk::BorderColor::INT_TRANSPARENT_BLACK;
    const FLOAT_OPAQUE_BLACK = vk::BorderColor::FLOAT_OPAQUE_BLACK;
    const INT_OPAQUE_BLACK = vk::BorderColor::INT_OPAQUE_BLACK;
    const FLOAT_OPAQUE_WHITE = vk::BorderColor::FLOAT_OPAQUE_WHITE;
    const INT_OPAQUE_WHITE = vk::BorderColor::INT_OPAQUE_WHITE;
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
