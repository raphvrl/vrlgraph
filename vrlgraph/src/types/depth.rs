use ash::vk;

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
