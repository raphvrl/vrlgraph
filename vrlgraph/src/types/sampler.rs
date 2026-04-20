use ash::vk;

use super::macros::vk_flags_newtype;

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
