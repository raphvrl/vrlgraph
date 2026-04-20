use ash::vk;

use super::macros::vk_flags_newtype;

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
