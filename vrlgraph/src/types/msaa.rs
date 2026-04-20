use ash::vk;

use super::macros::vk_flags_newtype;

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
