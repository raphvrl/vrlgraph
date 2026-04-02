use ash::vk;

/// Describes the Vulkan format for a single vertex attribute field.
///
/// Implemented for common primitive types out of the box. Enable the `glam`
/// feature to get implementations for `glam` vector types. For any other type
/// use `#[format(FORMAT)]` on the field when deriving [`VertexInput`].
pub trait VertexAttribute {
    const FORMAT: vk::Format;
}

/// Describes the full vertex input layout for a vertex struct.
///
/// Derive this trait with `#[derive(VertexInput)]` on any `#[repr(C)]` struct
/// whose fields all implement [`VertexAttribute`].
pub trait VertexInput {
    const BINDINGS: &'static [vk::VertexInputBindingDescription];
    const ATTRIBUTES: &'static [vk::VertexInputAttributeDescription];
}

macro_rules! impl_attr {
    ($ty:ty, $fmt:ident) => {
        impl VertexAttribute for $ty {
            const FORMAT: vk::Format = vk::Format::$fmt;
        }
    };
}

impl_attr!(f32, R32_SFLOAT);
impl_attr!([f32; 2], R32G32_SFLOAT);
impl_attr!([f32; 3], R32G32B32_SFLOAT);
impl_attr!([f32; 4], R32G32B32A32_SFLOAT);

impl_attr!(u32, R32_UINT);
impl_attr!([u32; 2], R32G32_UINT);
impl_attr!([u32; 3], R32G32B32_UINT);
impl_attr!([u32; 4], R32G32B32A32_UINT);

impl_attr!(i32, R32_SINT);
impl_attr!([i32; 2], R32G32_SINT);
impl_attr!([i32; 3], R32G32B32_SINT);
impl_attr!([i32; 4], R32G32B32A32_SINT);

impl_attr!([u8; 4], R8G8B8A8_UNORM);

impl_attr!(f64, R64_SFLOAT);
impl_attr!([f64; 2], R64G64_SFLOAT);
impl_attr!([f64; 3], R64G64B64_SFLOAT);
impl_attr!([f64; 4], R64G64B64A64_SFLOAT);

impl_attr!(u64, R64_UINT);
impl_attr!([u64; 2], R64G64_UINT);
impl_attr!([u64; 3], R64G64B64_UINT);
impl_attr!([u64; 4], R64G64B64A64_UINT);

impl_attr!(i64, R64_SINT);
impl_attr!([i64; 2], R64G64_SINT);
impl_attr!([i64; 3], R64G64B64_SINT);
impl_attr!([i64; 4], R64G64B64A64_SINT);

impl_attr!(u16, R16_UINT);
impl_attr!([u16; 2], R16G16_UINT);
impl_attr!([u16; 4], R16G16B16A16_UINT);

impl_attr!(i16, R16_SINT);
impl_attr!([i16; 2], R16G16_SINT);
impl_attr!([i16; 4], R16G16B16A16_SINT);

#[cfg(feature = "glam")]
mod glam_impls {
    use super::*;

    impl_attr!(glam::Vec2, R32G32_SFLOAT);
    impl_attr!(glam::Vec3, R32G32B32_SFLOAT);
    impl_attr!(glam::Vec4, R32G32B32A32_SFLOAT);

    impl_attr!(glam::UVec2, R32G32_UINT);
    impl_attr!(glam::UVec3, R32G32B32_UINT);
    impl_attr!(glam::UVec4, R32G32B32A32_UINT);

    impl_attr!(glam::IVec2, R32G32_SINT);
    impl_attr!(glam::IVec3, R32G32B32_SINT);
    impl_attr!(glam::IVec4, R32G32B32A32_SINT);

    impl_attr!(glam::DVec2, R64G64_SFLOAT);
    impl_attr!(glam::DVec3, R64G64B64_SFLOAT);
    impl_attr!(glam::DVec4, R64G64B64A64_SFLOAT);

    impl_attr!(glam::U64Vec2, R64G64_UINT);
    impl_attr!(glam::U64Vec3, R64G64B64_UINT);
    impl_attr!(glam::U64Vec4, R64G64B64A64_UINT);

    impl_attr!(glam::I64Vec2, R64G64_SINT);
    impl_attr!(glam::I64Vec3, R64G64B64_SINT);
    impl_attr!(glam::I64Vec4, R64G64B64A64_SINT);

    impl_attr!(glam::U16Vec2, R16G16_UINT);
    impl_attr!(glam::U16Vec3, R16G16B16_UINT);
    impl_attr!(glam::U16Vec4, R16G16B16A16_UINT);

    impl_attr!(glam::I16Vec2, R16G16_SINT);
    impl_attr!(glam::I16Vec3, R16G16B16_SINT);
    impl_attr!(glam::I16Vec4, R16G16B16A16_SINT);
}
