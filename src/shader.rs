/// Trait for types that can be serialized to GPU-compatible padded byte layouts
/// (std140 / std430).
///
/// All standard scalar, vector and matrix types implement this trait. Enable the
/// `glam` feature for `glam` type support.
///
/// Derive this trait with `#[derive(ShaderType)]` on a struct to automatically
/// generate `Clone`, `Copy`, and a [`write_padded`](ShaderType::write_padded)
/// implementation that inserts the correct padding for std140 (default) or
/// std430.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(ShaderType)]
/// struct Camera {
///     view: [[f32; 4]; 4],
///     proj: [[f32; 4]; 4],
///     position: [f32; 3],
/// }
///
/// let cam = Camera { view: ..., proj: ..., position: [0.0, 1.0, 0.0] };
/// cmd.push_shader(&cam);
/// ```
pub trait ShaderType {
    /// Size of this type in bytes after padding has been applied.
    const PADDED_SIZE: usize;

    /// Write this value into `dst` with the correct GPU layout padding.
    ///
    /// `dst` must be at least [`PADDED_SIZE`](ShaderType::PADDED_SIZE) bytes.
    /// Bytes beyond the data are left untouched (caller should zero-init).
    fn write_padded(&self, dst: &mut [u8]);
}

pub const fn round_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

macro_rules! impl_shader_type_pod {
    ($($ty:ty),*) => {
        $(impl ShaderType for $ty {
            const PADDED_SIZE: usize = ::core::mem::size_of::<$ty>();

            fn write_padded(&self, dst: &mut [u8]) {
                let bytes = bytemuck::bytes_of(self);
                dst[..bytes.len()].copy_from_slice(bytes);
            }
        })*
    };
}

impl_shader_type_pod!(
    f32,
    u32,
    i32,
    u64,
    [f32; 2],
    [f32; 3],
    [f32; 4],
    [u32; 2],
    [u32; 3],
    [u32; 4],
    [i32; 2],
    [i32; 3],
    [i32; 4],
    [[f32; 4]; 4],
    [[f32; 4]; 3]
);

#[cfg(feature = "glam")]
mod glam_impls {
    use super::*;

    impl_shader_type_pod!(
        glam::Vec2,
        glam::Vec3,
        glam::Vec3A,
        glam::Vec4,
        glam::UVec2,
        glam::UVec3,
        glam::UVec4,
        glam::IVec2,
        glam::IVec3,
        glam::IVec4,
        glam::Mat4
    );

    impl ShaderType for glam::Mat3 {
        const PADDED_SIZE: usize = 48;

        fn write_padded(&self, dst: &mut [u8]) {
            let cols = self.to_cols_array_2d();
            for (i, col) in cols.iter().enumerate() {
                let off = i * 16;
                dst[off..off + 12].copy_from_slice(bytemuck::cast_slice(col));
            }
        }
    }
}
