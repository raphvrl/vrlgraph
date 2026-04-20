/// Trait for types whose GPU layout size and alignment is known at compile time.
///
/// Implemented automatically by `#[derive(ShaderType)]` for structs with no `Vec<T>` fields.
/// All such types also implement [`DynShaderType`] via a blanket impl.
///
/// Use [`DynShaderType`] when you need to accept both fixed-size and dynamic types.
pub trait DynShaderType {
    fn scalar_align(&self) -> usize;
    fn padded_size(&self) -> usize;
    fn write_padded_dyn(&self, dst: &mut [u8]);
}

impl<T: ShaderType> DynShaderType for T {
    fn scalar_align(&self) -> usize {
        T::SCALAR_ALIGN
    }
    fn padded_size(&self) -> usize {
        T::PADDED_SIZE
    }
    fn write_padded_dyn(&self, dst: &mut [u8]) {
        self.write_padded(dst)
    }
}

impl<T: ShaderType> DynShaderType for Vec<T> {
    fn scalar_align(&self) -> usize {
        T::SCALAR_ALIGN
    }
    fn padded_size(&self) -> usize {
        T::PADDED_SIZE * self.len()
    }
    fn write_padded_dyn(&self, dst: &mut [u8]) {
        for (i, elem) in self.iter().enumerate() {
            elem.write_padded(&mut dst[i * T::PADDED_SIZE..]);
        }
    }
}

/// Trait for types that can be serialized to GPU-compatible padded byte layouts
/// using **scalar layout** (`VK_EXT_scalar_block_layout`).
///
/// Each type is aligned to the size of its scalar component (4 bytes for
/// `f32`/`u32`/`i32`, 8 bytes for `f64`/`u64`/`i64`, 2 bytes for `u16`/`i16`).
/// This matches the layout produced by Slang/SPIR-V for buffer references (BDA)
/// and push constants.
///
/// All standard scalar, vector and matrix types implement this trait. Enable the
/// `glam` feature for `glam` type support.
///
/// A **blanket implementation** is provided for `[T; N]` where `T: ShaderType`,
/// so any fixed-size array of a supported type is automatically supported
/// (including nested arrays like `[[f32; 4]; 4]` for mat4).
///
/// Derive this trait with `#[derive(ShaderType)]` on a struct to automatically
/// generate `Clone`, `Copy`, and a [`write_padded`](ShaderType::write_padded)
/// implementation that inserts the correct padding for scalar layout.
///
/// # Fixed-size vs dynamic fields
///
/// Prefer fixed-size arrays (`[T; N]`) over `Vec<T>` whenever the maximum
/// element count is known at compile time. Fixed-size structs implement `Copy`,
/// their size is a compile-time constant, and `write_padded` writes directly
/// into a stack buffer — no heap allocation occurs.
///
/// Use `Vec<T>` only when the count is truly dynamic at runtime. A struct with
/// a `Vec<T>` tail field derives [`DynShaderType`] instead of `ShaderType`:
/// it is not `Copy`, and writing it allocates a temporary `Vec<u8>` on the heap.
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
/// cmd.push_constants(&cam);
/// ```
pub trait ShaderType {
    const SCALAR_ALIGN: usize;
    const PADDED_SIZE: usize;
    const UNPADDED_SIZE: usize = Self::PADDED_SIZE;

    fn write_padded(&self, dst: &mut [u8]);
}

pub const fn round_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

macro_rules! impl_shader_type_pod {
    ($align:expr; $($ty:ty),*) => {
        $(impl ShaderType for $ty {
            const SCALAR_ALIGN: usize = $align;
            const PADDED_SIZE: usize = ::core::mem::size_of::<$ty>();

            fn write_padded(&self, dst: &mut [u8]) {
                let bytes = bytemuck::bytes_of(self);
                dst[..bytes.len()].copy_from_slice(bytes);
            }
        })*
    };
}

impl_shader_type_pod!(4; f32, u32, i32);
impl_shader_type_pod!(8; u64, f64, i64);
impl_shader_type_pod!(2; u16, i16);

impl ShaderType for bool {
    const SCALAR_ALIGN: usize = 4;
    const PADDED_SIZE: usize = 4;

    fn write_padded(&self, dst: &mut [u8]) {
        let v: u32 = if *self { 1 } else { 0 };
        dst[..4].copy_from_slice(&v.to_le_bytes());
    }
}

impl<T: ShaderType, const N: usize> ShaderType for [T; N] {
    const SCALAR_ALIGN: usize = T::SCALAR_ALIGN;
    const PADDED_SIZE: usize = T::PADDED_SIZE * N;

    fn write_padded(&self, dst: &mut [u8]) {
        for (i, elem) in self.iter().enumerate() {
            elem.write_padded(&mut dst[i * T::PADDED_SIZE..]);
        }
    }
}

#[cfg(feature = "glam")]
mod glam_impls {
    use super::*;

    impl_shader_type_pod!(4;
        glam::Vec2, glam::Vec3, glam::Vec3A, glam::Vec4,
        glam::UVec2, glam::UVec3, glam::UVec4,
        glam::IVec2, glam::IVec3, glam::IVec4,
        glam::Mat2, glam::Mat4
    );

    impl_shader_type_pod!(8;
        glam::DVec2, glam::DVec3, glam::DVec4,
        glam::DMat2, glam::DMat4,
        glam::U64Vec2, glam::U64Vec3, glam::U64Vec4,
        glam::I64Vec2, glam::I64Vec3, glam::I64Vec4
    );

    impl_shader_type_pod!(2;
        glam::U16Vec2, glam::U16Vec3, glam::U16Vec4,
        glam::I16Vec2, glam::I16Vec3, glam::I16Vec4
    );

    impl ShaderType for glam::Mat3 {
        const SCALAR_ALIGN: usize = 4;
        const PADDED_SIZE: usize = 36;

        fn write_padded(&self, dst: &mut [u8]) {
            let cols = self.to_cols_array_2d();
            for (i, col) in cols.iter().enumerate() {
                let off = i * 12;
                dst[off..off + 12].copy_from_slice(bytemuck::cast_slice(col));
            }
        }
    }

    impl ShaderType for glam::DMat3 {
        const SCALAR_ALIGN: usize = 8;
        const PADDED_SIZE: usize = 72;

        fn write_padded(&self, dst: &mut [u8]) {
            let cols = self.to_cols_array_2d();
            for (i, col) in cols.iter().enumerate() {
                let off = i * 24;
                dst[off..off + 24].copy_from_slice(bytemuck::cast_slice(col));
            }
        }
    }

    impl ShaderType for glam::BVec2 {
        const SCALAR_ALIGN: usize = 4;
        const PADDED_SIZE: usize = 8;

        fn write_padded(&self, dst: &mut [u8]) {
            let arr: [u32; 2] = [self.x as u32, self.y as u32];
            dst[..8].copy_from_slice(bytemuck::cast_slice(&arr));
        }
    }

    impl ShaderType for glam::BVec3 {
        const SCALAR_ALIGN: usize = 4;
        const PADDED_SIZE: usize = 12;

        fn write_padded(&self, dst: &mut [u8]) {
            let arr: [u32; 3] = [self.x as u32, self.y as u32, self.z as u32];
            dst[..12].copy_from_slice(bytemuck::cast_slice(&arr));
        }
    }

    impl ShaderType for glam::BVec4 {
        const SCALAR_ALIGN: usize = 4;
        const PADDED_SIZE: usize = 16;

        fn write_padded(&self, dst: &mut [u8]) {
            let arr: [u32; 4] = [self.x as u32, self.y as u32, self.z as u32, self.w as u32];
            dst[..16].copy_from_slice(bytemuck::cast_slice(&arr));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::round_up;

    #[test]
    fn round_up_already_aligned() {
        assert_eq!(round_up(0, 4), 0);
        assert_eq!(round_up(16, 16), 16);
        assert_eq!(round_up(8, 4), 8);
    }

    #[test]
    fn round_up_unaligned() {
        assert_eq!(round_up(1, 4), 4);
        assert_eq!(round_up(5, 4), 8);
        assert_eq!(round_up(13, 16), 16);
        assert_eq!(round_up(17, 16), 32);
    }
}
