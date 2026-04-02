use glam::{Mat3, Vec3, Vec4};
use vrlgraph::ShaderType;

fn read_f32(buf: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

fn read_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

fn read_f64(buf: &[u8], offset: usize) -> f64 {
    f64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap())
}

fn read_u16(buf: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(buf[offset..offset + 2].try_into().unwrap())
}

#[derive(ShaderType)]
struct Vec3ThenF32 {
    a: Vec3,
    b: f32,
}

#[test]
fn test_scalar_vec3_no_padding() {
    assert_eq!(Vec3ThenF32::PADDED_SIZE, 16);

    let v = Vec3ThenF32 {
        a: Vec3::new(1.0, 2.0, 3.0),
        b: 4.0,
    };
    let mut buf = [0u8; 16];
    v.write_padded(&mut buf);

    assert_eq!(read_f32(&buf, 0), 1.0);
    assert_eq!(read_f32(&buf, 4), 2.0);
    assert_eq!(read_f32(&buf, 8), 3.0);
    assert_eq!(read_f32(&buf, 12), 4.0);
}

#[derive(ShaderType)]
struct TwoVec3 {
    a: Vec3,
    b: Vec3,
}

#[test]
fn test_scalar_vec3_consecutive() {
    assert_eq!(TwoVec3::PADDED_SIZE, 24);

    let v = TwoVec3 {
        a: Vec3::new(1.0, 2.0, 3.0),
        b: Vec3::new(4.0, 5.0, 6.0),
    };
    let mut buf = [0u8; 24];
    v.write_padded(&mut buf);

    assert_eq!(read_f32(&buf, 0), 1.0);
    assert_eq!(read_f32(&buf, 4), 2.0);
    assert_eq!(read_f32(&buf, 8), 3.0);
    assert_eq!(read_f32(&buf, 12), 4.0);
    assert_eq!(read_f32(&buf, 16), 5.0);
    assert_eq!(read_f32(&buf, 20), 6.0);
}

#[test]
fn test_scalar_mat3_size() {
    assert_eq!(Mat3::PADDED_SIZE, 36);

    let m = Mat3::from_cols(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    );
    let mut buf = [0u8; 36];
    m.write_padded(&mut buf);

    assert_eq!(read_f32(&buf, 0), 1.0);
    assert_eq!(read_f32(&buf, 4), 2.0);
    assert_eq!(read_f32(&buf, 8), 3.0);
    assert_eq!(read_f32(&buf, 12), 4.0);
    assert_eq!(read_f32(&buf, 16), 5.0);
    assert_eq!(read_f32(&buf, 20), 6.0);
    assert_eq!(read_f32(&buf, 24), 7.0);
    assert_eq!(read_f32(&buf, 28), 8.0);
    assert_eq!(read_f32(&buf, 32), 9.0);
}

#[derive(ShaderType)]
struct MixedAlignment {
    a: u32,
    b: Vec3,
    c: f32,
}

#[test]
fn test_scalar_mixed_alignment() {
    assert_eq!(MixedAlignment::PADDED_SIZE, 20);

    let v = MixedAlignment {
        a: 42,
        b: Vec3::new(1.0, 2.0, 3.0),
        c: 7.0,
    };
    let mut buf = [0u8; 20];
    v.write_padded(&mut buf);

    let a = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    assert_eq!(a, 42);
    assert_eq!(read_f32(&buf, 4), 1.0);
    assert_eq!(read_f32(&buf, 8), 2.0);
    assert_eq!(read_f32(&buf, 12), 3.0);
    assert_eq!(read_f32(&buf, 16), 7.0);
}

#[derive(ShaderType)]
struct U64Alignment {
    a: u32,
    b: u64,
}

#[test]
fn test_scalar_u64_alignment() {
    assert_eq!(U64Alignment::PADDED_SIZE, 16);

    let v = U64Alignment {
        a: 0xDEAD_BEEF,
        b: 0x0123_4567_89AB_CDEF,
    };
    let mut buf = [0u8; 16];
    v.write_padded(&mut buf);

    let a = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    assert_eq!(a, 0xDEAD_BEEF);
    let b = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    assert_eq!(b, 0x0123_4567_89AB_CDEF);
}

// --- New tests ---

#[test]
fn test_bool_as_u32() {
    assert_eq!(bool::PADDED_SIZE, 4);

    let mut buf = [0u8; 4];
    true.write_padded(&mut buf);
    assert_eq!(read_u32(&buf, 0), 1);

    false.write_padded(&mut buf);
    assert_eq!(read_u32(&buf, 0), 0);
}

#[derive(ShaderType)]
struct WithBool {
    flag: bool,
    value: f32,
}

#[test]
fn test_bool_in_struct() {
    assert_eq!(WithBool::PADDED_SIZE, 8);

    let v = WithBool {
        flag: true,
        value: 3.14,
    };
    let mut buf = [0u8; 8];
    v.write_padded(&mut buf);

    assert_eq!(read_u32(&buf, 0), 1);
    assert_eq!(read_f32(&buf, 4), 3.14);
}

#[derive(ShaderType)]
struct F64Alignment {
    a: u32,
    b: f64,
}

#[test]
fn test_f64_alignment() {
    assert_eq!(F64Alignment::PADDED_SIZE, 16);

    let v = F64Alignment {
        a: 42,
        b: core::f64::consts::PI,
    };
    let mut buf = [0u8; 16];
    v.write_padded(&mut buf);

    assert_eq!(read_u32(&buf, 0), 42);
    assert_eq!(read_f64(&buf, 8), core::f64::consts::PI);
}

#[test]
fn test_mat2_raw_array() {
    assert_eq!(<[[f32; 2]; 2]>::PADDED_SIZE, 16);

    let mat: [[f32; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    let mut buf = [0u8; 16];
    mat.write_padded(&mut buf);

    assert_eq!(read_f32(&buf, 0), 1.0);
    assert_eq!(read_f32(&buf, 4), 2.0);
    assert_eq!(read_f32(&buf, 8), 3.0);
    assert_eq!(read_f32(&buf, 12), 4.0);
}

#[test]
fn test_mat3_raw_array() {
    assert_eq!(<[[f32; 3]; 3]>::PADDED_SIZE, 36);

    let mat: [[f32; 3]; 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let mut buf = [0u8; 36];
    mat.write_padded(&mut buf);

    for i in 0..9 {
        assert_eq!(read_f32(&buf, i * 4), (i + 1) as f32);
    }
}

#[test]
fn test_generic_array_n_greater_than_4() {
    assert_eq!(<[f32; 8]>::PADDED_SIZE, 32);

    let arr: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut buf = [0u8; 32];
    arr.write_padded(&mut buf);

    for i in 0..8 {
        assert_eq!(read_f32(&buf, i * 4), (i + 1) as f32);
    }
}

#[derive(ShaderType)]
struct WithVec4Array {
    lights: [Vec4; 4],
}

#[test]
fn test_vec4_array_in_struct() {
    assert_eq!(WithVec4Array::PADDED_SIZE, 64);
}

#[test]
fn test_dmat4_raw_array() {
    assert_eq!(<[[f64; 4]; 4]>::PADDED_SIZE, 128);

    let mat: [[f64; 4]; 4] = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ];
    let mut buf = [0u8; 128];
    mat.write_padded(&mut buf);

    for i in 0..16 {
        assert_eq!(read_f64(&buf, i * 8), (i + 1) as f64);
    }
}

#[test]
fn test_u16_type() {
    assert_eq!(u16::PADDED_SIZE, 2);

    let mut buf = [0u8; 2];
    42u16.write_padded(&mut buf);
    assert_eq!(read_u16(&buf, 0), 42);
}

#[derive(ShaderType)]
struct U16Alignment {
    a: u16,
    b: u16,
    c: f32,
}

#[test]
fn test_u16_alignment_in_struct() {
    assert_eq!(U16Alignment::PADDED_SIZE, 8);

    let v = U16Alignment {
        a: 1,
        b: 2,
        c: 3.0,
    };
    let mut buf = [0u8; 8];
    v.write_padded(&mut buf);

    assert_eq!(read_u16(&buf, 0), 1);
    assert_eq!(read_u16(&buf, 2), 2);
    assert_eq!(read_f32(&buf, 4), 3.0);
}

#[test]
fn test_nested_array_of_mat4() {
    assert_eq!(<[[[f32; 4]; 4]; 2]>::PADDED_SIZE, 128);
}

#[test]
fn test_bvec3() {
    use glam::BVec3;

    assert_eq!(BVec3::PADDED_SIZE, 12);

    let v = BVec3::new(true, false, true);
    let mut buf = [0u8; 12];
    v.write_padded(&mut buf);

    assert_eq!(read_u32(&buf, 0), 1);
    assert_eq!(read_u32(&buf, 4), 0);
    assert_eq!(read_u32(&buf, 8), 1);
}
