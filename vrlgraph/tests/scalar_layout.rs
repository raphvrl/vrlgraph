use glam::{Mat3, Vec3};
use vrlgraph::ShaderType;

fn read_f32(buf: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
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
