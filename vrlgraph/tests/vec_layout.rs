use vrlgraph::{DynShaderType, ShaderType};

fn read_f32(buf: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

fn read_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

// ── Vec<T> standalone ───────────────────────────────────────────────────────

#[test]
fn test_vec_f32_size() {
    let v: Vec<f32> = vec![1.0, 2.0, 3.0];
    assert_eq!(v.padded_size(), 12);
    assert_eq!(v.scalar_align(), 4);
}

#[test]
fn test_vec_f32_write() {
    let v: Vec<f32> = vec![1.0, 2.0, 3.0];
    let mut buf = vec![0u8; v.padded_size()];
    v.write_padded_dyn(&mut buf);
    assert_eq!(read_f32(&buf, 0), 1.0);
    assert_eq!(read_f32(&buf, 4), 2.0);
    assert_eq!(read_f32(&buf, 8), 3.0);
}

#[test]
fn test_vec_empty_size() {
    let v: Vec<f32> = vec![];
    assert_eq!(v.padded_size(), 0);
}

// ── Struct with only Vec field ───────────────────────────────────────────────

#[derive(ShaderType)]
struct OnlyVec {
    items: Vec<f32>,
}

#[test]
fn test_only_vec_size() {
    let s = OnlyVec { items: vec![1.0, 2.0, 3.0, 4.0] };
    assert_eq!(s.padded_size(), 16);
    assert_eq!(s.scalar_align(), 4);
}

#[test]
fn test_only_vec_write() {
    let s = OnlyVec { items: vec![10.0, 20.0] };
    let mut buf = vec![0u8; s.padded_size()];
    s.write_padded_dyn(&mut buf);
    assert_eq!(read_f32(&buf, 0), 10.0);
    assert_eq!(read_f32(&buf, 4), 20.0);
}

// ── Struct with fixed head + Vec tail ───────────────────────────────────────

#[derive(ShaderType)]
struct LightBuffer {
    count: u32,
    items: Vec<[f32; 4]>,
}

#[test]
fn test_light_buffer_empty() {
    let s = LightBuffer { count: 0, items: vec![] };
    assert_eq!(s.padded_size(), 4);
}

#[test]
fn test_light_buffer_size() {
    let s = LightBuffer {
        count: 2,
        items: vec![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
    };
    // count: 4 bytes, then 2 × [f32;4] = 2 × 16 = 32 bytes → total 36
    assert_eq!(s.padded_size(), 36);
}

#[test]
fn test_light_buffer_write() {
    let s = LightBuffer {
        count: 1,
        items: vec![[1.0, 2.0, 3.0, 4.0]],
    };
    let mut buf = vec![0u8; s.padded_size()];
    s.write_padded_dyn(&mut buf);

    assert_eq!(read_u32(&buf, 0), 1);
    assert_eq!(read_f32(&buf, 4), 1.0);
    assert_eq!(read_f32(&buf, 8), 2.0);
    assert_eq!(read_f32(&buf, 12), 3.0);
    assert_eq!(read_f32(&buf, 16), 4.0);
}

// ── Alignment: u32 head + Vec<u64> tail (u64 aligns to 8) ──────────────────

#[derive(ShaderType)]
struct AlignedVec {
    tag: u32,
    values: Vec<u64>,
}

#[test]
fn test_aligned_vec_offset() {
    let s = AlignedVec { tag: 0, values: vec![0xDEAD_BEEF_CAFE_BABE] };
    // tag at 0..4, pad to 8, values at 8..16 → padded_size = 16
    assert_eq!(s.padded_size(), 16);

    let mut buf = vec![0u8; s.padded_size()];
    s.write_padded_dyn(&mut buf);

    assert_eq!(read_u32(&buf, 0), 0);
    let v = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    assert_eq!(v, 0xDEAD_BEEF_CAFE_BABE);
}

// ── Existing ShaderType still works (no regression) ─────────────────────────

#[derive(ShaderType)]
struct Fixed {
    a: u32,
    b: f32,
}

#[test]
fn test_fixed_still_works() {
    assert_eq!(Fixed::PADDED_SIZE, 8);
    let s = Fixed { a: 42, b: 3.14 };
    let mut buf = [0u8; 8];
    s.write_padded(&mut buf);
    assert_eq!(read_u32(&buf, 0), 42);
    assert_eq!(read_f32(&buf, 4), 3.14);
}
