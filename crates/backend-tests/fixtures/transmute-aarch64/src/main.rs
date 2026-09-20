//! Runtime scalar transmutes: `f32::from_bits`/`to_bits` and friends are
//! `mem::transmute` in core, which MIR encodes as `CastKind::Transmute` —
//! a bit reinterpretation the importer must never turn into a numeric
//! conversion (u32 -> f32 classified structurally becomes `uitofp`, the
//! miscompile the real-kernel corpus caught on 2026-09-20). `black_box`
//! keeps the casts out of const-eval so the backend actually lowers them.
use std::hint::black_box;

fn main() {
    let bits32 = black_box(0x3F80_0000u32);
    let bits64 = black_box(0x4000_0000_0000_0000u64);
    let f = black_box(1.5f32);
    let d = black_box(-0.5f64);

    let x = f32::from_bits(bits32);
    let y = f64::from_bits(bits64);
    assert!((x - 1.0).abs() < 1e-9, "f32::from_bits: {x}");
    assert!((y - 2.0).abs() < 1e-12, "f64::from_bits: {y}");
    assert_eq!(f.to_bits(), 0x3FC0_0000, "f32::to_bits");
    assert_eq!(d.to_bits(), 0xBFE0_0000_0000_0000, "f64::to_bits");

    // Round-trips through both widths, on values with asymmetric bit
    // patterns so a conversion cannot accidentally match.
    let v = black_box(3.141592653589793f64);
    assert_eq!(f64::from_bits(v.to_bits()), v, "f64 round-trip");
    let w = black_box(-127.125f32);
    assert_eq!(f32::from_bits(w.to_bits()), w, "f32 round-trip");

    println!("transmute ok");
}
