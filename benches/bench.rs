use criterion::criterion_main;

mod util;
mod microbenches;
mod groth16;

use microbenches::*;
use groth16::*;

criterion_main!(
    bls12_microbenches,
    bn254_microbenches,
    bls12_gs_over_groth16,
    bn254_gs_over_groth16,
);
