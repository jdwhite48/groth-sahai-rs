use criterion::criterion_main;

mod microbenches;
mod groth16;
mod util;

use microbenches::*;
use groth16::*;

// Generated via script before running benchmarks
//use util::new_size_file as setup;

criterion_main!(
    bls12_bilinear_group_arith,
    bls12_pairing,
    bls12_commit,
    bls12_gs_over_groth16_commit,
    bls12_gs_over_groth16_proof,
    bls12_gs_over_groth16_verify,
    bls12_plain_groth16_verify,
    bn254_bilinear_group_arith,
    bn254_pairing,
    bn254_commit,
);
