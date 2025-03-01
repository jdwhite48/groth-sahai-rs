use criterion::criterion_main;

mod microbenches;
mod groth16;

use microbenches::*;
use groth16::*;

criterion_main!(
    bls12_bilinear_group_arith,
    bls12_pairing,
    bls12_commit,
    bls12_gs_over_groth16_commit,
    bls12_gs_over_groth16_proof,
    bls12_gs_over_groth16_verify,
    bn254_bilinear_group_arith,
    bn254_pairing,
    bn254_commit
);
