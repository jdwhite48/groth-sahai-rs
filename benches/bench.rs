use criterion::criterion_main;

mod microbenches;

use microbenches::*;

criterion_main!(bls12_bilinear_group_arith, bls12_pairing, bls12_commit, bn254_bilinear_group_arith, bn254_pairing, bn254_commit);
