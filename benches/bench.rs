
#![allow(non_snake_case)]

extern crate groth_sahai;

use criterion::{criterion_group, criterion_main, Criterion};

use ark_bls12_381::{Bls12_381 as F};
use ark_ff::{UniformRand, field_new, One};
use ark_ec::{AffineCurve, ProjectiveCurve, PairingEngine};
use ark_std::test_rng;

use groth_sahai::{B1, B2, BT, Com1, Com2, ComT, Mat, Matrix};

type G1Projective = <F as PairingEngine>::G1Projective;
type G1Affine = <F as PairingEngine>::G1Affine;
type G2Projective = <F as PairingEngine>::G2Projective;
type G2Affine = <F as PairingEngine>::G2Affine;
type GT = <F as PairingEngine>::Fqk;
type Fr = <F as PairingEngine>::Fr;

// Uses an affine group generator to produce an affine group element represented by the numeric
// string.
macro_rules! affine_group_new {
    ($gen:expr, $strnum:tt) => {
        $gen.mul(field_new!(Fr, $strnum)).into_affine()
    }
}

macro_rules! affine_group_rand {
    ($gen:expr, $rng:ident) => {
        $gen.mul(Fr::rand(&mut $rng)).into_affine()
    }
}

pub fn bench_small_field_matrix_mul(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();

    // 2 x 2 matrix
    let lhs: Matrix<Fr> = vec![
        vec![ Fr::rand(&mut rng), Fr::rand(&mut rng) ],
        vec![ Fr::rand(&mut rng), Fr::rand(&mut rng) ]
    ];
    // 2 x 1 matrix
    let rhs: Matrix<Fr> = vec![
        vec![ Fr::rand(&mut rng) ],
        vec![ Fr::rand(&mut rng) ]
    ];
    c.bench_function(
        &format!("sequential (2 x 2) * (2 x 1) field matrix mult"),
        |bench| {
            bench.iter(|| {
                lhs.right_mul(&rhs, false);
            });
        }
    );
}

pub fn bench_small_field_matrix_mul_par(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();

    // 2 x 2 matrix
    let lhs: Matrix<Fr> = vec![
        vec![ Fr::rand(&mut rng), Fr::rand(&mut rng) ],
        vec![ Fr::rand(&mut rng), Fr::rand(&mut rng) ]
    ];
    // 2 x 1 matrix
    let rhs: Matrix<Fr> = vec![
        vec![ Fr::rand(&mut rng) ],
        vec![ Fr::rand(&mut rng) ]
    ];
    c.bench_function(
        &format!("concurrent (2 x 2) * (2 x 1) field matrix mult"),
        |bench| {
            bench.iter(|| {
                lhs.right_mul(&rhs, true);
            });
        }
    );
}

pub fn bench_large_field_matrix_mul(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();

    // 334 x 2 matrix
    let m = 334;
    let mut lhs: Matrix<Fr> = Vec::with_capacity(m);
    for _ in 0..m {
        lhs.push(vec![ Fr::rand(&mut rng), Fr::rand(&mut rng) ]);
    }
    // 2 x 334 matrix
    let n = 334;
    let mut rhs: Matrix<Fr> = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut tmp: Vec<Fr> = Vec::with_capacity(n);
        for _ in 0..n {
            tmp.push( Fr::rand(&mut rng) );
        }
        rhs.push(tmp);
    }
    c.bench_function(
        &format!("sequential ({} x 2) * (2 x {}) field matrix mult", m, n),
        |bench| {
            bench.iter(|| {
                lhs.right_mul(&rhs, false);
            });
        }
    );
}

pub fn bench_large_field_matrix_mul_par(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();

    // 334 x 2 matrix
    let m = 334;
    let mut lhs: Matrix<Fr> = Vec::with_capacity(999);
    for _ in 0..m {
        lhs.push(vec![ Fr::rand(&mut rng), Fr::rand(&mut rng) ]);
    }
    // 2 x 334 matrix
    let n = 334;
    let mut rhs: Matrix<Fr> = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut tmp: Vec<Fr> = Vec::with_capacity(n);
        for _ in 0..n {
            tmp.push( Fr::rand(&mut rng) );
        }
        rhs.push(tmp);
    }
    c.bench_function(
        &format!("concurrent ({} x 2) * (2 x {}) field matrix mult", m, n),
        |bench| {
            bench.iter(|| {
                lhs.right_mul(&rhs, true);
            });
        }
    );
}

pub fn bench_small_B1_matrix_mul(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let g1gen = G1Projective::rand(&mut rng).into_affine();

    // 2 x 2 matrix
    let lhs: Matrix<Com1<F>> = vec![
        vec![
            Com1::<F>( affine_group_rand!(g1gen, rng), affine_group_rand!(g1gen, rng) ),
            Com1::<F>( affine_group_rand!(g1gen, rng), affine_group_rand!(g1gen, rng) )
        ],
        vec![
            Com1::<F>( affine_group_rand!(g1gen, rng), affine_group_rand!(g1gen, rng) ),
            Com1::<F>( affine_group_rand!(g1gen, rng), affine_group_rand!(g1gen, rng) )
        ],
    ];
    // 2 x 1 matrix
    let rhs: Matrix<Fr> = vec![
        vec![ Fr::rand(&mut rng) ],
        vec![ Fr::rand(&mut rng) ]
    ];
    c.bench_function(
        &format!("sequential (2 x 2) * (2 x 1) B1 matrix mult"),
        |bench| {
            bench.iter(|| {
                lhs.right_mul(&rhs, false);
            });
        }
    );
}

pub fn bench_small_B1_matrix_mul_par(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let g1gen = G1Projective::rand(&mut rng).into_affine();

    // 2 x 2 matrix
    let lhs: Matrix<Com1<F>> = vec![
        vec![
            Com1::<F>( affine_group_rand!(g1gen, rng), affine_group_rand!(g1gen, rng) ),
            Com1::<F>( affine_group_rand!(g1gen, rng), affine_group_rand!(g1gen, rng) )
        ],
        vec![
            Com1::<F>( affine_group_rand!(g1gen, rng), affine_group_rand!(g1gen, rng) ),
            Com1::<F>( affine_group_rand!(g1gen, rng), affine_group_rand!(g1gen, rng) )
        ],
    ];
    // 2 x 1 matrix
    let rhs: Matrix<Fr> = vec![
        vec![ Fr::rand(&mut rng) ],
        vec![ Fr::rand(&mut rng) ]
    ];
    c.bench_function(
        &format!("concurrent (2 x 2) * (2 x 1) B1 matrix mult"),
        |bench| {
            bench.iter(|| {
                lhs.right_mul(&rhs, true);
            });
        }
    );
}

pub fn bench_large_B1_matrix_mul(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let g1gen = G1Projective::rand(&mut rng).into_affine();

    // 334 x 2 matrix
    let m = 334;
    let mut lhs: Matrix<Com1<F>> = Vec::with_capacity(m);
    for _ in 0..m {
        lhs.push(vec![
            Com1::<F>( affine_group_rand!(g1gen, rng), affine_group_rand!(g1gen, rng) ),
            Com1::<F>( affine_group_rand!(g1gen, rng), affine_group_rand!(g1gen, rng) ),
        ]);
    }
    // 2 x 334 matrix
    let n = 334;
    let mut rhs: Matrix<Fr> = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut tmp: Vec<Fr> = Vec::with_capacity(n);
        for _ in 0..n {
            tmp.push( Fr::rand(&mut rng) );
        }
        rhs.push(tmp);
    }
    c.bench_function(
        &format!("sequential ({} x 2) * (2 x {}) B1 matrix mult", m, n),
        |bench| {
            bench.iter(|| {
                lhs.right_mul(&rhs, false);
            });
        }
    );
}

pub fn bench_large_B1_matrix_mul_par(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let g1gen = G1Projective::rand(&mut rng).into_affine();

    // 334 x 2 matrix
    let m = 334;
    let mut lhs: Matrix<Com1<F>> = Vec::with_capacity(m);
    for _ in 0..m {
        lhs.push(vec![
            Com1::<F>( affine_group_rand!(g1gen, rng), affine_group_rand!(g1gen, rng) ),
            Com1::<F>( affine_group_rand!(g1gen, rng), affine_group_rand!(g1gen, rng) ),
        ]);
    }
    // 2 x 334 matrix
    let n = 334;
    let mut rhs: Matrix<Fr> = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut tmp: Vec<Fr> = Vec::with_capacity(n);
        for _ in 0..n {
            tmp.push( Fr::rand(&mut rng) );
        }
        rhs.push(tmp);
    }
    c.bench_function(
        &format!("concurrent ({} x 2) * (2 x {}) B1 matrix mult", m, n),
        |bench| {
            bench.iter(|| {
                lhs.right_mul(&rhs, true);
            });
        }
    );
}

fn bench_B1_scalar_mul(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let g11 = G1Projective::rand(&mut rng).into_affine();
    let g12 = G1Projective::rand(&mut rng).into_affine();
    let b1 = Com1::<F>( g11, g12 );
    let fr = Fr::rand(&mut rng);

    c.bench_function(
        &format!("B1 scalar mul"),
        |bench| {
            bench.iter(|| {
                b1.scalar_mul(&fr);
            });
        }
    );
}

fn bench_B1_add(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let g11 = G1Projective::rand(&mut rng).into_affine();
    let g12 = G1Projective::rand(&mut rng).into_affine();
    let b11 = Com1::<F>( g11, g12 );
    let g21 = G1Projective::rand(&mut rng).into_affine();
    let g22 = G1Projective::rand(&mut rng).into_affine();
    let b12 = Com1::<F>( g21, g22 );

    c.bench_function(
        &format!("B1 add"),
        |bench| {
            bench.iter(|| {
                b11 + b12;
            });
        }
    );
}

fn bench_G1_scalar_mul(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let g1gen = G1Projective::rand(&mut rng).into_affine();
    let fr = Fr::rand(&mut rng);

    c.bench_function(
        &format!("G1 scalar mul"),
        |bench| {
            bench.iter(|| {
                g1gen.mul(fr);
            });
        }
    );
}

fn bench_G1_affine_add(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let g11 = G1Projective::rand(&mut rng).into_affine();
    let g12 = G1Projective::rand(&mut rng).into_affine();

    c.bench_function(
        &format!("G1 affine add"),
        |bench| {
            bench.iter(|| {
                g11 + g12;
            });
        }
    );
}

fn bench_G1_projective_add(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let g11 = G1Projective::rand(&mut rng);
    let g12 = G1Projective::rand(&mut rng);

    c.bench_function(
        &format!("G1 projective add"),
        |bench| {
            bench.iter(|| {
                g11 + g12;
            });
        }
    );
}

fn bench_G1_into_affine(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let g1gen = G1Projective::rand(&mut rng);

    c.bench_function(
        &format!("G1 projective into affine"),
        |bench| {
            bench.iter(|| {
                g1gen.into_affine();
            });
        }
    );
}

fn bench_G1_into_projective(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let g1gen = G1Projective::rand(&mut rng).into_affine();

    c.bench_function(
        &format!("G1 affine into projective"),
        |bench| {
            bench.iter(|| {
                g1gen.into_projective();
            });
        }
    );
}

criterion_group!{
    name = field_matrix_mul;
    config = Criterion::default().sample_size(50);
    targets = bench_small_field_matrix_mul, bench_small_field_matrix_mul_par, bench_large_field_matrix_mul, bench_large_field_matrix_mul_par
}
criterion_group!{
    name = B1_matrix_mul;
    config = Criterion::default().sample_size(10);
    targets = bench_small_B1_matrix_mul, bench_small_B1_matrix_mul_par//, bench_large_B1_matrix_mul_par, bench_large_B1_matrix_mul
}
/*
criterion_group!{
    name = G1_arith;
    config = Criterion::default().sample_size(100);
    targets = bench_G1_scalar_mul, bench_G1_affine_add, bench_G1_projective_add, bench_G1_into_affine, bench_G1_into_projective, bench_B1_add, bench_B1_scalar_mul
}
*/
criterion_main!(field_matrix_mul, B1_matrix_mul);//, G1_arith);
