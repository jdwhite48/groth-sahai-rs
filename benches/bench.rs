
#![allow(non_snake_case)]

extern crate groth_sahai;

use criterion::{criterion_group, criterion_main, Criterion};

use ark_bls12_381::{Bls12_381 as F};
use ark_ff::{UniformRand, field_new, One, Zero};
use ark_ec::{AffineCurve, ProjectiveCurve, PairingEngine};
use ark_std::{
    test_rng
};

use groth_sahai::{
    B1, B2, BT, Com1, Com2, ComT, Mat, Matrix, CRS,
    batch_commit_G1, batch_commit_G2, batch_commit_scalar_to_B1, batch_commit_scalar_to_B2, Commit1, Commit2,
    prover::*,
};

type G1Projective = <F as PairingEngine>::G1Projective;
type G1Affine = <F as PairingEngine>::G1Affine;
type G2Projective = <F as PairingEngine>::G2Projective;
type G2Affine = <F as PairingEngine>::G2Affine;
type Fqk = <F as PairingEngine>::Fqk;
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

fn bench_small_batch_commit_G1(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let crs = CRS::<F>::generate_crs(&mut rng);

    let xvars: Vec<G1Affine> = vec![
        crs.g1_gen,
        affine_group_new!(crs.g1_gen, "2")
    ];

    c.bench_function(
        &format!("commit G1"),
        |bench| {
            bench.iter(|| {
                batch_commit_G1(&xvars, &crs, &mut rng);
            });
        }
    );
}

fn bench_large_batch_commit_G1(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let crs = CRS::<F>::generate_crs(&mut rng);

    let m = 334;
    let mut xvars: Vec<G1Affine> = Vec::with_capacity(m);
    for _ in 0..m {
        xvars.push(crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine());
    }

    c.bench_function(
        &format!("commit {} G1", m),
        |bench| {
            bench.iter(|| {
                batch_commit_G1(&xvars, &crs, &mut rng);
            });
        }
    );
}

fn bench_small_batch_commit_G2(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let crs = CRS::<F>::generate_crs(&mut rng);

    let yvars: Vec<G2Affine> = vec![
        crs.g2_gen,
        affine_group_new!(crs.g2_gen, "2")
    ];

    c.bench_function(
        &format!("commit G2"),
        |bench| {
            bench.iter(|| {
                batch_commit_G2(&yvars, &crs, &mut rng);
            });
        }
    );
}

fn bench_large_batch_commit_G2(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let crs = CRS::<F>::generate_crs(&mut rng);

    let n = 334;
    let mut yvars: Vec<G2Affine> = Vec::with_capacity(n);
    for _ in 0..n {
        yvars.push(crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine());
    }

    c.bench_function(
        &format!("commit {} G2", n),
        |bench| {
            bench.iter(|| {
                batch_commit_G2(&yvars, &crs, &mut rng);
            });
        }
    );
}

fn bench_small_batch_commit_scalar_to_B1(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let crs = CRS::<F>::generate_crs(&mut rng);

    let scalar_xvars: Vec<Fr> = vec![ Fr::rand(&mut rng), Fr::rand(&mut rng) ];

    c.bench_function(
        &format!("commit scalar to B1"),
        |bench| {
            bench.iter(|| {
                batch_commit_scalar_to_B1(&scalar_xvars, &crs, &mut rng);
            });
        }
    );
}

fn bench_large_batch_commit_scalar_to_B1(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let crs = CRS::<F>::generate_crs(&mut rng);

    let m = 334;
    let mut scalar_xvars: Vec<Fr> = Vec::with_capacity(m);
    for _ in 0..m {
        scalar_xvars.push(Fr::rand(&mut rng));
    }

    c.bench_function(
        &format!("commit {} scalar to B1", m),
        |bench| {
            bench.iter(|| {
                batch_commit_scalar_to_B1(&scalar_xvars, &crs, &mut rng);
            });
        }
    );
}

fn bench_small_batch_commit_scalar_to_B2(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let crs = CRS::<F>::generate_crs(&mut rng);

    let scalar_yvars: Vec<Fr> = vec![ Fr::rand(&mut rng), Fr::rand(&mut rng) ];

    c.bench_function(
        &format!("commit scalar to B2"),
        |bench| {
            bench.iter(|| {
                batch_commit_scalar_to_B2(&scalar_yvars, &crs, &mut rng);
            });
        }
    );
}

fn bench_large_batch_commit_scalar_to_B2(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let crs = CRS::<F>::generate_crs(&mut rng);

    let n = 334;
    let mut scalar_yvars: Vec<Fr> = Vec::with_capacity(n);
    for _ in 0..n {
        scalar_yvars.push(Fr::rand(&mut rng));
    }

    c.bench_function(
        &format!("commit {} scalar to B2", n),
        |bench| {
            bench.iter(|| {
                batch_commit_scalar_to_B2(&scalar_yvars, &crs, &mut rng);
            });
        }
    );
}

fn bench_small_PPE_proof(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let crs = CRS::<F>::generate_crs(&mut rng);

    let xvars: Vec<G1Affine> = vec![
        crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
        crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()
    ];
    let yvars: Vec<G2Affine> = vec![
        crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()
    ];
    let xcoms: Commit1<F> = batch_commit_G1(&xvars, &crs, &mut rng);
    let ycoms: Commit2<F> = batch_commit_G2(&yvars, &crs, &mut rng);

    let equ: PPE<F> = PPE::<F> {
        a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
        b_consts: vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(), crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()],
        gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
        // NOTE: dummy variable for this bench
        target: Fqk::rand(&mut rng)
    };

    c.bench_function(
        &format!("prove equation with 2 G1 vars, 1 G2 var"),
        |bench| {
            bench.iter(|| {
                equ.prove(&xvars, &yvars, &xcoms, &ycoms, &crs, &mut rng);
            });
        }
    );
}

fn bench_large_PPE_proof(c: &mut Criterion) {

    std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
    let mut rng = test_rng();
    let crs = CRS::<F>::generate_crs(&mut rng);

    let m = 334;
    let mut xvars: Vec<G1Affine> = Vec::with_capacity(m);
    let mut a_consts: Vec<G1Affine> = Vec::with_capacity(m);
    for _ in 0..m {
        xvars.push(crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine());
        a_consts.push(crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine());
    }
    let n = 334;
    let mut yvars: Vec<G2Affine> = Vec::with_capacity(n);
    let mut b_consts: Vec<G2Affine> = Vec::with_capacity(n);
    for _ in 0..n {
        yvars.push(crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine());
        b_consts.push(crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine());
    }
    let xcoms: Commit1<F> = batch_commit_G1(&xvars, &crs, &mut rng);
    let ycoms: Commit2<F> = batch_commit_G2(&yvars, &crs, &mut rng);

    let mut gamma: Matrix<Fr> = Vec::with_capacity(m);
    for _ in 0..m {
        let mut vec: Vec<Fr> = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(Fr::rand(&mut rng));
        }
        gamma.push(vec);
    }

    let equ: PPE<F> = PPE::<F> {
        a_consts,
        b_consts,
        gamma,
        // NOTE: dummy variable for this bench
        target: Fqk::rand(&mut rng)
    };

    c.bench_function(
        &format!("prove equation with {} G1 vars, {} G2 var", m, n),
        |bench| {
            bench.iter(|| {
                equ.prove(&xvars, &yvars, &xcoms, &ycoms, &crs, &mut rng);
            });
        }
    );
}


criterion_group!{
    name = small_field_matrix_mul;
    config = Criterion::default().sample_size(100);
    targets =
        bench_small_field_matrix_mul,
        bench_small_field_matrix_mul_par,
}
criterion_group!{
    name = large_field_matrix_mul;
    config = Criterion::default().sample_size(50);
    targets =
        bench_large_field_matrix_mul,
        bench_large_field_matrix_mul_par
}
criterion_group!{
    name = large_B1_matrix_mul;
    config = Criterion::default().sample_size(25);
    targets =
        bench_small_B1_matrix_mul,
        bench_small_B1_matrix_mul_par,
}
criterion_group!{
    name = small_B1_matrix_mul;
    config = Criterion::default().sample_size(10);
    targets =
        bench_large_B1_matrix_mul_par,
//        bench_large_B1_matrix_mul
}
/*
criterion_group!{
    name = G1_arith;
    config = Criterion::default().sample_size(100);
    targets =
        bench_G1_scalar_mul,
        bench_G1_affine_add,
        bench_G1_projective_add,
        bench_G1_into_affine,
        bench_G1_into_projective,
        bench_B1_add,
        bench_B1_scalar_mul
}
*/
criterion_group!{
    name = small_commit;
    config = Criterion::default().sample_size(50);
    targets =
        bench_small_batch_commit_G1,
        bench_small_batch_commit_G2,
        bench_small_batch_commit_scalar_to_B1,
        bench_small_batch_commit_scalar_to_B2,
}

criterion_group!{
    name = large_commit;
    config = Criterion::default().sample_size(10);
    targets =
        bench_large_batch_commit_G1,
        bench_large_batch_commit_G2,
        bench_large_batch_commit_scalar_to_B1,
        bench_large_batch_commit_scalar_to_B2
}

criterion_group!{
    name = small_prove;
    config = Criterion::default().sample_size(50);
    targets =
        bench_small_PPE_proof,
}
criterion_group!{
    name = large_prove;
    config = Criterion::default().sample_size(10);
    targets =
        bench_large_PPE_proof
}

criterion_main!(
    small_field_matrix_mul,
    large_field_matrix_mul,
    small_B1_matrix_mul,
    large_B1_matrix_mul,
//    G1_arith
    small_commit,
    large_commit,
    small_prove,
    large_prove
);
