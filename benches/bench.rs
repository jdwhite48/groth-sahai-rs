extern crate groth_sahai;

use criterion::{criterion_group, criterion_main, Criterion};

use ark_bls12_381::{Bls12_381 as F};
use ark_ff::{UniformRand, field_new, One};
use ark_ec::{ProjectiveCurve, PairingEngine};
use ark_std::test_rng;

use groth_sahai::{B1, B2, BT, Com1, Com2, ComT, Matrix, matrix_mul};

type G1Projective = <F as PairingEngine>::G1Projective;
type G1Affine = <F as PairingEngine>::G1Affine;
type G2Projective = <F as PairingEngine>::G2Projective;
type G2Affine = <F as PairingEngine>::G2Affine;
type GT = <F as PairingEngine>::Fqk;
type Fr = <F as PairingEngine>::Fr;



pub fn bench_small_scalar_matrix_mul(c: &mut Criterion) {    
        
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
        &format!("sequential (2 x 2) * (2 x 1) matrix mult"),
        |bench| {            
            bench.iter(|| {
                matrix_mul::<Fr>(&lhs, &rhs, false);
            });
        }
    );
}

pub fn bench_small_scalar_matrix_mul_rayon(c: &mut Criterion) {    
    
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
        &format!("concurrent (2 x 2) * (2 x 1) matrix mult"),
        |bench| {            
            bench.iter(|| {
                matrix_mul::<Fr>(&lhs, &rhs, true);
            });
        }
    );
}

pub fn bench_large_scalar_matrix_mul(c: &mut Criterion) {    
        
    let mut rng = test_rng();

    // 9999 x 2 matrix
    let mut lhs: Matrix<Fr> = Vec::with_capacity(999);
    for _ in 0..999 {
        lhs.push(vec![ Fr::rand(&mut rng), Fr::rand(&mut rng) ]);
    }
    // 2 x 10000 matrix
    let mut rhs: Matrix<Fr> = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut tmp: Vec<Fr> = Vec::with_capacity(1000);
        for _ in 0..1000 {
            tmp.push( Fr::rand(&mut rng) );
        }
        rhs.push(tmp);
    }
    c.bench_function(
        &format!("sequential (999 x 2) * (2 x 1000) matrix mult"),
        |bench| {            
            bench.iter(|| {
                matrix_mul::<Fr>(&lhs, &rhs, false);
            });
        }
    );
}

pub fn bench_large_scalar_matrix_mul_rayon(c: &mut Criterion) {    
        
    let mut rng = test_rng();

    // 999 x 2 matrix
    let mut lhs: Matrix<Fr> = Vec::with_capacity(999);
    for _ in 0..999 {
        lhs.push(vec![ Fr::rand(&mut rng), Fr::rand(&mut rng) ]);
    }
    // 2 x 10000 matrix
    let mut rhs: Matrix<Fr> = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut tmp: Vec<Fr> = Vec::with_capacity(1000);
        for _ in 0..1000 {
            tmp.push( Fr::rand(&mut rng) );
        }
        rhs.push(tmp);
    }
    c.bench_function(
        &format!("concurrent (999 x 2) * (2 x 1000) matrix mult"),
        |bench| {            
            bench.iter(|| {
                matrix_mul::<Fr>(&lhs, &rhs, true);
            });
        }
    );
}

criterion_group!(benches, bench_small_scalar_matrix_mul, bench_small_scalar_matrix_mul_rayon, bench_large_scalar_matrix_mul, bench_large_scalar_matrix_mul_rayon);
criterion_main!(benches);
