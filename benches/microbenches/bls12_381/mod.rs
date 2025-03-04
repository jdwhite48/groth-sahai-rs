use criterion::{criterion_group, criterion_main, Criterion};

use std::time::Duration;

use ark_ec::{
    pairing::Pairing,
    CurveGroup,
};
use ark_ff::UniformRand;
use crate::util::*;

use ark_bls12_381::Bls12_381 as F;
type G1Projective = <F as Pairing>::G1;
type G2Projective = <F as Pairing>::G2;
//type G1Affine = <F as Pairing>::G1Affine;
//type G2Affine = <F as Pairing>::G2Affine;
//type GT = PairingOutput<F>;
type Fr = <F as Pairing>::ScalarField;

use groth_sahai::{
    prover::{
        commit_G1, commit_G2, commit_scalar_to_B1, commit_scalar_to_B2,
        //CProof, Commit1, Commit2, Provable,
    },
    AbstractCrs, B1, B2, BT, Com1, Com2, ComT, CRS,
};

pub fn bench_bls12_field_arith(c: &mut Criterion) {
    let mut g = c.benchmark_group("BLS12-381/Microbenchmarks");

    let mut rng = ark_std::test_rng();

    let r1 = Fr::rand(&mut rng);
    let r2 = Fr::rand(&mut rng);
    record_size("BLS12-381 scalar field size", &r1);

    g.bench_function("scalar field add", |b| {
        b.iter(|| {
            r1 + r2
        })
    });

    let s1 = Fr::rand(&mut rng);
    let s2 = Fr::rand(&mut rng);

    g.bench_function("scalar field multiply", |b| {
        b.iter(|| {
            s1 * s2
        })
    });

    g.finish();
}

// aka exponentiation (if framed as multiplicative group)
pub fn bench_bls12_scalar_mul(c: &mut Criterion) {
    let mut g = c.benchmark_group("BLS12-381/Microbenchmarks");

    let mut rng = ark_std::test_rng();

    let a1 = G1Projective::rand(&mut rng).into_affine();
    let r1 = Fr::rand(&mut rng);

    g.bench_function("G1 scalar multiply", |b| {
        b.iter(|| {
            a1 * r1
        })
    });

    let a2 = G2Projective::rand(&mut rng).into_affine();
    let r2 = Fr::rand(&mut rng);

    g.bench_function("G2 scalar multiply", |b| {
        b.iter(|| {
            a2 * r2
        })
    });

    let at = <F as Pairing>::pairing(a1, a2);
    let rt = Fr::rand(&mut rng);

    g.bench_function("GT scalar multiply", |b| {
        b.iter(|| {
            at * rt
        })
    });

    g.finish();
}

// group operation is represented by addition
pub fn bench_bls12_group_add(c: &mut Criterion) {
    let mut g = c.benchmark_group("BLS12-381/Microbenchmarks");

    let mut rng = ark_std::test_rng();

    let a1proj = G1Projective::rand(&mut rng);
    let a1 = a1proj.into_affine();
    let b1 = G1Projective::rand(&mut rng).into_affine();
    record_size("BLS12-381 G1 projective size", &a1proj);
    record_size("BLS12-381 G1 affine size", &a1);

    g.bench_function("G1 add", |b| {
        b.iter(|| {
            a1 + b1
        })
    });

    let a2proj = G2Projective::rand(&mut rng);
    let a2 = a2proj.into_affine();
    let b2 = G2Projective::rand(&mut rng).into_affine();
    record_size("BLS12-381 G2 projective size", &a2proj);
    record_size("BLS12-381 G2 affine size", &a2);

    g.bench_function("G2 add", |b| {
        b.iter(|| {
            a2 + b2
        })
    });

    let at = <F as Pairing>::pairing(a1, a2);
    let bt = <F as Pairing>::pairing(b1, b2);
    record_size("BLS12-381 GT size", &at);

    g.bench_function("GT add", |b| {
        b.iter(|| {
            at + bt
        })
    });

    g.finish();
}

pub fn bench_bls12_pairing_eval(c: &mut Criterion) {
    let mut g = c.benchmark_group("BLS12-381/Microbenchmarks");

    let mut rng = ark_std::test_rng();

    let a1 = G1Projective::rand(&mut rng).into_affine();
    let a2 = G2Projective::rand(&mut rng).into_affine();

    g.bench_function("pairing", |b| {
        b.iter(|| {
            <F as Pairing>::pairing(a1, a2)
        })
    });

    g.finish();
}

// Should be about 2x G1,G2; 4x GT
pub fn bench_bls12_commit_group_scalar_mul(c: &mut Criterion) {
    let mut g = c.benchmark_group("BLS12-381/Microbenchmarks");

    let mut rng = ark_std::test_rng();

    let a1 = G1Projective::rand(&mut rng).into_affine();
    let b1 = G1Projective::rand(&mut rng).into_affine();
    let c1 = Com1::<F>(a1, b1);
    let s1 = Fr::rand(&mut rng);

    g.bench_function("G1 commit group scalar multiply", |b| {
        b.iter(|| {
            c1.scalar_mul(&s1);
        });
    });

    let a2 = G2Projective::rand(&mut rng).into_affine();
    let b2 = G2Projective::rand(&mut rng).into_affine();
    let c2 = Com2::<F>(a2, b2);
    let s2 = Fr::rand(&mut rng);

    g.bench_function("G2 commit group scalar multiply", |b| {
        b.iter(|| {
            c2.scalar_mul(&s2);
        });
    });

    /*
    let ct = ComT::pairing(c1, c2);
    let st = Fr::rand(&mut rng);

    g.bench_function("GT commit group scalar multiply", |b| {
        b.iter(|| {
            ct.scalar_mul(&st);
        });
    });
    */

    g.finish();
}

// Should be about 2x G1,G2; 4x GT
// TODO: Figure out why G1 and G2 are so much slower (~26x and 10x)
pub fn bench_bls12_commit_group_add(c: &mut Criterion) {
    let mut g = c.benchmark_group("BLS12-381/Microbenchmarks");

    let mut rng = ark_std::test_rng();

    let a11 = G1Projective::rand(&mut rng).into_affine();
    let b11 = G1Projective::rand(&mut rng).into_affine();
    let c11 = Com1::<F>(a11, b11);
    let a12 = G1Projective::rand(&mut rng).into_affine();
    let b12 = G1Projective::rand(&mut rng).into_affine();
    let c12 = Com1::<F>(a12, b12);
    record_size("BLS12-381 Com1 size", &c11);

    g.bench_function("G1 commit group add", |b| {
        b.iter(|| {
            c11 + c12
        });
    });

    let a21 = G2Projective::rand(&mut rng).into_affine();
    let b21 = G2Projective::rand(&mut rng).into_affine();
    let c21 = Com2::<F>(a21, b21);
    let a22 = G2Projective::rand(&mut rng).into_affine();
    let b22 = G2Projective::rand(&mut rng).into_affine();
    let c22 = Com2::<F>(a22, b22);
    record_size("BLS12-381 Com2 size", &c21);

    g.bench_function("G2 commit group add", |b| {
        b.iter(|| {
            c21 + c22
        });
    });

    let ct1 = ComT::pairing(c11, c21);
    let ct2 = ComT::pairing(c12, c22);
    //record_size("BLS12-381 ComT size", &ct1);

    g.bench_function("GT commit group add", |b| {
        b.iter(|| {
            ct1 + ct2
        });
    });

    g.finish();
}

// Should be about 4x (G1,G2,GT) pairing
pub fn bench_bls12_commit_group_pairing_eval(c: &mut Criterion) {
    let mut g = c.benchmark_group("BLS12-381/Microbenchmarks");

    let mut rng = ark_std::test_rng();

    let a1 = G1Projective::rand(&mut rng).into_affine();
    let b1 = G1Projective::rand(&mut rng).into_affine();
    let c1 = Com1::<F>(a1, b1);

    let a2 = G2Projective::rand(&mut rng).into_affine();
    let b2 = G2Projective::rand(&mut rng).into_affine();
    let c2 = Com2::<F>(a2, b2);

    g.bench_function("GT commit group pairing", |b| {
        b.iter(|| {
            ComT::pairing(c1, c2);
        });
    });

    g.finish();
}

// Commit G1 should have 2x Com1 add, 2x Com1 scalar mul, 2x Fr randgen
pub fn bench_bls12_commit(c: &mut Criterion) {
    let mut g = c.benchmark_group("BLS12-381/Microbenchmarks");

    let mut rng = ark_std::test_rng();

    let crs = CRS::<F>::generate_crs(&mut rng);
    record_size("BLS12-381 GS CRS size", &crs);

    let a1 = G1Projective::rand(&mut rng).into_affine();

    g.bench_function("G1 commit", |b| {
        b.iter(|| {
           commit_G1(&a1, &crs, &mut rng);
        });
    });

    let s1 = Fr::rand(&mut rng);

    g.bench_function("scalar field to G1 commit", |b| {
        b.iter(|| {
           commit_scalar_to_B1(&s1, &crs, &mut rng);
        });
    });

    let a2 = G2Projective::rand(&mut rng).into_affine();

    g.bench_function("G2 commit", |b| {
        b.iter(|| {
           commit_G2(&a2, &crs, &mut rng);
        });
    });

    let s2 = Fr::rand(&mut rng);

    g.bench_function("scalar field to G2 commit", |b| {
        b.iter(|| {
           commit_scalar_to_B2(&s2, &crs, &mut rng);
        });
    });
}

criterion_group!(
    bls12_bilinear_group_arith,
    bench_bls12_field_arith,
    bench_bls12_scalar_mul,
    bench_bls12_group_add,
);

criterion_group! {
    name = bls12_pairing;
    config = Criterion::default().measurement_time(Duration::new(6, 0));
    targets =
        bench_bls12_pairing_eval,
}

criterion_group! {
    name = bls12_commit;
    config = Criterion::default().measurement_time(Duration::new(6, 500_000_000));
    targets =
        bench_bls12_commit_group_scalar_mul,
        bench_bls12_commit_group_add,
        bench_bls12_commit_group_pairing_eval,
        bench_bls12_commit,
}

criterion_main!(bls12_bilinear_group_arith, bls12_pairing, bls12_commit);
