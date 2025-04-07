use criterion::{criterion_group, criterion_main, Criterion};

//use std::time::Duration;

use ark_ec::{
    pairing::Pairing,
    CurveGroup,
};
use ark_ff::UniformRand;

use crate::util::*;

use ark_bls12_381::Bls12_381;
use ark_bn254::Bn254;

use groth_sahai::{
    prover::{
        commit_G1, commit_G2, commit_scalar_to_B1, commit_scalar_to_B2,
        //CProof, Commit1, Commit2, Provable,
    },
    AbstractCrs, B1, B2, BT, Com1, Com2, ComT, CRS,
};


macro_rules! make_field_benches {
    ($field_str: expr, $field_type: ident, $field_microbench_name: ident, $bilinear_arith_bench_name: ident, $commit_arith_bench_name: ident, $commit_bench_name: ident) => {

        pub fn $bilinear_arith_bench_name(c: &mut Criterion) {
            let mut g = c.benchmark_group(format!("{}/Microbenchmarks", $field_str));

            let mut rng = ark_std::test_rng();

            // scalar field arithmetic
            let r1 = <$field_type as Pairing>::ScalarField::rand(&mut rng);
            let r2 = <$field_type as Pairing>::ScalarField::rand(&mut rng);
            record_size(format!("{} scalar field size", $field_str), &r1);
            g.bench_function("scalar field add", |b| {
                b.iter(|| {
                    r1 + r2
                })
            });

            g.bench_function("scalar field multiply", |b| {
                b.iter(|| {
                    r1 * r2
                })
            });

            // cf. scalar multiplication (if additive group) = group exponentiation (if multiplicative group)
            let a1proj = <$field_type as Pairing>::G1::rand(&mut rng);
            let a1 = a1proj.into_affine();
            record_size(format!("{} G1 projective size", $field_str), &a1proj);
            record_size(format!("{} G1 affine size", $field_str), &a1);
            g.bench_function("G1 scalar multiply", |b| {
                b.iter(|| {
                    a1 * r1
                })
            });

            let a2proj = <$field_type as Pairing>::G2::rand(&mut rng);
            let a2 = a2proj.into_affine();
            record_size(format!("{} G2 projective size", $field_str), &a2proj);
            record_size(format!("{} G2 affine size", $field_str), &a2);
            g.bench_function("G2 scalar multiply", |b| {
                b.iter(|| {
                    a2 * r2
                })
            });

            let at = <$field_type as Pairing>::pairing(a1, a2);
            let rt = <$field_type as Pairing>::ScalarField::rand(&mut rng);
            record_size(format!("{} GT size", $field_str), &at);
            g.bench_function("GT scalar multiply", |b| {
                b.iter(|| {
                    at * rt
                })
            });

            let b1 = <$field_type as Pairing>::G1::rand(&mut rng).into_affine();
            g.bench_function("G1 add", |b| {
                b.iter(|| {
                    a1 + b1
                })
            });

            let b2 = <$field_type as Pairing>::G2::rand(&mut rng).into_affine();
            g.bench_function("G2 add", |b| {
                b.iter(|| {
                    a2 + b2
                })
            });

            let bt = <$field_type as Pairing>::pairing(b1, b2);
            g.bench_function("GT add", |b| {
                b.iter(|| {
                    at + bt
                })
            });

            g.bench_function("pairing", |b| {
                b.iter(|| {
                    <$field_type as Pairing>::pairing(a1, a2)
                })
            });

            g.finish();
        }


        pub fn $commit_arith_bench_name(c: &mut Criterion) {
            let mut g = c.benchmark_group(format!("{}/Microbenchmarks", $field_str));

            let mut rng = ark_std::test_rng();

            let a11 = <$field_type as Pairing>::G1::rand(&mut rng).into_affine();
            let b11 = <$field_type as Pairing>::G1::rand(&mut rng).into_affine();
            let c11 = Com1::<$field_type>(a11, b11);
            record_size(format!("{} Com1 size", $field_str), &c11);
            let a21 = <$field_type as Pairing>::G2::rand(&mut rng).into_affine();
            let b21 = <$field_type as Pairing>::G2::rand(&mut rng).into_affine();
            let c21 = Com2::<$field_type>(a21, b21);
            record_size(format!("{} Com2 size", $field_str), &c21);

            let s1 = <$field_type as Pairing>::ScalarField::rand(&mut rng);
            g.bench_function("G1 commit group scalar multiply", |b| {
                b.iter(|| {
                    c11.scalar_mul(&s1);
                });
            });

            let s2 = <$field_type as Pairing>::ScalarField::rand(&mut rng);
            g.bench_function("G2 commit group scalar multiply", |b| {
                b.iter(|| {
                    c21.scalar_mul(&s2);
                });
            });

            /*
            let ct = ComT::pairing(c1, c2);
            let st = <$field_type as Pairing>::ScalarField::rand(&mut rng);
            g.bench_function("GT commit group scalar multiply", |b| {
                b.iter(|| {
                    ct.scalar_mul(&st);
                });
            });
            */

            let a12 = <$field_type as Pairing>::G1::rand(&mut rng).into_affine();
            let b12 = <$field_type as Pairing>::G1::rand(&mut rng).into_affine();
            let c12 = Com1::<$field_type>(a12, b12);
            let a22 = <$field_type as Pairing>::G2::rand(&mut rng).into_affine();
            let b22 = <$field_type as Pairing>::G2::rand(&mut rng).into_affine();
            let c22 = Com2::<$field_type>(a22, b22);
            // TODO: Figure out why G1 and G2 are so much slower (~26x and 10x)
            // Should be about 2x G1 add
            g.bench_function("G1 commit group add", |b| {
                b.iter(|| {
                    c11 + c12
                });
            });

            // Should be about 2x G2 add
            g.bench_function("G2 commit group add", |b| {
                b.iter(|| {
                    c21 + c22
                });
            });

            // Should be about 4x (G1,G2,GT) pairing
            let ct1 = ComT::pairing(c11, c21);
            let ct2 = ComT::pairing(c12, c22);
            //record_size(format!("{} ComT size", $field_str), &ct1);
            g.bench_function("GT commit group pairing", |b| {
                b.iter(|| {
                    ComT::pairing(c11, c21);
                });
            });

            // Should be about 4x GT add
            g.bench_function("GT commit group add", |b| {
                b.iter(|| {
                    ct1 + ct2
                });
            });
        }

        // Commit G1 should have 2x Com1 add, 2x Com1 scalar mul, 2x <$field_type as Pairing>::ScalarField randgen
        pub fn $commit_bench_name(c: &mut Criterion) {
            let mut g = c.benchmark_group(format!("{}/Microbenchmarks", $field_str));

            let mut rng = ark_std::test_rng();

            let crs = CRS::<$field_type>::generate_crs(&mut rng);
            record_size(format!("{} GS CRS size", $field_str), &crs);

            let a1 = <$field_type as Pairing>::G1::rand(&mut rng).into_affine();

            g.bench_function("G1 commit", |b| {
                b.iter(|| {
                   commit_G1(&a1, &crs, &mut rng);
                });
            });

            let s1 = <$field_type as Pairing>::ScalarField::rand(&mut rng);

            g.bench_function("scalar field to G1 commit", |b| {
                b.iter(|| {
                   commit_scalar_to_B1(&s1, &crs, &mut rng);
                });
            });

            let a2 = <$field_type as Pairing>::G2::rand(&mut rng).into_affine();

            g.bench_function("G2 commit", |b| {
                b.iter(|| {
                   commit_G2(&a2, &crs, &mut rng);
                });
            });

            let s2 = <$field_type as Pairing>::ScalarField::rand(&mut rng);

            g.bench_function("scalar field to G2 commit", |b| {
                b.iter(|| {
                   commit_scalar_to_B2(&s2, &crs, &mut rng);
                });
            });
        }

        criterion_group! {
            name = $field_microbench_name;
            config = Criterion::default();//.measurement_time(Duration::new(5, 0));
            targets =
                $bilinear_arith_bench_name,
                $commit_arith_bench_name,
                $commit_bench_name,
        }
    }
}

make_field_benches!("BLS12-381", Bls12_381, bls12_microbenches, bench_bls12_bilinear_arith, bench_bls12_commit_arith, bench_bls12_commit);
make_field_benches!("BN254", Bn254, bn254_microbenches, bench_bn254_bilinear_arith, bench_bn254_commit_arith, bench_bn254_commit);

criterion_main!(bls12_microbenches, bn254_microbenches);
