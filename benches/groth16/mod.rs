#![cfg(feature = "groth16")]
use criterion::{criterion_group, criterion_main, Criterion};

use ark_bls12_381::Bls12_381;
use ark_bn254::Bn254;

use ark_ff::{PrimeField, Field};
use ark_groth16::{
    Groth16,
    Proof as Groth16Proof,
    VerifyingKey as Groth16VerifyingKey,
    PreparedVerifyingKey as Groth16PreparedVerifyingKey,
};

use ark_std::{
    test_rng, UniformRand,
    rand::{
        RngCore, SeedableRng,
        prelude::StdRng,
    },
};
use ark_snark::SNARK;
use ark_relations::{
    ns,
    r1cs::{
        ConstraintSynthesizer,
        ConstraintSystemRef,
        SynthesisError,
    },
};
use ark_r1cs_std::{
    fields::fp::FpVar,
    prelude::{EqGadget, AllocVar},
};
use ark_ec::{
    pairing::Pairing, // GT is of type PairingOutput
    // AffineRepr, CurveGroup,
};

use groth_sahai::{
    CRS,
    prover::{batch_commit_G1, batch_commit_G2},
    generator::AbstractCrs,
};
#[cfg(feature = "groth16")]
use groth_sahai::gadgets::groth16::*;
use crate::util::*;

#[derive(Clone)]
// Groth16 zkSNARK circuit as proven with GS is irrelevant; just verify that a + b = c for simplicity, with public input a
pub struct TestCircuit<ConstraintF: Field>{
    a: ConstraintF,
    b: ConstraintF,
    c: ConstraintF,
}
impl<ConstraintF: PrimeField> ConstraintSynthesizer<ConstraintF>
    for TestCircuit<ConstraintF> {

    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        let a = FpVar::<ConstraintF>::new_input(ns!(cs, "a"), || Ok(self.a))?;
        let b = FpVar::<ConstraintF>::new_witness(cs.clone(), || Ok(self.b))?;
        let c = FpVar::<ConstraintF>::new_witness(cs.clone(), || Ok(self.c))?;
        let c2 = a + b;
        c.enforce_equal(&c2)?;
        Ok(())
    }
}

const NUM_ITERATIONS: [usize; 5] = [1, 5, 10, 25, 50];

macro_rules! make_gs_over_g16_benches {
    ($field_str: expr, $field_type: ident, $gs_bench_group_name: ident, $commit_bench_name: ident, $proof_bench_name: ident, $gs_verify_bench_name: ident, $g16_verify_bench_name: ident) => {

        macro_rules! make_plain_g16_verify_bench {
            ($num_proofs: expr, $bench_name: ident) => {
                pub fn $bench_name(c: &mut Criterion) {

                    let mut g = c.benchmark_group(format!("{}/Groth16", $field_str));

                    let mut rng = StdRng::seed_from_u64(test_rng().next_u64());

                    let a = <$field_type as Pairing>::ScalarField::from(1337u32);
                    let b = <$field_type as Pairing>::ScalarField::rand(&mut rng);
                    let c = a + b;
                    let public_inputs = [a.clone()];
                    let circuit = TestCircuit {a, b, c};

                    let max_num_proofs = *$num_proofs.iter().max().unwrap();
                    let mut g16_proofs: Vec<(Groth16PreparedVerifyingKey<$field_type>, Groth16Proof<$field_type>)> = Vec::with_capacity(max_num_proofs);
                    for _ in 0..max_num_proofs {
                        let (pk, vk) = Groth16::<$field_type>::circuit_specific_setup(circuit.clone(), &mut rng).unwrap();
                        let g16_proof = Groth16::<$field_type>::prove(&pk, circuit.clone(), &mut rng).unwrap();
                        let pvk = ark_groth16::prepare_verifying_key(&vk);
                        g16_proofs.push((pvk, g16_proof));
                    }

                    for num_equs in $num_proofs.iter() {
                        if *num_equs == 1 {
                            record_size(format!("{} Groth16 prepared verification key size", $field_str), &g16_proofs[0].0);
                            record_size(format!("{} Groth16 proof size", $field_str), &g16_proofs[0].1);
                        }
                        let g16_proofs_n: Vec<(Groth16PreparedVerifyingKey<$field_type>, Groth16Proof<$field_type>)> = g16_proofs.clone().into_iter().take(*num_equs).collect();
                        g.bench_function(format!("verify {} plain Groth16 equations", *num_equs), |b| {
                            b.iter(|| {
                                for i in 0..*num_equs {
                                    let _g16_verifies = Groth16::<$field_type>::verify_with_processed_vk(
                                        &g16_proofs_n[i].0,
                                        &public_inputs,
                                        &g16_proofs_n[i].1,
                                    ).unwrap();
                                }
                            })
                        });
                    }

                    g.finish();
                }
            }
        }

        macro_rules! make_gs_over_g16_commit_bench {
            ($num_proofs: expr, $bench_name: ident) => {
                pub fn $bench_name(c: &mut Criterion) {

                    let mut g = c.benchmark_group(format!("{}/Groth16", $field_str));

                    let mut rng = StdRng::seed_from_u64(test_rng().next_u64());

                    let a = <$field_type as Pairing>::ScalarField::from(1337u32);
                    let b = <$field_type as Pairing>::ScalarField::rand(&mut rng);
                    let c = a + b;
                    let public_inputs = [a.clone()];
                    let circuit = TestCircuit {a, b, c};

                    let max_num_proofs = *$num_proofs.iter().max().unwrap();
                    let mut xvars: Vec<<$field_type as Pairing>::G1Affine> = Vec::with_capacity(2*max_num_proofs);
                    let mut yvars: Vec<<$field_type as Pairing>::G2Affine> = Vec::with_capacity(max_num_proofs);
                    for _ in 0..max_num_proofs {
                        let (pk, vk) = Groth16::<$field_type>::circuit_specific_setup(circuit.clone(), &mut rng).unwrap();
                        let g16_proof = Groth16::<$field_type>::prove(&pk, circuit.clone(), &mut rng).unwrap();
                        let pvk = ark_groth16::prepare_verifying_key(&vk);
                        let verifies = Groth16::<$field_type>::verify_proof(&pvk, &g16_proof, &public_inputs).unwrap();
                        assert!(verifies);
                        xvars.push(g16_proof.a);
                        xvars.push(g16_proof.c);
                        yvars.push(g16_proof.b);
                    }
                    // Now do a GS-over-canon-Groth16 and verify that (does not hide public inputs)
                    let crs = CRS::<$field_type>::generate_crs(&mut rng);

                    for num_equs in $num_proofs.iter() {
                        let xvars_n: Vec<<$field_type as Pairing>::G1Affine> = xvars.clone().into_iter().take(*num_equs).collect();
                        let yvars_n: Vec<<$field_type as Pairing>::G2Affine> = yvars.clone().into_iter().take(*num_equs).collect();
                        g.bench_function(format!("commit {} proofs", *num_equs), |b| {
                            b.iter(|| {
                                let _xcoms = batch_commit_G1(&xvars_n, &crs, &mut rng);
                                let _ycoms = batch_commit_G2(&yvars_n, &crs, &mut rng);
                            })
                        });
                    }

                    g.finish();
                }
            }
        }

        macro_rules! make_gs_over_g16_proof_bench {
            ($num_proofs: expr, $bench_name: ident) => {
                pub fn $bench_name(c: &mut Criterion) {

                    let mut g = c.benchmark_group(format!("{}/Groth16", $field_str));

                    let mut rng = StdRng::seed_from_u64(test_rng().next_u64());

                    let a = <$field_type as Pairing>::ScalarField::from(1337u32);
                    let b = <$field_type as Pairing>::ScalarField::rand(&mut rng);
                    let c = a + b;
                    let public_inputs = [a.clone()];
                    let circuit = TestCircuit {a, b, c};


                    let max_num_proofs = *$num_proofs.iter().max().unwrap();
                    let mut g16_proofs: Vec<(Groth16Proof<$field_type>, Groth16VerifyingKey<$field_type>, <$field_type as Pairing>::G1)> = Vec::with_capacity(max_num_proofs);
                    for _ in 0..max_num_proofs {
                        let (pk, vk) = Groth16::<$field_type>::circuit_specific_setup(circuit.clone(), &mut rng).unwrap();
                        let g16_proof = Groth16::<$field_type>::prove(&pk, circuit.clone(), &mut rng).unwrap();
                        let pvk = ark_groth16::prepare_verifying_key(&vk);
                        let prep_inputs = Groth16::<$field_type>::prepare_inputs(&pvk, &public_inputs).unwrap();
                        let verifies = Groth16::<$field_type>::verify_proof(&pvk, &g16_proof, &public_inputs).unwrap();
                        assert!(verifies);
                        g16_proofs.push((g16_proof, vk, prep_inputs));
                    }

                    // TODO: some cursed re-storing entries as its references to deal with mid API
                    let g16_refs: Vec<(&Groth16Proof<$field_type>, &Groth16VerifyingKey<$field_type>, &<$field_type as Pairing>::G1)> = g16_proofs.iter().map(|(pf, vk, pi)| (pf, vk, pi)).collect::<Vec<_>>();

                    // Now do a GS-over-canon-Groth16 and verify that (does not hide public inputs)
                    let crs = CRS::<$field_type>::generate_crs(&mut rng);

                    for num_equs in $num_proofs.iter() {
                        let g16_refs_n: Vec<(&Groth16Proof<$field_type>, &Groth16VerifyingKey<$field_type>, &<$field_type as Pairing>::G1)> = g16_refs.clone().into_iter().take(*num_equs).collect();
                        g.bench_function(format!("commit and prove {} equations satisfied", *num_equs), |b| {
                            b.iter(|| {
                                let _gs_proofs = prove_groth16_equations_sat::<$field_type, _>(
                                    g16_refs_n.as_slice(),
                                    &crs,
                                    &mut rng,
                                );
                            })
                        });
                    }

                    g.finish();
                }
            }
        }

        macro_rules! make_gs_over_g16_verify_bench {
            ($num_proofs: expr, $bench_name: ident) => {
                pub fn $bench_name(c: &mut Criterion) {

                    let mut g = c.benchmark_group(format!("{}/Groth16", $field_str));

                    let mut rng = StdRng::seed_from_u64(test_rng().next_u64());

                    let a = <$field_type as Pairing>::ScalarField::from(1337u32);
                    let b = <$field_type as Pairing>::ScalarField::rand(&mut rng);
                    let c = a + b;
                    let public_inputs = [a.clone()];
                    let circuit = TestCircuit {a, b, c};

                    let max_num_proofs = *$num_proofs.iter().max().unwrap();
                    let mut g16_proofs: Vec<(Groth16Proof<$field_type>, Groth16VerifyingKey<$field_type>, <$field_type as Pairing>::G1)> = Vec::with_capacity(max_num_proofs);
                    for _ in 0..max_num_proofs {
                        let (pk, vk) = Groth16::<$field_type>::circuit_specific_setup(circuit.clone(), &mut rng).unwrap();
                        let g16_proof = Groth16::<$field_type>::prove(&pk, circuit.clone(), &mut rng).unwrap();
                        let pvk = ark_groth16::prepare_verifying_key(&vk);
                        let prep_inputs = Groth16::<$field_type>::prepare_inputs(&pvk, &public_inputs).unwrap();
                        let verifies = Groth16::<$field_type>::verify_proof(&pvk, &g16_proof, &public_inputs).unwrap();
                        assert!(verifies);
                        g16_proofs.push((g16_proof, vk, prep_inputs));
                    }

                    // TODO: some cursed re-storing entries as its references to deal with mid API
                    let g16_refs: Vec<(&Groth16Proof<$field_type>, &Groth16VerifyingKey<$field_type>, &<$field_type as Pairing>::G1)> = g16_proofs.iter().map(|(pf, vk, pi)| (pf, vk, pi)).collect::<Vec<_>>();
                    let g16_verif_refs: Vec<(&Groth16VerifyingKey<$field_type>, &<$field_type as Pairing>::G1)> = g16_proofs.iter().map(|(_, vk, pi)| (vk, pi)).collect::<Vec<_>>();

                    // Now do a GS-over-canon-Groth16 and verify that (does not hide public inputs)
                    let crs = CRS::<$field_type>::generate_crs(&mut rng);

                    for num_equs in $num_proofs.iter() {

                        let g16_refs_n: Vec<(&Groth16Proof<$field_type>, &Groth16VerifyingKey<$field_type>, &<$field_type as Pairing>::G1)> = g16_refs.clone().into_iter().take(*num_equs).collect();
                        let g16_verif_refs_n: Vec<(&Groth16VerifyingKey<$field_type>, &<$field_type as Pairing>::G1)> = g16_verif_refs.clone().into_iter().take(*num_equs).collect();
                        let gs_proofs_n = prove_groth16_equations_sat::<$field_type, _>(g16_refs_n.as_slice(), &crs, &mut rng);
                        record_size(format!("{} GS-over-Groth16 {} equation Com1 size", $field_str, *num_equs), &gs_proofs_n.xcoms.coms);
                        record_size(format!("{} GS-over-Groth16 {} equation Com2 size", $field_str, *num_equs), &gs_proofs_n.ycoms.coms);
                        record_size(format!("{} GS-over-Groth16 {} equation proof size", $field_str, *num_equs), &gs_proofs_n.equ_proofs);

                        g.bench_function(format!("verify {} equations satisfied", *num_equs), |b| {
                            b.iter(|| {
                                let _gs_verifies = verify_groth16_equations_sat::<$field_type>(
                                    &g16_verif_refs_n.as_slice(),
                                    &gs_proofs_n,
                                    &crs
                                );
                            })
                        });
                    }

                    g.finish();
                }
            }
        }

        make_gs_over_g16_commit_bench!(&NUM_ITERATIONS, $commit_bench_name);
        make_gs_over_g16_proof_bench!(&NUM_ITERATIONS, $proof_bench_name);
        make_plain_g16_verify_bench!(&NUM_ITERATIONS, $g16_verify_bench_name);
        make_gs_over_g16_verify_bench!(&NUM_ITERATIONS, $gs_verify_bench_name);

        criterion_group!(
            $gs_bench_group_name,
            $commit_bench_name,
            $proof_bench_name,
            $g16_verify_bench_name,
            $gs_verify_bench_name,
        );
    }
}

make_gs_over_g16_benches!("BLS12-381", Bls12_381, bls12_gs_over_groth16, bench_bls12_gs_over_groth16_commit, bench_bls12_gs_over_groth16_proof, bench_bls12_groth16_plain_verify, bench_bls12_gs_over_groth16_verify);
make_gs_over_g16_benches!("BN254", Bn254, bn254_gs_over_groth16, bench_bn254_gs_over_groth16_commit, bench_bn254_gs_over_groth16_proof, bench_bn254_groth16_plain_verify, bench_bn254_gs_over_groth16_verify);

criterion_main!(bls12_gs_over_groth16, bn254_gs_over_groth16);
