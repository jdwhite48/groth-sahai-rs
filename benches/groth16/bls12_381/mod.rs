#![cfg(feature = "groth16")]
use criterion::{criterion_group, criterion_main, Criterion};

use ark_bls12_381::{Bls12_381, G1Affine, G2Affine};
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

type G16 = Groth16::<Bls12_381>;
type G16Proof = Groth16Proof<Bls12_381>;
//type G16ProvingKey = Groth16ProvingKey<Bls12_381>;
type G16VerifyingKey = Groth16VerifyingKey<Bls12_381>;
type G16PreparedVerifyingKey = Groth16PreparedVerifyingKey<Bls12_381>;
type GSCrs = CRS::<Bls12_381>;
type Fr = <Bls12_381 as Pairing>::ScalarField;
type G1 = <Bls12_381 as Pairing>::G1;
//type G2 = <Bls12_381 as Pairing>::G2;
//type G1Affine = <F as Pairing>::G1Affine;
//type G2Affine = <F as Pairing>::G2Affine;
//type GT = PairingOutput<F>;

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

macro_rules! make_plain_g16_verify_bench {
    ($num_proofs: expr, $bench_name: ident) => {
        pub fn $bench_name(c: &mut Criterion) {

            let mut g = c.benchmark_group("BLS12-381/Groth16");

            let mut rng = StdRng::seed_from_u64(test_rng().next_u64());

            let a = Fr::from(1337u32);
            let b = Fr::rand(&mut rng);
            let c = a + b;
            let public_inputs = [a.clone()];
            let circuit = TestCircuit {a, b, c};


            let mut g16_proofs: Vec<(G16PreparedVerifyingKey, G16Proof)> = Vec::with_capacity($num_proofs);
            for _ in 0..$num_proofs {
                let (pk, vk) = G16::circuit_specific_setup(circuit.clone(), &mut rng).unwrap();
                let g16_proof = G16::prove(&pk, circuit.clone(), &mut rng).unwrap();
                let pvk = ark_groth16::prepare_verifying_key(&vk);
                g16_proofs.push((pvk, g16_proof));
            }
            if $num_proofs == 1 {
                record_size("BLS12-381 Groth16 prepared verification key size", &g16_proofs[0].0);
                record_size("BLS12-381 Groth16 proof size", &g16_proofs[0].1);
            }

            g.bench_function(format!("verify {} plain Groth16 equations", $num_proofs), |b| {
                b.iter(|| {
                    for i in 0..$num_proofs {
                        let _g16_verifies = G16::verify_with_processed_vk(
                            &g16_proofs[i].0,
                            &public_inputs,
                            &g16_proofs[i].1,
                        ).unwrap();
                    }
                })
            });

            g.finish();
        }
    }
}

macro_rules! make_gs_g16_commit_bench {
    ($num_proofs: expr, $bench_name: ident) => {
        pub fn $bench_name(c: &mut Criterion) {

            let mut g = c.benchmark_group("BLS12-381/Groth16");

            let mut rng = StdRng::seed_from_u64(test_rng().next_u64());

            let a = Fr::from(1337u32);
            let b = Fr::rand(&mut rng);
            let c = a + b;
            let public_inputs = [a.clone()];
            let circuit = TestCircuit {a, b, c};

            let mut xvars: Vec<G1Affine> = Vec::with_capacity(2*$num_proofs);
            let mut yvars: Vec<G2Affine> = Vec::with_capacity($num_proofs);
            for _ in 0..$num_proofs {
                let (pk, vk) = G16::circuit_specific_setup(circuit.clone(), &mut rng).unwrap();
                let g16_proof = G16::prove(&pk, circuit.clone(), &mut rng).unwrap();
                let pvk = ark_groth16::prepare_verifying_key(&vk);
                let verifies = G16::verify_proof(&pvk, &g16_proof, &public_inputs).unwrap();
                assert!(verifies);
                xvars.push(g16_proof.a);
                xvars.push(g16_proof.c);
                yvars.push(g16_proof.b);
            }
            // Now do a GS-over-canon-Groth16 and verify that (does not hide public inputs)
            let crs = GSCrs::generate_crs(&mut rng);

            g.bench_function(format!("commit {} proofs", $num_proofs), |b| {
                b.iter(|| {
                    let _xcoms = batch_commit_G1(&xvars, &crs, &mut rng);
                    let _ycoms = batch_commit_G2(&yvars, &crs, &mut rng);
                })
            });

            g.finish();
        }
    }
}

macro_rules! make_gs_g16_proof_bench {
    ($num_proofs: expr, $bench_name: ident) => {
        pub fn $bench_name(c: &mut Criterion) {

            let mut g = c.benchmark_group("BLS12-381/Groth16");

            let mut rng = StdRng::seed_from_u64(test_rng().next_u64());

            let a = Fr::from(1337u32);
            let b = Fr::rand(&mut rng);
            let c = a + b;
            let public_inputs = [a.clone()];
            let circuit = TestCircuit {a, b, c};


            let mut g16_proofs: Vec<(G16Proof, G16VerifyingKey, G1)> = Vec::with_capacity($num_proofs);
            for _ in 0..$num_proofs {
                let (pk, vk) = G16::circuit_specific_setup(circuit.clone(), &mut rng).unwrap();
                let g16_proof = G16::prove(&pk, circuit.clone(), &mut rng).unwrap();
                let pvk = ark_groth16::prepare_verifying_key(&vk);
                let prep_inputs = G16::prepare_inputs(&pvk, &public_inputs).unwrap();
                let verifies = G16::verify_proof(&pvk, &g16_proof, &public_inputs).unwrap();
                assert!(verifies);
                g16_proofs.push((g16_proof, vk, prep_inputs));
            }

            // TODO: some cursed re-storing entries as its references to deal with mid API
            let g16_refs: Vec<(&G16Proof, &G16VerifyingKey, &G1)> = g16_proofs.iter().map(|(pf, vk, pi)| (pf, vk, pi)).collect::<Vec<_>>();

            // Now do a GS-over-canon-Groth16 and verify that (does not hide public inputs)
            let crs = GSCrs::generate_crs(&mut rng);

            g.bench_function(format!("commit and prove {} equations satisfied", $num_proofs), |b| {
                b.iter(|| {
                    let _gs_proofs = prove_groth16_equations_sat::<Bls12_381, _>(
                        &g16_refs[..],
                        &crs,
                        &mut rng,
                    );
                })
            });

            g.finish();
        }
    }
}

macro_rules! make_gs_g16_verify_bench {
    ($num_proofs: expr, $bench_name: ident) => {
        pub fn $bench_name(c: &mut Criterion) {

            let mut g = c.benchmark_group("BLS12-381/Groth16");

            let mut rng = StdRng::seed_from_u64(test_rng().next_u64());

            let a = Fr::from(1337u32);
            let b = Fr::rand(&mut rng);
            let c = a + b;
            let public_inputs = [a.clone()];
            let circuit = TestCircuit {a, b, c};


            let mut g16_proofs: Vec<(G16Proof, G16VerifyingKey, G1)> = Vec::with_capacity($num_proofs);
            for _ in 0..$num_proofs {
                let (pk, vk) = G16::circuit_specific_setup(circuit.clone(), &mut rng).unwrap();
                let g16_proof = G16::prove(&pk, circuit.clone(), &mut rng).unwrap();
                let pvk = ark_groth16::prepare_verifying_key(&vk);
                let prep_inputs = G16::prepare_inputs(&pvk, &public_inputs).unwrap();
                let verifies = G16::verify_proof(&pvk, &g16_proof, &public_inputs).unwrap();
                assert!(verifies);
                g16_proofs.push((g16_proof, vk, prep_inputs));
            }

            // TODO: some cursed re-storing entries as its references to deal with mid API
            let g16_refs: Vec<(&G16Proof, &G16VerifyingKey, &G1)> = g16_proofs.iter().map(|(pf, vk, pi)| (pf, vk, pi)).collect::<Vec<_>>();
            let g16_verif_refs: Vec<(&G16VerifyingKey, &G1)> = g16_proofs.iter().map(|(_, vk, pi)| (vk, pi)).collect::<Vec<_>>();

            // Now do a GS-over-canon-Groth16 and verify that (does not hide public inputs)
            let crs = GSCrs::generate_crs(&mut rng);

            let gs_proofs = prove_groth16_equations_sat::<Bls12_381, _>(&g16_refs[..], &crs, &mut rng);
            if $num_proofs == 1 {
            }

            record_size(format!("BLS12-381 GS-over-Groth16 {} equation Com1 size", $num_proofs), &gs_proofs.xcoms.coms);
            record_size(format!("BLS12-381 GS-over-Groth16 {} equation Com2 size", $num_proofs), &gs_proofs.ycoms.coms);
            record_size(format!("BLS12-381 GS-over-Groth16 {} equation proof size", $num_proofs), &gs_proofs.equ_proofs);

            g.bench_function(format!("verify {} equations satisfied", $num_proofs), |b| {
                b.iter(|| {
                    let _gs_verifies = verify_groth16_equations_sat::<Bls12_381>(
                        &g16_verif_refs[..],
                        &gs_proofs,
                        &crs
                    );
                })
            });

            g.finish();
        }
    }
}

make_gs_g16_commit_bench!(1, bench_bls12_gs_over_groth16_commit_1);
make_gs_g16_commit_bench!(5, bench_bls12_gs_over_groth16_commit_5);
make_gs_g16_commit_bench!(10, bench_bls12_gs_over_groth16_commit_10);
make_gs_g16_commit_bench!(25, bench_bls12_gs_over_groth16_commit_25);
make_gs_g16_commit_bench!(50, bench_bls12_gs_over_groth16_commit_50);

criterion_group!(
    bls12_gs_over_groth16_commit,
    bench_bls12_gs_over_groth16_commit_1,
    bench_bls12_gs_over_groth16_commit_5,
    bench_bls12_gs_over_groth16_commit_10,
    bench_bls12_gs_over_groth16_commit_25,
    bench_bls12_gs_over_groth16_commit_50,
);

make_gs_g16_proof_bench!(1, bench_bls12_gs_over_groth16_proof_1);
make_gs_g16_proof_bench!(5, bench_bls12_gs_over_groth16_proof_5);
make_gs_g16_proof_bench!(10, bench_bls12_gs_over_groth16_proof_10);
make_gs_g16_proof_bench!(25, bench_bls12_gs_over_groth16_proof_25);
make_gs_g16_proof_bench!(50, bench_bls12_gs_over_groth16_proof_50);

criterion_group!(
    bls12_gs_over_groth16_proof,
    bench_bls12_gs_over_groth16_proof_1,
    bench_bls12_gs_over_groth16_proof_5,
    bench_bls12_gs_over_groth16_proof_10,
    bench_bls12_gs_over_groth16_proof_25,
    bench_bls12_gs_over_groth16_proof_50,
);

make_gs_g16_verify_bench!(1, bench_bls12_gs_over_groth16_verify_1);
make_gs_g16_verify_bench!(5, bench_bls12_gs_over_groth16_verify_5);
make_gs_g16_verify_bench!(10, bench_bls12_gs_over_groth16_verify_10);
make_gs_g16_verify_bench!(25, bench_bls12_gs_over_groth16_verify_25);
make_gs_g16_verify_bench!(50, bench_bls12_gs_over_groth16_verify_50);

criterion_group!(
    bls12_gs_over_groth16_verify,
    bench_bls12_gs_over_groth16_verify_1,
    bench_bls12_gs_over_groth16_verify_5,
    bench_bls12_gs_over_groth16_verify_10,
    bench_bls12_gs_over_groth16_verify_25,
    bench_bls12_gs_over_groth16_verify_50,
);

make_plain_g16_verify_bench!(1, bench_bls12_plain_groth16_verify_1);
make_plain_g16_verify_bench!(5, bench_bls12_plain_groth16_verify_5);
make_plain_g16_verify_bench!(10, bench_bls12_plain_groth16_verify_10);
make_plain_g16_verify_bench!(25, bench_bls12_plain_groth16_verify_25);
make_plain_g16_verify_bench!(50, bench_bls12_plain_groth16_verify_50);

criterion_group!(
    bls12_plain_groth16_verify,
    bench_bls12_plain_groth16_verify_1,
    bench_bls12_plain_groth16_verify_5,
    bench_bls12_plain_groth16_verify_10,
    bench_bls12_plain_groth16_verify_25,
    bench_bls12_plain_groth16_verify_50,
);

criterion_main!(bls12_gs_over_groth16_commit, bls12_gs_over_groth16_proof, bls12_gs_over_groth16_verify, bls12_plain_groth16_verify);
