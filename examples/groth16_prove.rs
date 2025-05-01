#![cfg(feature = "groth16")]
mod utils;

use ark_ec::pairing::Pairing;
use ark_bls12_381::Bls12_381;
use ark_std::{
    test_rng, UniformRand,
    rand::{
        RngCore, SeedableRng,
        prelude::StdRng,
    },
};
use ark_ff::{PrimeField, Field};
use ark_groth16::{
    Groth16,
    Proof as Groth16Proof,
    ProvingKey as Groth16ProvingKey,
    VerifyingKey as Groth16VerifyingKey,
    PreparedVerifyingKey as Groth16PreparedVerifyingKey,
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
use groth_sahai::{
    CRS,
    prover::{Commit1, Commit2, batch_commit_G1, batch_commit_G2},
    gadgets::groth16::*,
};
use crate::utils::*;

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

pub fn main() {
    let mut rng = StdRng::seed_from_u64(test_rng().next_u64());
    let crs = get_value::<CRS<Bls12_381>>("../examples/testdata/test_crs");
    let a = <Bls12_381 as Pairing>::ScalarField::from(1337u32);
    let b = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
    let c = a + b;
    let public_inputs = [a.clone()];
    let circuit = TestCircuit {a, b, c};
    let (pk, _, _) = get_value::<(
        Groth16ProvingKey<Bls12_381>,
        Groth16VerifyingKey<Bls12_381>,
        Groth16PreparedVerifyingKey<Bls12_381>,
    )>("../examples/testdata/test_Groth16_keys");
    let g16_proof = Groth16::<Bls12_381>::prove(&pk, circuit.clone(), &mut rng).unwrap();
    record_value::<Groth16Proof<Bls12_381>>("../examples/testdata/test_Groth16_Proof", &g16_proof);
}
