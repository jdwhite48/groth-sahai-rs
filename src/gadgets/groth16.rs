#![cfg(feature = "groth16")]
//! **NOTE**: The Groth-Sahai CRS is meant to be generated as a part of trusted setup or trusted
//! computation. CRS generation is left up to the user depending on the use-case.
//! See [crate::generator](https://github.com/jdwhite84/groth-sahai-rs/blob/main/src/generator.rs) for more details.

use ark_ec::{
    pairing::Pairing,
    AffineRepr, CurveGroup,
};
use ark_ff::{One, Zero};
use ark_groth16::{Proof, VerifyingKey};
use ark_std::rand::{CryptoRng, Rng};

use crate::{
    Matrix, CRS,
    prover::{batch_commit_G1, batch_commit_G2, Commit1, Commit2, CProof, EquProof, Provable},
    verifier::Verifiable,
    statement::PPE,
};

/// Prove that a list of canonical Groth16 verification equations are all satisfied by the provided
/// `Proof`.
///
/// The G1 element, `prepared_pub_input`, is expected to be `Σ_{i=0}^{\ell [p]} a_p,i W_p,i`
/// where `W_p,i` corresponds to the `p`th `VerifyingKey`'s `gamma_abc_g1[i]`.
pub fn prove_groth16_equations_sat<E, R>(
    g16_pf_elems: &[(&Proof<E>, &VerifyingKey<E>, &E::G1)],
    gs_crs: &CRS<E>,
    rng: &mut R,
) -> CProof<E>
where
    R: Rng + CryptoRng,
    E: Pairing,
{
    let num_equ = g16_pf_elems.len();

    // TODO: destructure tuple vector in a saner way
    // Construct the GS statement, i.e. Groth16 verification equations
    let g16_ver_equs = prepare_groth16_equations_sat::<E>(
        &g16_pf_elems
            .iter()
            .map(|(_, vk, pub_input)| (*vk, *pub_input))
            .collect::<Vec<_>>(),
    );

    // X = [ ..., A_p, C_p, ... ]
    let m = 2 * num_equ;
    let mut xvars: Vec<E::G1Affine> = Vec::with_capacity(m);
    for (pf, _, _) in g16_pf_elems.iter() {
        xvars.append(&mut vec![pf.a, pf.c]);
    }
    // Y = [ ..., B_p, ... ]
    let yvars: Vec<E::G2Affine> = g16_pf_elems
        .iter()
        .map(|(pf, _, _)| pf.b)
        .collect::<Vec<E::G2Affine>>();
    // Commit to these variables in GS
    let xcoms: Commit1<E> = batch_commit_G1(&xvars, gs_crs, rng);
    let ycoms: Commit2<E> = batch_commit_G2(&yvars, gs_crs, rng);

    CProof::<E> {
        xcoms: xcoms.clone(),
        ycoms: ycoms.clone(),
        equ_proofs: g16_ver_equs
            .iter()
            .map(|equ| equ.prove(&xvars, &yvars, &xcoms, &ycoms, gs_crs, rng))
            .collect::<Vec<EquProof<E>>>(),
    }
}

/// Verify that a list of canonical Groth16 verification equations are all satisfied by the Groth-Sahai
/// `EquProof`.
///
/// The G1 element, `prepared_pub_input`, is expected to be `Σ_{i=0}^{\ell [p]} a_p,i W_p,i`
/// where `W_p,i` corresponds to the `p`th `VerifyingKey`'s `gamma_abc_g1[i]`.
#[must_use]
pub fn verify_groth16_equations_sat<E: Pairing>(
    g16_ver_elems: &[(&VerifyingKey<E>, &E::G1)],
    gs_proofs: &CProof<E>,
    gs_crs: &CRS<E>,
) -> bool {
    let num_equ = g16_ver_elems.len();
    if (*gs_proofs).equ_proofs.len() != num_equ {
        return false;
    }

    // Reconstruct the GS statement (i.e. equations) the prover should have used
    let g16_ver_equs: Vec<PPE<E>> = prepare_groth16_equations_sat::<E>(g16_ver_elems);
    if g16_ver_equs.len() != num_equ {
        return false;
    }

    for (_, g16_equ) in g16_ver_equs.iter().enumerate() {
        let verifies: bool = g16_equ.verify(gs_proofs, gs_crs);
        if !verifies {
            return false;
        }
    }
    // Don't allow an empty or trivial proof to verify. Otherwise, if it's proceeded to this point,
    // all GS-over-Groth16 equations successfully satisfied
    !g16_ver_equs.is_empty() && !(*gs_proofs).equ_proofs.is_empty()
}

/// Expressed in the form of a GS statement, a canonical Groth16 verification equation has the form:
/// `e( C, -vk.delta_g2 ) * e(A, B) = e( vk.alpha_g1, vk.beta_g2 ) * e( prepared_pub_input,
/// vk.gamma_g2 )` where (A,B,C) are the Groth16 proof elements / GS witness variables, and the
/// rest are public constants.
///
/// The G1 element, `prepared_pub_input`, is expected to be `Σ_{i=0}^{\ell [p]} a_p,i W_p,i`
/// where `W_p,i` corresponds to the `p`th `VerifyingKey`'s `gamma_abc_g1[i]`.
pub fn prepare_groth16_equations_sat<E: Pairing>(
    g16_elems: &[(&VerifyingKey<E>, &E::G1)],
) -> Vec<PPE<E>> {
    let num_equ = g16_elems.len();
    // The number of G1 variables: A, C in `Proof`
    let m = 2 * num_equ;
    // The number of G2 variables: B in `Proof`
    let n = num_equ;

    let mut gs_equs: Vec<PPE<E>> = Vec::with_capacity(num_equ);

    for (p, (vk, pub_input)) in g16_elems.iter().enumerate() {
        // `m` x `n` matrix defining how the variables `X = [..., A_p, C_p, ...]`, `Y = [..., B_p, ...]` are paired
        let mut gs_gamma: Matrix<E::ScalarField> = Vec::with_capacity(m);

        // not paired with previous equations' variables
        for _ in 0..p {
            gs_gamma.push(vec![E::ScalarField::zero(); n]);
            gs_gamma.push(vec![E::ScalarField::zero(); n]);
        }

        // Add a 1 at (2p, p) corresponding to the variable pairing e(A_p, B_p) in equation
        let mut ab_row = vec![E::ScalarField::zero(); n];
        ab_row[p] = E::ScalarField::one();
        gs_gamma.push(ab_row);
        gs_gamma.push(vec![E::ScalarField::zero(); n]);

        // not paired with next equations' variables
        for _ in (p + 1)..num_equ {
            gs_gamma.push(vec![E::ScalarField::zero(); n]);
            gs_gamma.push(vec![E::ScalarField::zero(); n]);
        }
        assert_eq!(gs_gamma.len(), m);
        assert_eq!(gs_gamma[0].len(), n);

        // Constants paired with none of the `n` variables in `G2`
        let gs_a_consts: Vec<E::G1Affine> = vec![E::G1Affine::zero(); n];
        // Constants paired with only second variable associated with each equation, i.e. e(C_p, - vk[p].delta_g2)
        let mut gs_b_consts: Vec<E::G2Affine> = vec![E::G2Affine::zero(); m];
        gs_b_consts[2 * p + 1] = (-vk.delta_g2.into_group()).into_affine();

        let gs_rhs = E::pairing(vk.alpha_g1, vk.beta_g2) + E::pairing(pub_input.into_affine(), vk.gamma_g2);

        // Add the `p`th Groth16 verification equation to the list of Groth-Sahai equations
        gs_equs.push(PPE::<E> {
            a_consts: gs_a_consts,
            b_consts: gs_b_consts,
            gamma: gs_gamma,
            target: gs_rhs,
        });
    }

    gs_equs
}

#[cfg(test)]
mod tests {
    use super::*;

    use ark_bls12_381::Bls12_381;
    use ark_ff::{PrimeField, Field};
    use ark_groth16::Groth16;

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

    use crate::{CRS, generator::AbstractCrs};

    pub type G16 = Groth16::<Bls12_381>;
    pub type GSCrs = CRS::<Bls12_381>;
    pub type Fr = <Bls12_381 as Pairing>::ScalarField;

    /*
    type G16 = Groth16::<Bls12_381>;
    type G16ProvingKey = <G16 as SNARK<BlsFr>>::ProvingKey;
    type G16VerifyingKey = <G16 as SNARK<BlsFr>>::VerifyingKey;
    type G16Proof = <G16 as SNARK<BlsFr>>::Proof;
    */

    #[derive(Clone)]
    // Groth16 zkSNARK circuit as proven with GS is irrelevant; just verify that a + b = c for simplicity, with public input a
    pub struct TestCircuit<ConstraintF: Field>{
        a: ConstraintF,
        b: ConstraintF,
        c: ConstraintF,
    }
    impl<ConstraintF: PrimeField> ConstraintSynthesizer<ConstraintF>
        for TestCircuit<ConstraintF>
    {
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

    #[test]
    fn gs_over_groth16_correctness() {
        let mut rng = StdRng::seed_from_u64(test_rng().next_u64());

        let a = Fr::from(1337u32);
        let b = Fr::rand(&mut rng);
        let c = a + b;
        let public_inputs = [a.clone()];
        let circuit = TestCircuit {a, b, c};

        let (pk, vk) = G16::circuit_specific_setup(circuit.clone(), &mut rng).unwrap();
        let g16_proof = G16::prove(&pk, circuit.clone(), &mut rng).unwrap();

        let pvk = ark_groth16::prepare_verifying_key(&vk);

        let verifies = G16::verify_proof(&pvk, &g16_proof, &public_inputs).unwrap();
        assert!(verifies);

        // Now do a GS-over-canon-Groth16 and verify that (does not hide public inputs)
        let crs = GSCrs::generate_crs(&mut rng);
        let prep_inputs = G16::prepare_inputs(&pvk, &public_inputs).unwrap();
        let gs_proofs = prove_groth16_equations_sat::<Bls12_381, _>(
            &[
                (&g16_proof, &vk, &prep_inputs),
            ],
            &crs,
            &mut rng,
        );

        assert!(verify_groth16_equations_sat::<Bls12_381>(
            &[
                (&vk, &prep_inputs),
            ],
            &gs_proofs,
            &crs
        ));
    }
}

