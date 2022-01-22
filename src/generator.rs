use crate::data_structures::*;

use ark_std::rand::{CryptoRng, Rng};
use ark_ff::{Zero, UniformRand};
use ark_ec::{AffineCurve, ProjectiveCurve, PairingEngine};

/// Commitment keys for G1 and G2, as well as generators for the bilinear group
pub struct CRS<E: PairingEngine> {
    pub u: Matrix<Com1<E>>,
    pub v: Matrix<Com2<E>>,
    pub g1_gen: E::G1Affine,
    pub g2_gen: E::G2Affine,
    pub gt_gen: E::Fqk,
}

impl<E: PairingEngine> CRS<E> {

    /// Generates the commitment keys u for G1 and v for G2.
    ///
    /// It should be indistinguishable under the SXDH assumption of the chosen pairing whether u and v were instantiated as a:
    ///    1) Perfect soundness string (i.e. perfectly binding), or
    ///    2) Composable witness-indistinguishability string (i.e. perfectly hiding)
    pub fn generate_crs<R>(rng: &mut R) -> CRS<E>
    where
        R: Rng + CryptoRng,
    {

        // Generators for G1 and G2
        let p1 = E::G1Projective::rand(rng);
        let p2 = E::G2Projective::rand(rng);

        // Scalar intermediate values
        let a1 = E::Fr::rand(rng);
        let a2 = E::Fr::rand(rng);
        let t1 = E::Fr::rand(rng);
        let t2 = E::Fr::rand(rng);

        // Projective intermediate values
        // TODO: OPTIMIZATION -- convert scalar for mul into AsRef<[u64]> to multiply with ProjectiveCurve, if that's more efficient
        let q1 = p1.into_affine().mul(a1);
        let q2 = p2.into_affine().mul(a2);
        let u1 = p1.into_affine().mul(t1);
        let u2 = p2.into_affine().mul(t2);
        let v1: E::G1Projective;
        let v2: E::G2Projective;

        // NOTE: This is supposed to be computationally indistinguishable

        // Instantiate GS key
        if rng.gen_bool(0.5) {
            // Binding
            v1 = q1.into_affine().mul(t1) - E::G1Projective::zero();
            v2 = q2.into_affine().mul(t2) - E::G2Projective::zero();
        }
        else {
            // Hiding
            v1 = q1.into_affine().mul(t1) - p1;
            v2 = q2.into_affine().mul(t2) - p2;
        }

        // TODO: OPTIMIZATION -- Check if ((u1, v1), (u2, v2)) are normalized and (if not) batch
        // normalize by slice before converting into affine equivalents?

        // B1 commitment key for G1
        let u11 = Com1::<E>(p1.into_affine(), q1.into_affine());
        let u12 = Com1::<E>(u1.into_affine(), v1.into_affine());

        // B2 commitment key for G2
        let u21 = Com2::<E>(p2.into_affine(), q2.into_affine());
        let u22 = Com2::<E>(u2.into_affine(), v2.into_affine());

        CRS::<E> {
            u: vec![vec![u11], vec![u12]],
            v: vec![vec![u21], vec![u22]],
            g1_gen: p1.into_affine(),
            g2_gen: p2.into_affine(),
            gt_gen: E::pairing::<E::G1Affine, E::G2Affine>(p1.into_affine(), p2.into_affine())
        }
    }
}


#[cfg(test)]
mod tests {

    use ark_bls12_381::{Bls12_381 as F};
    use ark_ff::{Zero, One};
    use ark_ec::PairingEngine;
    use ark_std::test_rng;

    use super::*;

//    type G1Projective = <F as PairingEngine>::G1Projective;
    type G1Affine = <F as PairingEngine>::G1Affine;
//    type G2Projective = <F as PairingEngine>::G2Projective;
    type G2Affine = <F as PairingEngine>::G2Affine;
    type GT = <F as PairingEngine>::Fqk;

    #[allow(non_snake_case)]
    #[test]
    fn test_valid_CRS() {

        let mut rng = test_rng();

        let crs = CRS::<F>::generate_crs(&mut rng);

        // Generator for GT is e(g1,g2)
        assert_eq!(crs.gt_gen, F::pairing::<G1Affine, G2Affine>(crs.g1_gen, crs.g2_gen));
        // Non-degeneracy of bilinear pairing will hold
        assert_ne!(crs.g1_gen, G1Affine::zero());
        assert_ne!(crs.g2_gen, G2Affine::zero());
        assert_ne!(crs.gt_gen, GT::one());

        // Generated commitment keys are non-trivial
        assert_ne!(crs.u[0][0], Com1::zero());
        assert_ne!(crs.u[1][0], Com1::zero());
        assert_ne!(crs.v[0][0], Com2::zero());
        assert_ne!(crs.v[1][0], Com2::zero());
    }
}

