use crate::commit::{B1, B2};

use ark_std::rand::{CryptoRng, Rng};
use ark_ff::{Zero, UniformRand};
use ark_ec::{AffineCurve, ProjectiveCurve, PairingEngine};

/// Commitment keys for G1 and G2, as well as generators for the bilinear group
pub struct CommonReferenceString<E: PairingEngine> {
    u: (B1<E>, B1<E>),
    v: (B2<E>, B2<E>),
    g1: E::G1Affine,
    g2: E::G2Affine,
    gt: E::Fqk,
}

impl<E: PairingEngine> CommonReferenceString<E> {

    /// Generates the commitment keys u for G1 and v for G2.
    ///
    /// It should be indistinguishable under the SXDH assumption of the chosen pairing whether u and v were instantiated as a:
    ///    1) Perfect soundness string (i.e. perfectly binding), or
    ///    2) Composable witness-indistinguishability string (i.e. perfectly hiding)
    pub fn generate_crs<R>(rng: &mut R) -> CommonReferenceString<E>
    where
        R: Rng + CryptoRng,
    {
        // Affine
        let p1 = E::G1Affine::prime_subgroup_generator();
        let a1 = <E::G1Affine as AffineCurve>::ScalarField::rand(rng);
        let t1 = <E::G1Affine as AffineCurve>::ScalarField::rand(rng);
        let p2 = E::G2Affine::prime_subgroup_generator();
        let a2 = <E::G2Affine as AffineCurve>::ScalarField::rand(rng);
        let t2 = <E::G2Affine as AffineCurve>::ScalarField::rand(rng);

        // Projective
        let q1 = p1.clone().mul(a1);
        let u1: E::G1Projective = p1.clone().mul(t1);
        let q2 = p2.clone().mul(a2);
        let u2: E::G2Projective = p2.clone().mul(t2);

        let v1: E::G1Projective;
        let v2: E::G2Projective;

        // NOTE: Should be computationally indistinguishable
        // SUBJECT TO SIDE CHANNEL ATTACKS ON GENERATOR

        // TODO: Change t1 and t2 into AsRef<[u64]> to multiply in ProjectiveCurve
        let i = rng.next_u32();
        if i % 2 == 0 {
            // GS Binding Key
            v1 = q1.clone().into_affine().mul(t1) - E::G1Projective::zero();
            v2 = q2.clone().into_affine().mul(t2) - E::G2Projective::zero();
        }
        else {
            // GS Hiding Key
            v1 = q1.clone().into_affine().mul(t1) - p1.into_projective();
            v2 = q2.clone().into_affine().mul(t2) - p2.into_projective();
        }

        // TODO: Optimization: Check if ((u1, v1), (u2, v2)) are normalized and (if not) batch
        // normalize by slice before converting into affine equivalents

        // B1 commitment key for G1
        let u11 = B1::<E>(p1.clone(), q1.into_affine());
        let u12 = B1::<E>(u1.into_affine(), v1.into_affine());

        // B2 commitment key for G2
        let u21 = B2::<E>(p2.clone(), q2.into_affine());
        let u22 = B2::<E>(u2.into_affine(), v2.into_affine());

        CommonReferenceString::<E> {
            u: (u11, u12),
            v: (u21, u22),
            g1: p1.clone(),
            g2: p2.clone(),
            gt: E::pairing::<E::G1Affine, E::G2Affine>(p1.clone(), p2.clone())
        }
    }
}
