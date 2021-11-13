use crate::commit::{B1, B2};

use ark_std::rand::{CryptoRng, Rng};
use ark_ff::{Zero, UniformRand};
use ark_ec::{AffineCurve, ProjectiveCurve, PairingEngine};

/// Commitment keys for G1 and G2, as well as generators for the bilinear group
pub struct CRS<E: PairingEngine> {
    u: (B1<E>, B1<E>),
    v: (B2<E>, B2<E>),
    g1_gen: E::G1Affine,
    g2_gen: E::G2Affine,
    gt_gen: E::Fqk,
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
        let a1 = <E::G1Affine as AffineCurve>::ScalarField::rand(rng);
        let a2 = <E::G2Affine as AffineCurve>::ScalarField::rand(rng);
        let t1 = <E::G1Affine as AffineCurve>::ScalarField::rand(rng);
        let t2 = <E::G2Affine as AffineCurve>::ScalarField::rand(rng);

        // Projective intermediate values
        // TODO: Convert scalar for mul into AsRef<[u64]> to multiply with ProjectiveCurve, if that's more efficient
        let q1 = p1.into_affine().mul(a1);
        let q2 = p2.into_affine().mul(a2);
        let u1 = p1.into_affine().mul(t1);
        let u2 = p2.into_affine().mul(t2);
        let v1: E::G1Projective;
        let v2: E::G2Projective;

        // NOTE: This is supposed to be computationally indistinguishable
        // MAY BE SUBJECT TO SIDE CHANNEL ATTACKS ON GENERATOR

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

        // TODO: Optimization: Check if ((u1, v1), (u2, v2)) are normalized and (if not) batch
        // normalize by slice before converting into affine equivalents?

        // B1 commitment key for G1
        let u11 = B1::<E>(p1.into_affine(), q1.into_affine());
        let u12 = B1::<E>(u1.into_affine(), v1.into_affine());

        // B2 commitment key for G2
        let u21 = B2::<E>(p2.into_affine(), q2.into_affine());
        let u22 = B2::<E>(u2.into_affine(), v2.into_affine());

        CRS::<E> {
            u: (u11, u12),
            v: (u21, u22),
            g1_gen: p1.into_affine(),
            g2_gen: p2.into_affine(),
            gt_gen: E::pairing::<E::G1Affine, E::G2Affine>(p1.into_affine(), p2.into_affine())
        }
    }
}
