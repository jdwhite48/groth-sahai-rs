use ark_ec::{PairingEngine};

/// B1,B2,BT forms a bilinear group for GS commitments
pub struct B1<E: PairingEngine>(pub E::G1Affine, pub E::G1Affine);
pub struct B2<E: PairingEngine>(pub E::G2Affine, pub E::G2Affine);
pub struct BT<E: PairingEngine>(pub E::Fqk, pub E::Fqk, pub E::Fqk, pub E::Fqk);

#[allow(non_snake_case)]
#[inline]
/// B_pairing takes entry-wise pairing products
pub(crate) fn B_pairing<E: PairingEngine>(x: &B1<E>, y: &B2<E>) -> BT<E> {
    BT::<E>(
        E::pairing::<E::G1Affine, E::G2Affine>(x.0.clone(), y.0.clone()),
        E::pairing::<E::G1Affine, E::G2Affine>(x.0.clone(), y.1.clone()),
        E::pairing::<E::G1Affine, E::G2Affine>(x.1.clone(), y.0.clone()),
        E::pairing::<E::G1Affine, E::G2Affine>(x.1.clone(), y.1.clone()),
    )
}


#[cfg(test)]
mod tests {

    use ark_bls12_381::{Bls12_381 as F};
    use ark_ff::{UniformRand, Zero, One};
    use ark_ec::{ProjectiveCurve, PairingEngine};
    use ark_std::test_rng;

    use crate::commit::*;

    type G1Projective = <F as PairingEngine>::G1Projective;
    type G1Affine = <F as PairingEngine>::G1Affine;
    type G2Projective = <F as PairingEngine>::G2Projective;
    type G2Affine = <F as PairingEngine>::G2Affine;
    type GT = <F as PairingEngine>::Fqk;

    
    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing() {
        let mut rng = test_rng();
        let b1 = B1::<F>(
            G1Projective::rand(&mut rng).into_affine(),
            G1Projective::rand(&mut rng).into_affine()
        );
        let b2 = B2::<F>(
            G2Projective::rand(&mut rng).into_affine(),
            G2Projective::rand(&mut rng).into_affine()
        );
        let bt = B_pairing::<F>(&b1, &b2);

        assert_eq!(bt.0, <F>::pairing::<G1Affine, G2Affine>(b1.0.clone(), b2.0.clone()));
        assert_eq!(bt.1, <F>::pairing::<G1Affine, G2Affine>(b1.0.clone(), b2.1.clone()));
        assert_eq!(bt.2, <F>::pairing::<G1Affine, G2Affine>(b1.1.clone(), b2.0.clone()));
        assert_eq!(bt.3, <F>::pairing::<G1Affine, G2Affine>(b1.1.clone(), b2.1.clone()));
    
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing_zero_G1() {
        let mut rng = test_rng();
        let b1 = B1::<F>(
            G1Affine::zero(),
            G1Affine::zero()
        );
        let b2 = B2::<F>(
            G2Projective::rand(&mut rng).into_affine(),
            G2Projective::rand(&mut rng).into_affine()
        );
        let bt = B_pairing::<F>(&b1, &b2);

        assert_eq!(bt.0, GT::one());
        assert_eq!(bt.1, GT::one());
        assert_eq!(bt.2, GT::one());
        assert_eq!(bt.3, GT::one());
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing_zero_G2() {
        let mut rng = test_rng();
        let b1 = B1::<F>(
            G1Projective::rand(&mut rng).into_affine(),
            G1Projective::rand(&mut rng).into_affine()
        );
        let b2 = B2::<F>(
            G2Affine::zero(),
            G2Affine::zero()
        );
        let bt = B_pairing::<F>(&b1, &b2);

        assert_eq!(bt.0, GT::one());
        assert_eq!(bt.1, GT::one());
        assert_eq!(bt.2, GT::one());
        assert_eq!(bt.3, GT::one());
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_B_pairing_commit() {
        let mut rng = test_rng();
        let b1 = B1::<F>(
            G1Affine::zero(),
            G1Projective::rand(&mut rng).into_affine()
        );
        let b2 = B2::<F>(
            G2Affine::zero(),
            G2Projective::rand(&mut rng).into_affine()
        );
        let bt = B_pairing::<F>(&b1, &b2);

        assert_eq!(bt.0, GT::one());
        assert_eq!(bt.1, GT::one());
        assert_eq!(bt.2, GT::one());
        assert_eq!(bt.3, <F>::pairing::<G1Affine, G2Affine>(b1.1.clone(), b2.1.clone()));
    }
}
