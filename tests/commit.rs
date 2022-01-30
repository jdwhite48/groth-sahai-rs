#![allow(non_snake_case)]
extern crate groth_sahai;

#[cfg(test)]
mod SXDH_commit_tests {

    use ark_bls12_381::{Bls12_381 as F};
    use ark_ec::{PairingEngine, ProjectiveCurve, AffineCurve};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use groth_sahai::CRS;
    use groth_sahai::data_structures::*;
//    use groth_sahai::commit::*;

    type G1Projective = <F as PairingEngine>::G1Projective;
    type G2Projective = <F as PairingEngine>::G2Projective;
    type Fr = <F as PairingEngine>::Fr;

    #[test]
    fn PPE_linear_bilinear_map_commutativity() {

        let mut rng = test_rng();
        let a1 = G1Projective::rand(&mut rng).into_affine();
        let a2 = G2Projective::rand(&mut rng).into_affine();
        let at = F::pairing(a1.clone(), a2.clone());
        let b1 = Com1::<F>::linear_map(&a1);
        let b2 = Com2::<F>::linear_map(&a2);

        let bt_lin_bilin = ComT::<F>::pairing(b1.clone(), b2.clone());
        let bt_bilin_lin = ComT::<F>::linear_map_PPE(&at);

        assert_eq!(bt_lin_bilin, bt_bilin_lin);
    }

    #[test]
    fn MSMEG1_linear_bilinear_map_commutativity() {

        let mut rng = test_rng();
        let key = CRS::<F>::generate_crs(&mut rng);

        let a1 = G1Projective::rand(&mut rng).into_affine();
        let a2 = Fr::rand(&mut rng);
        let at = a1.mul(a2).into_affine();
        let b1 = Com1::<F>::linear_map(&a1);
        let b2 = Com2::<F>::scalar_linear_map(&a2, &key);

        let bt_lin_bilin = ComT::<F>::pairing(b1.clone(), b2.clone());
        let bt_bilin_lin = ComT::<F>::linear_map_MSG1(&at, &key);

        assert_eq!(bt_lin_bilin, bt_bilin_lin);
    }

    #[test]
    fn MSMEG2_linear_bilinear_map_commutativity() {

        let mut rng = test_rng();
        let key = CRS::<F>::generate_crs(&mut rng);

        let a1 = Fr::rand(&mut rng);
        let a2 = G2Projective::rand(&mut rng).into_affine();
        let at = a2.mul(a1).into_affine();
        let b1 = Com1::<F>::scalar_linear_map(&a1, &key);
        let b2 = Com2::<F>::linear_map(&a2);

        let bt_lin_bilin = ComT::<F>::pairing(b1.clone(), b2.clone());
        let bt_bilin_lin = ComT::<F>::linear_map_MSG2(&at, &key);

        assert_eq!(bt_lin_bilin, bt_bilin_lin);
    }
}
