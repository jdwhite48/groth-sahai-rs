#![allow(non_snake_case)]

#[cfg(test)]
mod SXDH_commit_tests {

    use ark_bls12_381::Bls12_381 as F;
    use ark_ec::pairing::Pairing;
    use ark_ec::CurveGroup;
    use ark_ff::UniformRand;
    use ark_std::ops::Mul;
    use ark_std::test_rng;

    use groth_sahai::data_structures::*;
    use groth_sahai::{AbstractCrs, CRS};
    //    use groth_sahai::commit::*;

    type G1Projective = <F as Pairing>::G1;
    type G2Projective = <F as Pairing>::G2;
    type Fr = <F as Pairing>::ScalarField;

    #[test]
    fn PPE_linear_bilinear_map_commutativity() {
        let mut rng = test_rng();
        let a1 = G1Projective::rand(&mut rng).into_affine();
        let a2 = G2Projective::rand(&mut rng).into_affine();
        let at = F::pairing(a1, a2);
        let b1 = Com1::<F>::linear_map(&a1);
        let b2 = Com2::<F>::linear_map(&a2);

        let bt_lin_bilin = ComT::<F>::pairing(b1, b2);
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

        let bt_lin_bilin = ComT::<F>::pairing(b1, b2);
        let bt_bilin_lin = ComT::<F>::linear_map_MSMEG1(&at, &key);

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

        let bt_lin_bilin = ComT::<F>::pairing(b1, b2);
        let bt_bilin_lin = ComT::<F>::linear_map_MSMEG2(&at, &key);

        assert_eq!(bt_lin_bilin, bt_bilin_lin);
    }

    #[test]
    fn QuadEqu_linear_bilinear_map_commutativity() {
        let mut rng = test_rng();
        let key = CRS::<F>::generate_crs(&mut rng);

        let a1 = Fr::rand(&mut rng);
        let a2 = Fr::rand(&mut rng);
        let at = a1 * a2;
        let b1 = Com1::<F>::scalar_linear_map(&a1, &key);
        let b2 = Com2::<F>::scalar_linear_map(&a2, &key);

        let bt_lin_bilin = ComT::<F>::pairing(b1, b2);
        let bt_bilin_lin = ComT::<F>::linear_map_quad(&at, &key);

        assert_eq!(bt_lin_bilin, bt_bilin_lin);
    }
}
