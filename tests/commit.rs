extern crate groth_sahai;

#[cfg(test)]
mod commit_int_tests {

    use ark_bls12_381::{Bls12_381 as F};
    use ark_ec::{ProjectiveCurve, PairingEngine};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use groth_sahai::{B1, B2, BT, Com1, Com2, ComT};

    type G1Projective = <F as PairingEngine>::G1Projective;
    type G1Affine = <F as PairingEngine>::G1Affine;
    type G2Projective = <F as PairingEngine>::G2Projective;
    type G2Affine = <F as PairingEngine>::G2Affine;
    type GT = <F as PairingEngine>::Fqk;

    #[allow(non_snake_case)]
    #[test]
    fn test_PPE_map_commutativity() {

        let mut rng = test_rng();
        let g1 = G1Projective::rand(&mut rng).into_affine();
        let g2 = G2Projective::rand(&mut rng).into_affine();
        let gt = F::pairing::<G1Affine, G2Affine>(g1.clone(), g2.clone());
        let b1 = Com1::<F>::linear_map(g1);
        let b2 = Com2::<F>::linear_map(g2);
        let bt_lin_bilin = ComT::<F>::pairing(b1.clone(), b2.clone());
        let bt_bilin_lin = ComT::<F>::linear_map(gt);

        assert_eq!(bt_lin_bilin.0, bt_bilin_lin.0);
        assert_eq!(bt_lin_bilin.1, bt_bilin_lin.1);
        assert_eq!(bt_lin_bilin.2, bt_bilin_lin.2);
        assert_eq!(bt_lin_bilin.3, bt_bilin_lin.3);
    }
}
