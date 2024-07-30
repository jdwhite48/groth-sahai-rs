#![allow(non_snake_case)]

#[cfg(test)]
mod SXDH_prover_tests {

    use ark_bls12_381::Bls12_381 as F;
    use ark_ec::pairing::{Pairing, PairingOutput};
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_std::ops::Mul;
    use ark_std::str::FromStr;
    use ark_std::{test_rng, UniformRand, Zero};

    use groth_sahai::data_structures::*;
    use groth_sahai::prover::*;
    use groth_sahai::statement::*;
    use groth_sahai::verifier::Verifiable;
    use groth_sahai::{AbstractCrs, CRS};

    type G1Affine = <F as Pairing>::G1Affine;
    type G2Affine = <F as Pairing>::G2Affine;
    type Fr = <F as Pairing>::ScalarField;
    type GT = PairingOutput<F>;

    #[test]
    fn pairing_product_equation_verifies() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        // An equation of the form e(X_2, c_2) * e(c_1, Y_1) * e(X_1, Y_1)^5 = t where t = e(3 g1, c_2) * e(c_1, 4 g2) * e(2 g1, 4 g2)^5 is satisfied
        // by variables X_1, X_2 in G1 and Y_1 in G2, and constants c_1 in G1 and c_2 in G2

        // X = [ X_1, X_2 ] = [2 g1, 3 g1]
        let xvars: Vec<G1Affine> = vec![
            crs.g1_gen.mul(Fr::from_str("2").unwrap()).into_affine(),
            crs.g1_gen.mul(Fr::from_str("3").unwrap()).into_affine(),
        ];
        // Y = [ Y_1 ] = [4 g2]
        let yvars: Vec<G2Affine> = vec![crs.g2_gen.mul(Fr::from_str("4").unwrap()).into_affine()];

        // A = [ c_1 ] (i.e. e(c_1, Y_1) term in equation)
        let a_consts: Vec<G1Affine> = vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()];
        // B = [ 0, c_2 ] (i.e. only e(X_2, c_2) term in equation)
        let b_consts: Vec<G2Affine> = vec![
            G2Affine::zero(),
            crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
        ];
        // Gamma = [ 5, 0 ] (i.e. only e(X_1, Y_1)^5 term)
        let gamma: Matrix<Fr> = vec![vec![Fr::from_str("5").unwrap()], vec![Fr::zero()]];
        // Target -> all together (n.b. e(X_1, Y_1)^5 = e(X_1, 5 Y_1) = e(5 X_1, Y_1) by the properties of non-degenerate bilinear maps)
        let target: GT = F::pairing(xvars[1], b_consts[1])
            + F::pairing(a_consts[0], yvars[0])
            + F::pairing(xvars[0], yvars[0].mul(gamma[0][0]).into_affine());
        let equ: PPE<F> = PPE::<F> {
            a_consts,
            b_consts,
            gamma,
            target,
        };

        let proof: CProof<F> = equ.commit_and_prove(&xvars, &yvars, &crs, &mut rng);
        assert!(equ.verify(&proof, &crs));
    }

    #[test]
    fn multi_scalar_mult_equation_G1_verifies() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        // An equation of the form c_2 * X_2 + y_1 * c_1 + (y_1 * X_1)*5 = t where t = c_2 * (3 g1) + 4 * c_1 + (4 * (2 g1))*5 is satisfied
        // by variables X_1, X_2 in G1 and y_1 in Fr, and constants c_1 in G1 and c_2 in Fr

        // X = [ X_1, X_2 ] = [2 g1, 3 g1]
        let xvars: Vec<G1Affine> = vec![
            crs.g1_gen.mul(Fr::from_str("2").unwrap()).into_affine(),
            crs.g1_gen.mul(Fr::from_str("3").unwrap()).into_affine(),
        ];
        // y = [ y_1 ] = [ 4 ]
        let scalar_yvars: Vec<Fr> = vec![Fr::from_str("4").unwrap()];

        // A = [ c_1 ] (i.e. y_1 * c_1 term in equation)
        let a_consts: Vec<G1Affine> = vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()];
        // B = [ 0, c_2 ] (i.e. only c_2 * X_2 term in equation)
        let b_consts: Vec<Fr> = vec![Fr::zero(), Fr::rand(&mut rng)];
        // Gamma = [ 5, 0 ] (i.e. only (y_1 * X_1)*5 term)
        let gamma: Matrix<Fr> = vec![vec![Fr::from_str("5").unwrap()], vec![Fr::zero()]];
        // Target -> all together
        let target: G1Affine = (xvars[1].mul(b_consts[1])
            + a_consts[0].mul(scalar_yvars[0])
            + xvars[0].mul(scalar_yvars[0] * gamma[0][0]))
        .into_affine();
        let equ: MSMEG1<F> = MSMEG1::<F> {
            a_consts,
            b_consts,
            gamma,
            target,
        };

        let proof: CProof<F> = equ.commit_and_prove(&xvars, &scalar_yvars, &crs, &mut rng);
        assert!(equ.verify(&proof, &crs));
    }

    #[test]
    fn multi_scalar_mult_equation_G2_verifies() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        // An equation of the form x_2 * c_2 + c_1 * Y_1 + (x_1 * Y_1)*5 = t where t = 3 * c_2 + c_1 * (4 g2) + (2 * (4 g2))*5 is satisfied
        // by variables x_1, x_2 in Fr and Y_1 in G2, and constants c_1 in Fr and c_2 in G2

        // x = [ x_1, x_2 ] = [2, 3]
        let scalar_xvars: Vec<Fr> = vec![Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()];
        // Y = [ y_1 ] = [ 4 g2 ]
        let yvars: Vec<G2Affine> = vec![crs.g2_gen.mul(Fr::from_str("4").unwrap()).into_affine()];

        // A = [ c_1 ] (i.e. c_1 * Y_1 term in equation)
        let a_consts: Vec<Fr> = vec![Fr::rand(&mut rng)];
        // B = [ 0, c_2 ] (i.e. only x_2 * c_2 term in equation)
        let b_consts: Vec<G2Affine> = vec![
            G2Affine::zero(),
            crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
        ];
        // Gamma = [ 5, 0 ] (i.e. only (x_1 * Y_1)*5 term)
        let gamma: Matrix<Fr> = vec![vec![Fr::from_str("5").unwrap()], vec![Fr::zero()]];
        // Target -> all together
        let target: G2Affine = (b_consts[1].mul(scalar_xvars[1])
            + yvars[0].mul(a_consts[0])
            + yvars[0].mul(scalar_xvars[0] * gamma[0][0]))
        .into_affine();
        let equ: MSMEG2<F> = MSMEG2::<F> {
            a_consts,
            b_consts,
            gamma,
            target,
        };

        let proof: CProof<F> = equ.commit_and_prove(&scalar_xvars, &yvars, &crs, &mut rng);
        assert!(equ.verify(&proof, &crs));
    }

    #[test]
    fn quadratic_equation_verifies() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        // An equation of the form c_2 * x_2 + c_1 * y_1 + (x_1 * y_1)*5 = t where t = c_2 * 3 + c_1 * 4 + (2 * 4)*5 is satisfied
        // by variables x_1, x_2 and y_1 in Fr, and constants c_1 and c_2 in Fr

        // x = [ x_1, x_2 ] = [2, 3]
        let scalar_xvars: Vec<Fr> = vec![Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()];
        // y = [ y_1 ] = [ 4 ]
        let scalar_yvars: Vec<Fr> = vec![Fr::from_str("4").unwrap()];

        // A = [ c_1 ] (i.e. c_1 * y_1 term in equation)
        let a_consts: Vec<Fr> = vec![Fr::rand(&mut rng)];
        // B = [ 0, c_2 ] (i.e. only c_2 * x2 term in equation)
        let b_consts: Vec<Fr> = vec![Fr::zero(), Fr::rand(&mut rng)];
        // Gamma = [ 5, 0 ] (i.e. only (x_1 * y_1)*5 term)
        let gamma: Matrix<Fr> = vec![vec![Fr::from_str("5").unwrap()], vec![Fr::zero()]];
        // Target -> all together
        let target: Fr = b_consts[1] * scalar_xvars[1]
            + scalar_yvars[0] * a_consts[0]
            + scalar_yvars[0] * scalar_xvars[0] * gamma[0][0];
        let equ: QuadEqu<F> = QuadEqu::<F> {
            a_consts,
            b_consts,
            gamma,
            target,
        };

        let proof: CProof<F> = equ.commit_and_prove(&scalar_xvars, &scalar_yvars, &crs, &mut rng);
        assert!(equ.verify(&proof, &crs));
    }
}
