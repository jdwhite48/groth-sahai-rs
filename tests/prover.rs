#![allow(non_snake_case)]
extern crate groth_sahai;

#[cfg(test)]
mod SXDH_prover_tests {
    
    use ark_bls12_381::{Bls12_381 as F};
    use ark_ec::{PairingEngine, ProjectiveCurve, AffineCurve};
    use ark_ff::{UniformRand, Zero, field_new};
    use ark_std::test_rng;

    use groth_sahai::CRS;
    use groth_sahai::data_structures::*;
    use groth_sahai::commit::*;
    use groth_sahai::prover::*;
    
    type G1Affine = <F as PairingEngine>::G1Affine;
    type G2Affine = <F as PairingEngine>::G2Affine;
    type Fr = <F as PairingEngine>::Fr;
    type Fqk = <F as PairingEngine>::Fqk;

    #[test]
    fn PPE_verifies() {

        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        // An equation of the form e(X_2, c_2) * e(c_1, Y_1) * e(X_1, Y_1)^5 = t where t = e(3 g1, c_2) * e(c_1, 4 g2) * e(2 g1, 4 g2)^5 is satisfied
        // by variables X_1, X_2 in G1 and Y_1 in G2 and constants c_1 in G1 and c_2 in G2

        // X = [ X_1, X_2 ] = [2 g1, 3 g1]
        let xvars: Vec<G1Affine> = vec![
            crs.g1_gen.mul(field_new!(Fr, "2")).into_affine(),
            crs.g1_gen.mul(field_new!(Fr, "3")).into_affine()
        ];
        // Y = [ Y_1 ] = [4 g2]
        let yvars: Vec<G2Affine> = vec![
            crs.g2_gen.mul(field_new!(Fr, "4")).into_affine()
        ];
        let xcoms: Commit1<F> = batch_commit_G1(&xvars, &crs, &mut rng);
        let ycoms: Commit2<F> = batch_commit_G2(&yvars, &crs, &mut rng);

        // A = [ c_1 ] (i.e. e(c_1, Y_1) term in equation)
        let a_consts: Vec<G1Affine> = vec![ crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine() ];
        // B = [ 0, c_2 ] (i.e. only e(X_2, c_2) term in equation)
        let b_consts: Vec<G2Affine> = vec![ G2Affine::zero(), crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()];
        // Gamma = [ 5, 0 ] (i.e. only e(X_1, Y_1)^5 term)
        let gamma: Matrix<Fr> = vec![vec![field_new!(Fr, "5")], vec![Fr::zero()]];
        // Target -> all together (n.b. e(X_1, Y_1)^5 = e(X_1, 5 Y_1) = e(5 X_1, Y_1) by the properties of non-degenerate bilinear maps)
        let target: Fqk = F::pairing(xvars[1].clone(), b_consts[1].clone()) * F::pairing(a_consts[0].clone(), yvars[0].clone()) * F::pairing(xvars[0].clone(), yvars[0].mul(gamma[0][0].clone()).into_affine());
        let equ: PPE<F> = PPE::<F> {
            a_consts, b_consts, gamma, target
        };

        let proof: EquProof<F> = equ.prove(&xvars, &yvars, &xcoms, &ycoms, &crs, &mut rng);
        assert!(equ.verify(&proof, &xcoms, &ycoms, &crs));
    }
}
