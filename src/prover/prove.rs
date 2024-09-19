//! Contains the functionality for proving about the satisfiability of Groth-Sahai equations over bilinear groups.
//!
//! Abstractly, a proof for an equation for the SXDH instantiation of Groth-Sahai consists of the following values,
//! with respect to a pre-defined bilinear group `(A1, A2, AT)`:
//!
//! - `π`: 1-2 elements in [`B2`](crate::data_structures::Com2) (equiv. 2-4 elements in [`G2`](ark_ec::Pairing::G2Affine))
//!     which prove about the satisfiability of `A2` variables in the equation, and
//! - `θ`: 1-2 elements in [`B1`](crate::data_structures::Com1) (equiv. 2-4 elements in [`G1`](ark_ec::Pairing::G1Affine))
//!     which prove about the satisfiability of `A1` variables in the equation
//!
//! Computing these proofs primarily involves matrix multiplication in the [scalar field](ark_ec::Pairing::Fr) and in `B1` and `B2`.
//!
//! See the [`statement`](crate::statement) module for more details about the structure of the equations being proven about.

use ark_ec::pairing::Pairing;
use ark_ec::pairing::PairingOutput;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{rand::Rng, UniformRand};

use super::commit::{
    batch_commit_G1, batch_commit_G2, batch_commit_scalar_to_B1, batch_commit_scalar_to_B2,
    Commit1, Commit2,
};
use crate::data_structures::{col_vec_to_vec, vec_to_col_vec, Com1, Com2, Mat, Matrix, B1, B2};
use crate::generator::CRS;
use crate::statement::{EquType, QuadEqu, MSMEG1, MSMEG2, PPE};

/// A collection  of attributes containing prover functionality for an [`Equation`](crate::statement::Equation).
pub trait Provable<E: Pairing, A1, A2, AT> {
    /// Commits to the witness variables and then produces a Groth-Sahai proof for this equation.
    fn commit_and_prove<CR>(
        &self,
        xvars: &[A1],
        yvars: &[A2],
        crs: &CRS<E>,
        rng: &mut CR,
    ) -> CProof<E>
    where
        CR: Rng;
    /// Produces a proof `(π, θ)` for this equation that the already-committed `x` and `y` variables will satisfy a single Groth-Sahai equation.
    fn prove<CR>(
        &self,
        xvars: &[A1],
        yvars: &[A2],
        xcoms: &Commit1<E>,
        ycoms: &Commit2<E>,
        crs: &CRS<E>,
        rng: &mut CR,
    ) -> EquProof<E>
    where
        CR: Rng;
}

/// A witness-indistinguishable proof for a single [`Equation`](crate::statement::Equation).
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct EquProof<E: Pairing> {
    pub pi: Vec<Com2<E>>,
    pub theta: Vec<Com1<E>>,
    pub equ_type: EquType,
    rand: Matrix<E::ScalarField>,
}

/// A collection of committed variables and proofs for Groth-Sahai compatible bilinear equations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CProof<E: Pairing> {
    pub xcoms: Commit1<E>,
    pub ycoms: Commit2<E>,
    pub equ_proofs: Vec<EquProof<E>>,
}

impl<E: Pairing> Provable<E, E::G1Affine, E::G2Affine, PairingOutput<E>> for PPE<E> {
    fn commit_and_prove<CR>(
        &self,
        xvars: &[E::G1Affine],
        yvars: &[E::G2Affine],
        crs: &CRS<E>,
        rng: &mut CR,
    ) -> CProof<E>
    where
        CR: Rng,
    {
        let xcoms: Commit1<E> = batch_commit_G1(xvars, crs, rng);
        let ycoms: Commit2<E> = batch_commit_G2(yvars, crs, rng);

        CProof::<E> {
            xcoms: xcoms.clone(),
            ycoms: ycoms.clone(),
            equ_proofs: vec![self.prove(xvars, yvars, &xcoms, &ycoms, crs, rng)],
        }
    }

    fn prove<CR>(
        &self,
        xvars: &[E::G1Affine],
        yvars: &[E::G2Affine],
        xcoms: &Commit1<E>,
        ycoms: &Commit2<E>,
        crs: &CRS<E>,
        rng: &mut CR,
    ) -> EquProof<E>
    where
        CR: Rng,
    {
        // Gamma is an (m x n) matrix with m x variables and n y variables
        // x's commit randomness (i.e. R) is a (m x 2) matrix
        assert_eq!(xvars.len(), xcoms.rand.len());
        assert_eq!(self.gamma.len(), xcoms.rand.len());
        assert_eq!(xcoms.rand[0].len(), 2);
        let _m = xvars.len();
        // y's commit randomness (i.e. S) is a (n x 2) matrix
        assert_eq!(yvars.len(), ycoms.rand.len());
        assert_eq!(self.gamma[0].len(), ycoms.rand.len());
        assert_eq!(ycoms.rand[0].len(), 2);
        let _n = yvars.len();

        let is_parallel = true;

        // (2 x m) field matrix R^T, in GS parlance
        let x_rand_trans = xcoms.rand.transpose();
        // (2 x n) field matrix S^T, in GS parlance
        let y_rand_trans = ycoms.rand.transpose();
        // (2 x 2) field matrix T, in GS parlance
        let pf_rand: Matrix<E::ScalarField> = vec![
            vec![E::ScalarField::rand(rng), E::ScalarField::rand(rng)],
            vec![E::ScalarField::rand(rng), E::ScalarField::rand(rng)],
        ];

        // (2 x 1) Com2 matrix
        let x_rand_lin_b = vec_to_col_vec(&Com2::<E>::batch_linear_map(&self.b_consts))
            .left_mul(&x_rand_trans, is_parallel);

        // (2 x n) field matrix
        let x_rand_stmt = x_rand_trans.right_mul(&self.gamma, is_parallel);
        // (2 x 1) Com2 matrix
        let x_rand_stmt_lin_y =
            vec_to_col_vec(&Com2::<E>::batch_linear_map(yvars)).left_mul(&x_rand_stmt, is_parallel);

        // (2 x 2) field matrix
        let pf_rand_stmt = x_rand_trans
            .right_mul(&self.gamma, is_parallel)
            .right_mul(&ycoms.rand, is_parallel)
            .add(&pf_rand.transpose().neg());
        // (2 x 1) Com2 matrix
        let pf_rand_stmt_com2 = vec_to_col_vec(&crs.v).left_mul(&pf_rand_stmt, is_parallel);

        let pi = col_vec_to_vec(&x_rand_lin_b.add(&x_rand_stmt_lin_y).add(&pf_rand_stmt_com2));
        assert_eq!(pi.len(), 2);

        // (2 x 1) Com1 matrix
        let y_rand_lin_a = vec_to_col_vec(&Com1::<E>::batch_linear_map(&self.a_consts))
            .left_mul(&y_rand_trans, is_parallel);

        // (2 x m) field matrix
        let y_rand_stmt = y_rand_trans.right_mul(&self.gamma.transpose(), is_parallel);
        // (2 x 1) Com1 matrix
        let y_rand_stmt_lin_x =
            vec_to_col_vec(&Com1::<E>::batch_linear_map(xvars)).left_mul(&y_rand_stmt, is_parallel);

        // (2 x 1) Com1 matrix
        let pf_rand_com1 = vec_to_col_vec(&crs.u).left_mul(&pf_rand, is_parallel);

        let theta = col_vec_to_vec(&y_rand_lin_a.add(&y_rand_stmt_lin_x).add(&pf_rand_com1));
        assert_eq!(theta.len(), 2);

        EquProof::<E> {
            pi,
            theta,
            equ_type: EquType::PairingProduct,
            rand: pf_rand,
        }
    }
}

impl<E: Pairing> Provable<E, E::G1Affine, E::ScalarField, E::G1Affine> for MSMEG1<E> {
    fn commit_and_prove<CR>(
        &self,
        xvars: &[E::G1Affine],
        scalar_yvars: &[E::ScalarField],
        crs: &CRS<E>,
        rng: &mut CR,
    ) -> CProof<E>
    where
        CR: Rng,
    {
        let xcoms: Commit1<E> = batch_commit_G1(xvars, crs, rng);
        let scalar_ycoms: Commit2<E> = batch_commit_scalar_to_B2(scalar_yvars, crs, rng);

        CProof::<E> {
            xcoms: xcoms.clone(),
            ycoms: scalar_ycoms.clone(),
            equ_proofs: vec![self.prove(xvars, scalar_yvars, &xcoms, &scalar_ycoms, crs, rng)],
        }
    }

    fn prove<CR>(
        &self,
        xvars: &[E::G1Affine],
        scalar_yvars: &[E::ScalarField],
        xcoms: &Commit1<E>,
        scalar_ycoms: &Commit2<E>,
        crs: &CRS<E>,
        rng: &mut CR,
    ) -> EquProof<E>
    where
        CR: Rng,
    {
        // Gamma is an (m x n') matrix with m x variables and n' scalar y variables
        // x's commit randomness (i.e. R) is a (m x 2) matrix
        assert_eq!(xvars.len(), xcoms.rand.len());
        assert_eq!(self.gamma.len(), xcoms.rand.len());
        assert_eq!(xcoms.rand[0].len(), 2);
        let _m = xvars.len();
        // scalar y's commit randomness (i.e. s) is a (n' x 1) matrix (i.e. column vector)
        assert_eq!(scalar_yvars.len(), scalar_ycoms.rand.len());
        assert_eq!(self.gamma[0].len(), scalar_ycoms.rand.len());
        assert_eq!(scalar_ycoms.rand[0].len(), 1);
        let _n_prime = scalar_yvars.len();

        let is_parallel = true;

        // (2 x m) field matrix R^T, in GS parlance
        let x_rand_trans = xcoms.rand.transpose();
        // (1 x n') field matrix s^T, in GS parlance
        let y_rand_trans = scalar_ycoms.rand.transpose();
        // (1 x 2) field matrix T, in GS parlance
        let pf_rand: Matrix<E::ScalarField> =
            vec![vec![E::ScalarField::rand(rng), E::ScalarField::rand(rng)]];

        // (2 x 1) Com2 matrix
        let x_rand_lin_b = vec_to_col_vec(&Com2::<E>::batch_scalar_linear_map(&self.b_consts, crs))
            .left_mul(&x_rand_trans, is_parallel);

        // (2 x n) field matrix
        let x_rand_stmt = x_rand_trans.right_mul(&self.gamma, is_parallel);
        // (2 x 1) Com2 matrix
        let x_rand_stmt_lin_y =
            vec_to_col_vec(&Com2::<E>::batch_scalar_linear_map(scalar_yvars, crs))
                .left_mul(&x_rand_stmt, is_parallel);

        // (2 x 1) field matrix
        let pf_rand_stmt = x_rand_trans
            .right_mul(&self.gamma, is_parallel)
            .right_mul(&scalar_ycoms.rand, is_parallel)
            .add(&pf_rand.transpose().neg());
        // (2 x 1) Com2 matrix
        let v1: Matrix<Com2<E>> = vec![vec![crs.v[0]]];
        let pf_rand_stmt_com2 = v1.left_mul(&pf_rand_stmt, is_parallel);

        let pi = col_vec_to_vec(&x_rand_lin_b.add(&x_rand_stmt_lin_y).add(&pf_rand_stmt_com2));
        assert_eq!(pi.len(), 2);

        // (1 x 1) Com1 matrix
        let y_rand_lin_a = vec_to_col_vec(&Com1::<E>::batch_linear_map(&self.a_consts))
            .left_mul(&y_rand_trans, is_parallel);

        // (1 x m) field matrix
        let y_rand_stmt = y_rand_trans.right_mul(&self.gamma.transpose(), is_parallel);
        // (1 x 1) Com1 matrix
        let y_rand_stmt_lin_x =
            vec_to_col_vec(&Com1::<E>::batch_linear_map(xvars)).left_mul(&y_rand_stmt, is_parallel);

        // (1 x 1) Com1 matrix
        let pf_rand_com1 = vec_to_col_vec(&crs.u).left_mul(&pf_rand, is_parallel);

        let theta = col_vec_to_vec(&y_rand_lin_a.add(&y_rand_stmt_lin_x).add(&pf_rand_com1));
        assert_eq!(theta.len(), 1);

        EquProof::<E> {
            pi,
            theta,
            equ_type: EquType::MultiScalarG1,
            rand: pf_rand,
        }
    }
}

impl<E: Pairing> Provable<E, E::ScalarField, E::G2Affine, E::G2Affine> for MSMEG2<E> {
    fn commit_and_prove<CR>(
        &self,
        scalar_xvars: &[E::ScalarField],
        yvars: &[E::G2Affine],
        crs: &CRS<E>,
        rng: &mut CR,
    ) -> CProof<E>
    where
        CR: Rng,
    {
        let scalar_xcoms: Commit1<E> = batch_commit_scalar_to_B1(scalar_xvars, crs, rng);
        let ycoms: Commit2<E> = batch_commit_G2(yvars, crs, rng);

        CProof::<E> {
            xcoms: scalar_xcoms.clone(),
            ycoms: ycoms.clone(),
            equ_proofs: vec![self.prove(scalar_xvars, yvars, &scalar_xcoms, &ycoms, crs, rng)],
        }
    }

    fn prove<CR>(
        &self,
        scalar_xvars: &[E::ScalarField],
        yvars: &[E::G2Affine],
        scalar_xcoms: &Commit1<E>,
        ycoms: &Commit2<E>,
        crs: &CRS<E>,
        rng: &mut CR,
    ) -> EquProof<E>
    where
        CR: Rng,
    {
        // Gamma is an (m' x n) matrix with m' x variables and n y variables
        // x's commit randomness (i.e. r) is a (m' x 1) matrix (i.e. column vector)
        assert_eq!(scalar_xvars.len(), scalar_xcoms.rand.len());
        assert_eq!(self.gamma.len(), scalar_xcoms.rand.len());
        assert_eq!(scalar_xcoms.rand[0].len(), 1);
        let _m_prime = scalar_xvars.len();
        // y's commit randomness (i.e. S) is a (n x 2) matrix
        assert_eq!(yvars.len(), ycoms.rand.len());
        assert_eq!(self.gamma[0].len(), ycoms.rand.len());
        assert_eq!(ycoms.rand[0].len(), 2);
        let _n = yvars.len();

        let is_parallel = true;

        // (1 x m') field matrix r^T, in GS parlance
        let x_rand_trans = scalar_xcoms.rand.transpose();
        // (2 x n) field matrix S^T, in GS parlance
        let y_rand_trans = ycoms.rand.transpose();
        // (2 x 1) field matrix T, in GS parlance
        let pf_rand: Matrix<E::ScalarField> = vec![
            vec![E::ScalarField::rand(rng)],
            vec![E::ScalarField::rand(rng)],
        ];

        // (1 x 1) Com2 matrix
        let x_rand_lin_b = vec_to_col_vec(&Com2::<E>::batch_linear_map(&self.b_consts))
            .left_mul(&x_rand_trans, is_parallel);

        // (1 x n) field matrix
        let x_rand_stmt = x_rand_trans.right_mul(&self.gamma, is_parallel);
        // (1 x 1) Com2 matrix
        let x_rand_stmt_lin_y =
            vec_to_col_vec(&Com2::<E>::batch_linear_map(yvars)).left_mul(&x_rand_stmt, is_parallel);

        // (1 x 2) field matrix
        let pf_rand_stmt = x_rand_trans
            .right_mul(&self.gamma, is_parallel)
            .right_mul(&ycoms.rand, is_parallel)
            .add(&pf_rand.transpose().neg());
        // (1 x 1) Com2 matrix
        let pf_rand_stmt_com2 = vec_to_col_vec(&crs.v).left_mul(&pf_rand_stmt, is_parallel);

        let pi = col_vec_to_vec(&x_rand_lin_b.add(&x_rand_stmt_lin_y).add(&pf_rand_stmt_com2));
        assert_eq!(pi.len(), 1);

        // (2 x 1) Com1 matrix
        let y_rand_lin_a = vec_to_col_vec(&Com1::<E>::batch_scalar_linear_map(&self.a_consts, crs))
            .left_mul(&y_rand_trans, is_parallel);

        // (2 x m') field matrix
        let y_rand_stmt = y_rand_trans.right_mul(&self.gamma.transpose(), is_parallel);
        // (2 x 1) Com1 matrix
        let y_rand_stmt_lin_x =
            vec_to_col_vec(&Com1::<E>::batch_scalar_linear_map(scalar_xvars, crs))
                .left_mul(&y_rand_stmt, is_parallel);

        // (2 x 1) Com1 matrix
        let u1: Matrix<Com1<E>> = vec![vec![crs.u[0]]];
        let pf_rand_com1 = u1.left_mul(&pf_rand, is_parallel);

        let theta = col_vec_to_vec(&y_rand_lin_a.add(&y_rand_stmt_lin_x).add(&pf_rand_com1));
        assert_eq!(theta.len(), 2);

        EquProof::<E> {
            pi,
            theta,
            equ_type: EquType::MultiScalarG2,
            rand: pf_rand,
        }
    }
}

impl<E: Pairing> Provable<E, E::ScalarField, E::ScalarField, E::ScalarField> for QuadEqu<E> {
    fn commit_and_prove<CR>(
        &self,
        scalar_xvars: &[E::ScalarField],
        scalar_yvars: &[E::ScalarField],
        crs: &CRS<E>,
        rng: &mut CR,
    ) -> CProof<E>
    where
        CR: Rng,
    {
        let scalar_xcoms: Commit1<E> = batch_commit_scalar_to_B1(scalar_xvars, crs, rng);
        let scalar_ycoms: Commit2<E> = batch_commit_scalar_to_B2(scalar_yvars, crs, rng);

        CProof::<E> {
            xcoms: scalar_xcoms.clone(),
            ycoms: scalar_ycoms.clone(),
            equ_proofs: vec![self.prove(
                scalar_xvars,
                scalar_yvars,
                &scalar_xcoms,
                &scalar_ycoms,
                crs,
                rng,
            )],
        }
    }
    fn prove<CR>(
        &self,
        scalar_xvars: &[E::ScalarField],
        scalar_yvars: &[E::ScalarField],
        scalar_xcoms: &Commit1<E>,
        scalar_ycoms: &Commit2<E>,
        crs: &CRS<E>,
        rng: &mut CR,
    ) -> EquProof<E>
    where
        CR: Rng,
    {
        // Gamma is an (m' x n') matrix with m' x variables and n' y variables
        // x's commit randomness (i.e. r) is a (m' x 1) matrix (i.e. column vector)
        assert_eq!(scalar_xvars.len(), scalar_xcoms.rand.len());
        assert_eq!(self.gamma.len(), scalar_xcoms.rand.len());
        assert_eq!(scalar_xcoms.rand[0].len(), 1);
        let _m_prime = scalar_xvars.len();
        // y's commit randomness (i.e. s) is a (n' x 1) matrix (i.e. column vector)
        assert_eq!(scalar_yvars.len(), scalar_ycoms.rand.len());
        assert_eq!(self.gamma[0].len(), scalar_ycoms.rand.len());
        assert_eq!(scalar_ycoms.rand[0].len(), 1);
        let _n_prime = scalar_yvars.len();

        let is_parallel = true;

        // (1 x m') field matrix r^T, in GS parlance
        let x_rand_trans = scalar_xcoms.rand.transpose();
        // (1 x n') field matrix s^T, in GS parlance
        let y_rand_trans = scalar_ycoms.rand.transpose();
        // field element T, in GS parlance
        let pf_rand: Matrix<E::ScalarField> = vec![vec![E::ScalarField::rand(rng)]];

        let x_rand_lin_b = vec_to_col_vec(&Com2::<E>::batch_scalar_linear_map(&self.b_consts, crs))
            .left_mul(&x_rand_trans, is_parallel);

        // (1 x n') field matrix
        let x_rand_stmt = x_rand_trans.right_mul(&self.gamma, is_parallel);
        // (1 x 1) Com2 matrix
        let x_rand_stmt_lin_y =
            vec_to_col_vec(&Com2::<E>::batch_scalar_linear_map(scalar_yvars, crs))
                .left_mul(&x_rand_stmt, is_parallel);

        // (1 x 2) field matrix
        let pf_rand_stmt = x_rand_trans
            .right_mul(&self.gamma, is_parallel)
            .right_mul(&scalar_ycoms.rand, is_parallel)
            .add(&pf_rand.transpose().neg());
        let v1: Matrix<Com2<E>> = vec![vec![crs.v[0]]];
        // (1 x 1) Com2 matrix
        let pf_rand_stmt_com2 = v1.left_mul(&pf_rand_stmt, is_parallel);

        let pi = col_vec_to_vec(&x_rand_lin_b.add(&x_rand_stmt_lin_y).add(&pf_rand_stmt_com2));
        assert_eq!(pi.len(), 1);

        // (1 x 1) Com1 matrix
        let y_rand_lin_a = vec_to_col_vec(&Com1::<E>::batch_scalar_linear_map(&self.a_consts, crs))
            .left_mul(&y_rand_trans, is_parallel);

        // (1 x m') field matrix
        let y_rand_stmt = y_rand_trans.right_mul(&self.gamma.transpose(), is_parallel);
        // (1 x 1) Com1 matrix
        let y_rand_stmt_lin_x =
            vec_to_col_vec(&Com1::<E>::batch_scalar_linear_map(scalar_xvars, crs))
                .left_mul(&y_rand_stmt, is_parallel);

        // (1 x 1) Com1 matrix
        let u1: Matrix<Com1<E>> = vec![vec![crs.u[0]]];
        let pf_rand_com1 = u1.left_mul(&pf_rand, is_parallel);

        let theta = col_vec_to_vec(&y_rand_lin_a.add(&y_rand_stmt_lin_x).add(&pf_rand_com1));
        assert_eq!(theta.len(), 1);

        EquProof::<E> {
            pi,
            theta,
            equ_type: EquType::Quadratic,
            rand: pf_rand,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use ark_bls12_381::Bls12_381 as F;
    use ark_ec::CurveGroup;
    use ark_ff::{One, UniformRand, Zero};
    use ark_std::ops::Mul;
    use ark_std::test_rng;

    use crate::AbstractCrs;

    use super::*;

    type G1Affine = <F as Pairing>::G1Affine;
    type G2Affine = <F as Pairing>::G2Affine;
    type Fr = <F as Pairing>::ScalarField;
    type GT = PairingOutput<F>;

    #[test]
    fn test_PPE_proof_type() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let xvars: Vec<G1Affine> = vec![
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
        ];
        let yvars: Vec<G2Affine> = vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()];
        let xcoms: Commit1<F> = batch_commit_G1(&xvars, &crs, &mut rng);
        let ycoms: Commit2<F> = batch_commit_G2(&yvars, &crs, &mut rng);

        let equ: PPE<F> = PPE::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
            ],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: GT::rand(&mut rng),
        };
        let proof: EquProof<F> = equ.prove(&xvars, &yvars, &xcoms, &ycoms, &crs, &mut rng);

        assert_eq!(proof.equ_type, EquType::PairingProduct);
    }

    #[test]
    fn test_PPE_cproof_is_commit_and_prove() {
        std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
        let mut rng = test_rng();
        let mut rng2 = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let xvars: Vec<G1Affine> = vec![
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
        ];
        let yvars: Vec<G2Affine> = vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()];
        let equ: PPE<F> = PPE::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
            ],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: GT::rand(&mut rng),
        };

        // Individually commit then prove
        let xcoms: Commit1<F> = batch_commit_G1(&xvars, &crs, &mut rng);
        let ycoms: Commit2<F> = batch_commit_G2(&yvars, &crs, &mut rng);
        let proof: EquProof<F> = equ.prove(&xvars, &yvars, &xcoms, &ycoms, &crs, &mut rng);
        let cproof = CProof::<F> {
            xcoms,
            ycoms,
            equ_proofs: vec![proof],
        };

        // Mock calls to CRS to get them in sync
        let _ = CRS::<F>::generate_crs(&mut rng2);
        for _ in 0..xvars.len() {
            let _ = Fr::rand(&mut rng2);
        }
        for _ in 0..yvars.len() {
            let _ = Fr::rand(&mut rng2);
        }
        for _ in 0..equ.a_consts.len() {
            let _ = Fr::rand(&mut rng2);
        }
        for _ in 0..equ.b_consts.len() {
            let _ = Fr::rand(&mut rng2);
        }
        let _ = GT::rand(&mut rng2);

        // Use the helper function to commit-and-prove in one step
        let cproof2 = equ.commit_and_prove(&xvars, &yvars, &crs, &mut rng2);

        assert_eq!(cproof, cproof2);
    }

    #[test]
    fn test_PPE_cproof_serde() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let xvars: Vec<G1Affine> = vec![
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
        ];
        let yvars: Vec<G2Affine> = vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()];
        let xcoms: Commit1<F> = batch_commit_G1(&xvars, &crs, &mut rng);
        let ycoms: Commit2<F> = batch_commit_G2(&yvars, &crs, &mut rng);

        let equ: PPE<F> = PPE::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
            ],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: GT::rand(&mut rng),
        };
        let proof: EquProof<F> = equ.prove(&xvars, &yvars, &xcoms, &ycoms, &crs, &mut rng);

        // Serialize and deserialize the proof
        let mut c_bytes = Vec::new();
        proof.serialize_compressed(&mut c_bytes).unwrap();
        let proof_de = EquProof::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(proof, proof_de);

        let mut u_bytes = Vec::new();
        proof.serialize_uncompressed(&mut u_bytes).unwrap();
        let proof_de = EquProof::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(proof, proof_de);
    }

    #[test]
    fn test_MSMEG1_proof_type() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let xvars: Vec<G1Affine> = vec![
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
        ];
        let scalar_yvars: Vec<Fr> = vec![Fr::rand(&mut rng)];
        let xcoms: Commit1<F> = batch_commit_G1(&xvars, &crs, &mut rng);
        let scalar_ycoms: Commit2<F> = batch_commit_scalar_to_B2(&scalar_yvars, &crs, &mut rng);

        let equ: MSMEG1<F> = MSMEG1::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![Fr::rand(&mut rng), Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
        };
        let proof: EquProof<F> =
            equ.prove(&xvars, &scalar_yvars, &xcoms, &scalar_ycoms, &crs, &mut rng);

        assert_eq!(proof.equ_type, EquType::MultiScalarG1);
    }

    #[test]
    fn test_MSMEG1_cproof_is_commit_and_prove() {
        std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
        let mut rng = test_rng();
        let mut rng2 = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let xvars: Vec<G1Affine> = vec![
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
        ];
        let scalar_yvars: Vec<Fr> = vec![Fr::rand(&mut rng)];
        let equ: MSMEG1<F> = MSMEG1::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![Fr::rand(&mut rng), Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
        };

        // Individually commit then prove
        let xcoms: Commit1<F> = batch_commit_G1(&xvars, &crs, &mut rng);
        let scalar_ycoms: Commit2<F> = batch_commit_scalar_to_B2(&scalar_yvars, &crs, &mut rng);
        let proof: EquProof<F> =
            equ.prove(&xvars, &scalar_yvars, &xcoms, &scalar_ycoms, &crs, &mut rng);
        let cproof = CProof::<F> {
            xcoms,
            ycoms: scalar_ycoms,
            equ_proofs: vec![proof],
        };

        // Mock calls to CRS to get them in sync
        let _ = CRS::<F>::generate_crs(&mut rng2);
        for _ in 0..xvars.len() {
            let _ = Fr::rand(&mut rng2);
        }
        for _ in 0..scalar_yvars.len() {
            let _ = Fr::rand(&mut rng2);
        }
        for _ in 0..equ.a_consts.len() {
            let _ = Fr::rand(&mut rng2);
        }
        for _ in 0..equ.b_consts.len() {
            let _ = Fr::rand(&mut rng2);
        }
        let _ = Fr::rand(&mut rng2);

        // Use the helper function to commit-and-prove in one step
        let cproof2 = equ.commit_and_prove(&xvars, &scalar_yvars, &crs, &mut rng2);

        assert_eq!(cproof, cproof2);
    }

    #[test]
    fn test_MSGMEG1_cproof_serde() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let xvars: Vec<G1Affine> = vec![
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
        ];
        let scalar_yvars: Vec<Fr> = vec![Fr::rand(&mut rng)];
        let xcoms: Commit1<F> = batch_commit_G1(&xvars, &crs, &mut rng);
        let scalar_ycoms: Commit2<F> = batch_commit_scalar_to_B2(&scalar_yvars, &crs, &mut rng);

        let equ: MSMEG1<F> = MSMEG1::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![Fr::rand(&mut rng), Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
        };
        let proof: EquProof<F> =
            equ.prove(&xvars, &scalar_yvars, &xcoms, &scalar_ycoms, &crs, &mut rng);

        // Serialize and deserialize the proof
        let mut c_bytes = Vec::new();
        proof.serialize_compressed(&mut c_bytes).unwrap();
        let proof_de = EquProof::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(proof, proof_de);

        let mut u_bytes = Vec::new();
        proof.serialize_uncompressed(&mut u_bytes).unwrap();
        let proof_de = EquProof::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(proof, proof_de);
    }

    #[test]
    fn test_MSMEG2_proof_type() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let scalar_xvars: Vec<Fr> = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let yvars: Vec<G2Affine> = vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()];
        let scalar_xcoms: Commit1<F> = batch_commit_scalar_to_B1(&scalar_xvars, &crs, &mut rng);
        let ycoms: Commit2<F> = batch_commit_G2(&yvars, &crs, &mut rng);

        let equ: MSMEG2<F> = MSMEG2::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
            ],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
        };
        let proof: EquProof<F> =
            equ.prove(&scalar_xvars, &yvars, &scalar_xcoms, &ycoms, &crs, &mut rng);

        assert_eq!(proof.equ_type, EquType::MultiScalarG2);
    }

    #[test]
    fn test_MSMEG2_cproof_is_commit_and_prove() {
        let mut rng = test_rng();
        let mut rng2 = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let scalar_xvars: Vec<Fr> = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let yvars: Vec<G2Affine> = vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()];

        let equ: MSMEG2<F> = MSMEG2::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
            ],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
        };

        // Individually commit then prove
        let scalar_xcoms: Commit1<F> = batch_commit_scalar_to_B1(&scalar_xvars, &crs, &mut rng);
        let ycoms: Commit2<F> = batch_commit_G2(&yvars, &crs, &mut rng);
        let proof: EquProof<F> =
            equ.prove(&scalar_xvars, &yvars, &scalar_xcoms, &ycoms, &crs, &mut rng);
        let cproof = CProof::<F> {
            xcoms: scalar_xcoms,
            ycoms,
            equ_proofs: vec![proof],
        };

        // Mock calls to CRS to get them in sync
        let _ = CRS::<F>::generate_crs(&mut rng2);
        for _ in 0..scalar_xvars.len() {
            let _ = Fr::rand(&mut rng2);
        }
        for _ in 0..yvars.len() {
            let _ = Fr::rand(&mut rng2);
        }
        for _ in 0..equ.a_consts.len() {
            let _ = Fr::rand(&mut rng2);
        }
        for _ in 0..equ.b_consts.len() {
            let _ = Fr::rand(&mut rng2);
        }
        let _ = Fr::rand(&mut rng2);

        // Use the helper function to commit-and-prove in one step
        let cproof2 = equ.commit_and_prove(&scalar_xvars, &yvars, &crs, &mut rng2);

        assert_eq!(cproof, cproof2);
    }

    #[test]
    fn test_MSMEG2_proof_serde() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let scalar_xvars: Vec<Fr> = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let yvars: Vec<G2Affine> = vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()];
        let scalar_xcoms: Commit1<F> = batch_commit_scalar_to_B1(&scalar_xvars, &crs, &mut rng);
        let ycoms: Commit2<F> = batch_commit_G2(&yvars, &crs, &mut rng);

        let equ: MSMEG2<F> = MSMEG2::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
            ],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
        };
        let proof: EquProof<F> =
            equ.prove(&scalar_xvars, &yvars, &scalar_xcoms, &ycoms, &crs, &mut rng);

        // Serialize and deserialize the proof
        let mut c_bytes = Vec::new();
        proof.serialize_compressed(&mut c_bytes).unwrap();
        let proof_de = EquProof::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(proof, proof_de);

        let mut u_bytes = Vec::new();
        proof.serialize_uncompressed(&mut u_bytes).unwrap();
        let proof_de = EquProof::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(proof, proof_de);
    }

    #[test]
    fn test_quadratic_proof_type() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let scalar_xvars: Vec<Fr> = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let scalar_yvars: Vec<Fr> = vec![Fr::rand(&mut rng)];

        let equ: QuadEqu<F> = QuadEqu::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![Fr::rand(&mut rng), Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: Fr::rand(&mut rng),
        };

        // Individually commit then prove
        let scalar_xcoms: Commit1<F> = batch_commit_scalar_to_B1(&scalar_xvars, &crs, &mut rng);
        let scalar_ycoms: Commit2<F> = batch_commit_scalar_to_B2(&scalar_yvars, &crs, &mut rng);
        let proof: EquProof<F> = equ.prove(
            &scalar_xvars,
            &scalar_yvars,
            &scalar_xcoms,
            &scalar_ycoms,
            &crs,
            &mut rng,
        );

        assert_eq!(proof.equ_type, EquType::Quadratic);
    }

    #[test]
    fn test_quadratic_cproof_is_commit_and_prove() {
        let mut rng = test_rng();
        let mut rng2 = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let scalar_xvars: Vec<Fr> = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let scalar_yvars: Vec<Fr> = vec![Fr::rand(&mut rng)];

        let equ: QuadEqu<F> = QuadEqu::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![Fr::rand(&mut rng), Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: Fr::rand(&mut rng),
        };

        // Individually commit then prove
        let scalar_xcoms: Commit1<F> = batch_commit_scalar_to_B1(&scalar_xvars, &crs, &mut rng);
        let scalar_ycoms: Commit2<F> = batch_commit_scalar_to_B2(&scalar_yvars, &crs, &mut rng);
        let proof: EquProof<F> = equ.prove(
            &scalar_xvars,
            &scalar_yvars,
            &scalar_xcoms,
            &scalar_ycoms,
            &crs,
            &mut rng,
        );
        let cproof = CProof::<F> {
            xcoms: scalar_xcoms,
            ycoms: scalar_ycoms,
            equ_proofs: vec![proof],
        };

        // Mock calls to CRS to get them in sync
        let _ = CRS::<F>::generate_crs(&mut rng2);
        for _ in 0..scalar_xvars.len() {
            let _ = Fr::rand(&mut rng2);
        }
        for _ in 0..scalar_yvars.len() {
            let _ = Fr::rand(&mut rng2);
        }
        for _ in 0..equ.a_consts.len() {
            let _ = Fr::rand(&mut rng2);
        }
        for _ in 0..equ.b_consts.len() {
            let _ = Fr::rand(&mut rng2);
        }
        let _ = Fr::rand(&mut rng2);

        // Use the helper function to commit-and-prove in one step
        let cproof2 = equ.commit_and_prove(&scalar_xvars, &scalar_yvars, &crs, &mut rng2);

        assert_eq!(cproof, cproof2);
    }

    #[test]
    fn test_quadratic_proof_serde() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let scalar_xvars: Vec<Fr> = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let scalar_yvars: Vec<Fr> = vec![Fr::rand(&mut rng)];

        let equ: QuadEqu<F> = QuadEqu::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![Fr::rand(&mut rng), Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: Fr::rand(&mut rng),
        };

        // Individually commit then prove
        let scalar_xcoms: Commit1<F> = batch_commit_scalar_to_B1(&scalar_xvars, &crs, &mut rng);
        let scalar_ycoms: Commit2<F> = batch_commit_scalar_to_B2(&scalar_yvars, &crs, &mut rng);
        let proof: EquProof<F> = equ.prove(
            &scalar_xvars,
            &scalar_yvars,
            &scalar_xcoms,
            &scalar_ycoms,
            &crs,
            &mut rng,
        );

        // Serialize and deserialize the proof
        let mut c_bytes = Vec::new();
        proof.serialize_compressed(&mut c_bytes).unwrap();
        let proof_de = EquProof::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(proof, proof_de);

        let mut u_bytes = Vec::new();
        proof.serialize_uncompressed(&mut u_bytes).unwrap();
        let proof_de = EquProof::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(proof, proof_de);
    }
}

/*
 * NOTE:
 *
 * Proof verification tests are considered integration tests for the Groth-Sahai proof system.
 *
 * See tests/prover.rs for more details.
 */
