//! Contains the functionality for proving about the satisfiability of Groth-Sahai equations over bilinear groups.
//!
//! Abstractly, a proof for an equation for the SXDH instantiation of Groth-Sahai consists of the following values,
//! with respect to a pre-defined bilinear group `(A1, A2, AT)`:
//!
//! - `π`: 1-2 elements in [`B2`](crate::data_structures::Com2) (equiv. 2-4 elements in [`G2`](ark_ec::PairingEngine::G2Affine))
//!     which prove about the satisfiability of `A2` variables in the equation, and
//! - `θ`: 1-2 elements in [`B1`](crate::data_structures::Com1) (equiv. 2-4 elements in [`G1`](ark_ec::PairingEngine::G1Affine))
//!     which prove about the satisfiability of `A1` variables in the equation
//!
//! Computing these proofs primarily involves matrix multiplication in the [scalar field](ark_ec::PairingEngine::Fr) and in `B1` and `B2`.
//!
//! See the [`statement`](crate::statement) module for more details about the structure of the equations being proven about.

use ark_ec::PairingEngine;
use ark_std::{
    UniformRand,
    rand::{CryptoRng, Rng}
};

use crate::data_structures::*;
use crate::generator::CRS;
use crate::statement::*;
use super::commit::*;


/// A collection  of attributes containing prover functionality for an [`Equation`](crate::statement::Equation).
pub trait Provable<E: PairingEngine, A1, A2, AT> {

    /// Produces a proof `(π, θ)` that the `x` and `y` variables and their associated committments satisfy a single Groth-Sahai equation.
    fn prove<CR>(&self, x_vars: &Vec<A1>, y_vars: &Vec<A2>, x_coms: &Commit1<E>, y_coms: &Commit2<E>, crs: &CRS<E>, rng: &mut CR) -> EquProof<E>
        where
            CR: Rng + CryptoRng;
}

/// A witness-indistinguishable proof for a single [`Equation`](crate::statement::Equation).
pub struct EquProof<E: PairingEngine> {
    pub pi: Vec<Com2<E>>,
    pub theta: Vec<Com1<E>>,
    pub equ_type: EquType
}

/// A collection of proofs for Groth-Sahai compatible bilinear equations.
pub type Proof<E> = Vec<EquProof<E>>;

impl<E: PairingEngine> Provable<E, E::G1Affine, E::G2Affine, E::Fqk> for PPE<E> {

    fn prove<CR>(&self, x_vars: &Vec<E::G1Affine>, y_vars: &Vec<E::G2Affine>, x_coms: &Commit1<E>, y_coms: &Commit2<E>, crs: &CRS<E>, rng: &mut CR) -> EquProof<E> 
    where
        CR: Rng + CryptoRng
    {
        // Gamma is an (m x n) matrix with m x variables and n y variables
        // x's commit randomness (i.e. R) is a (m x 2) matrix
        assert_eq!(x_vars.len(), x_coms.rand.len());
        assert_eq!(self.gamma.len(), x_coms.rand.len());
        assert_eq!(x_coms.rand[0].len(), 2);
        let _m = x_vars.len();
        // y's commit randomness (i.e. S) is a (n x 2) matrix
        assert_eq!(y_vars.len(), y_coms.rand.len());
        assert_eq!(self.gamma[0].len(), y_coms.rand.len());
        assert_eq!(y_coms.rand[0].len(), 2);
        let _n = y_vars.len();

        let is_parallel = true;

        // (2 x m) field matrix R^T, in GS parlance
        let x_rand_trans = x_coms.rand.transpose();
        // (2 x n) field matrix S^T, in GS parlance
        let y_rand_trans = y_coms.rand.transpose();
        // (2 x 2) field matrix T, in GS parlance
        let pf_rand: Matrix<E::Fr> = vec![
            vec![ E::Fr::rand(rng), E::Fr::rand(rng) ],
            vec![ E::Fr::rand(rng), E::Fr::rand(rng) ]
        ];

        // (2 x 1) Com2 matrix
        let x_rand_lin_b = vec_to_col_vec(&Com2::<E>::batch_linear_map(&self.b_consts)).left_mul(&x_rand_trans, is_parallel);

        // (2 x n) field matrix
        let x_rand_stmt = x_rand_trans.right_mul(&self.gamma, is_parallel);
        // (2 x 1) Com2 matrix
        let x_rand_stmt_lin_y = vec_to_col_vec(&Com2::<E>::batch_linear_map(&y_vars)).left_mul(&x_rand_stmt, is_parallel);

        // (2 x 2) field matrix
        let pf_rand_stmt = x_rand_trans.right_mul(&self.gamma, is_parallel).right_mul(&y_coms.rand, is_parallel).add(&pf_rand.transpose().neg());
        // (2 x 1) Com2 matrix
        let pf_rand_stmt_com2 = vec_to_col_vec(&crs.v).left_mul(&pf_rand_stmt, is_parallel);

        let pi = col_vec_to_vec(&x_rand_lin_b.add(&x_rand_stmt_lin_y).add(&pf_rand_stmt_com2));
        assert_eq!(pi.len(), 2);

        // (2 x 1) Com1 matrix
        let y_rand_lin_a = vec_to_col_vec(&Com1::<E>::batch_linear_map(&self.a_consts)).left_mul(&y_rand_trans, is_parallel);

        // (2 x m) field matrix
        let y_rand_stmt = y_rand_trans.right_mul(&self.gamma.transpose(), is_parallel);
        // (2 x 1) Com1 matrix
        let y_rand_stmt_lin_x = vec_to_col_vec(&Com1::<E>::batch_linear_map(&x_vars)).left_mul(&y_rand_stmt, is_parallel);

        // (2 x 1) Com1 matrix
        let pf_rand_com1 = vec_to_col_vec(&crs.u).left_mul(&pf_rand, is_parallel);

        let theta = col_vec_to_vec(&y_rand_lin_a.add(&y_rand_stmt_lin_x).add(&pf_rand_com1));
        assert_eq!(theta.len(), 2);

        EquProof::<E> {
            pi,
            theta,
            equ_type: EquType::PairingProduct
        }
    }
}

impl<E: PairingEngine> Provable<E, E::G1Affine, E::Fr, E::G1Affine> for MSMEG1<E> {

    fn prove<CR>(&self, x_vars: &Vec<E::G1Affine>, scalar_y_vars: &Vec<E::Fr>, x_coms: &Commit1<E>, scalar_y_coms: &Commit2<E>, crs: &CRS<E>, rng: &mut CR) -> EquProof<E> 
    where
        CR: Rng + CryptoRng
    {
        // Gamma is an (m x n') matrix with m x variables and n' scalar y variables
        // x's commit randomness (i.e. R) is a (m x 2) matrix
        assert_eq!(x_vars.len(), x_coms.rand.len());
        assert_eq!(self.gamma.len(), x_coms.rand.len());
        assert_eq!(x_coms.rand[0].len(), 2);
        let _m = x_vars.len();
        // scalar y's commit randomness (i.e. s) is a (n' x 1) matrix (i.e. column vector)
        assert_eq!(scalar_y_vars.len(), scalar_y_coms.rand.len());
        assert_eq!(self.gamma[0].len(), scalar_y_coms.rand.len());
        assert_eq!(scalar_y_coms.rand[0].len(), 1);
        let _n_prime = scalar_y_vars.len();

        let is_parallel = true;

        // (2 x m) field matrix R^T, in GS parlance
        let x_rand_trans = x_coms.rand.transpose();
        // (1 x n') field matrix s^T, in GS parlance
        let y_rand_trans = scalar_y_coms.rand.transpose();
        // (1 x 2) field matrix T, in GS parlance
        let pf_rand: Matrix<E::Fr> = vec![
            vec![ E::Fr::rand(rng), E::Fr::rand(rng) ],
        ];

        // (2 x 1) Com2 matrix
        let x_rand_lin_b = vec_to_col_vec(&Com2::<E>::batch_scalar_linear_map(&self.b_consts, &crs)).left_mul(&x_rand_trans, is_parallel);

        // (2 x n) field matrix
        let x_rand_stmt = x_rand_trans.right_mul(&self.gamma, is_parallel);
        // (2 x 1) Com2 matrix
        let x_rand_stmt_lin_y = vec_to_col_vec(&Com2::<E>::batch_scalar_linear_map(&scalar_y_vars, &crs)).left_mul(&x_rand_stmt, is_parallel);

        // (2 x 1) field matrix
        let pf_rand_stmt = x_rand_trans.right_mul(&self.gamma, is_parallel).right_mul(&scalar_y_coms.rand, is_parallel).add(&pf_rand.transpose().neg());
        // (2 x 1) Com2 matrix
        let v1: Matrix<Com2<E>> = vec![vec![crs.v[0]]];
        let pf_rand_stmt_com2 = v1.left_mul(&pf_rand_stmt, is_parallel);

        let pi = col_vec_to_vec(&x_rand_lin_b.add(&x_rand_stmt_lin_y).add(&pf_rand_stmt_com2));
        assert_eq!(pi.len(), 2);

        // (1 x 1) Com1 matrix
        let y_rand_lin_a = vec_to_col_vec(&Com1::<E>::batch_linear_map(&self.a_consts)).left_mul(&y_rand_trans, is_parallel);

        // (1 x m) field matrix
        let y_rand_stmt = y_rand_trans.right_mul(&self.gamma.transpose(), is_parallel);
        // (1 x 1) Com1 matrix
        let y_rand_stmt_lin_x = vec_to_col_vec(&Com1::<E>::batch_linear_map(&x_vars)).left_mul(&y_rand_stmt, is_parallel);

        // (1 x 1) Com1 matrix
        let pf_rand_com1 = vec_to_col_vec(&crs.u).left_mul(&pf_rand, is_parallel);

        let theta = col_vec_to_vec(&y_rand_lin_a.add(&y_rand_stmt_lin_x).add(&pf_rand_com1));
        assert_eq!(theta.len(), 1);

        EquProof::<E> {
            pi,
            theta,
            equ_type: EquType::MultiScalarG1
        }
    }
}

impl<E: PairingEngine> Provable<E, E::Fr, E::G2Affine, E::G2Affine> for MSMEG2<E> {

    fn prove<CR>(&self, scalar_x_vars: &Vec<E::Fr>, y_vars: &Vec<E::G2Affine>, scalar_x_coms: &Commit1<E>, y_coms: &Commit2<E>, crs: &CRS<E>, rng: &mut CR) -> EquProof<E> 
    where
        CR: Rng + CryptoRng
    {
        // Gamma is an (m' x n) matrix with m' x variables and n y variables
        // x's commit randomness (i.e. r) is a (m' x 1) matrix (i.e. column vector)
        assert_eq!(scalar_x_vars.len(), scalar_x_coms.rand.len());
        assert_eq!(self.gamma.len(), scalar_x_coms.rand.len());
        assert_eq!(scalar_x_coms.rand[0].len(), 1);
        let _m_prime = scalar_x_vars.len();
        // y's commit randomness (i.e. S) is a (n x 2) matrix
        assert_eq!(y_vars.len(), y_coms.rand.len());
        assert_eq!(self.gamma[0].len(), y_coms.rand.len());
        assert_eq!(y_coms.rand[0].len(), 2);
        let _n = y_vars.len();

        let is_parallel = true;

        // (1 x m') field matrix r^T, in GS parlance
        let x_rand_trans = scalar_x_coms.rand.transpose();
        // (2 x n) field matrix S^T, in GS parlance
        let y_rand_trans = y_coms.rand.transpose();
        // (2 x 1) field matrix T, in GS parlance
        let pf_rand: Matrix<E::Fr> = vec![
            vec![ E::Fr::rand(rng) ],
            vec![ E::Fr::rand(rng) ]
        ];

        // (1 x 1) Com2 matrix
        let x_rand_lin_b = vec_to_col_vec(&Com2::<E>::batch_linear_map(&self.b_consts)).left_mul(&x_rand_trans, is_parallel);

        // (1 x n) field matrix
        let x_rand_stmt = x_rand_trans.right_mul(&self.gamma, is_parallel);
        // (1 x 1) Com2 matrix
        let x_rand_stmt_lin_y = vec_to_col_vec(&Com2::<E>::batch_linear_map(&y_vars)).left_mul(&x_rand_stmt, is_parallel);

        // (1 x 2) field matrix
        let pf_rand_stmt = x_rand_trans.right_mul(&self.gamma, is_parallel).right_mul(&y_coms.rand, is_parallel).add(&pf_rand.transpose().neg());
        // (1 x 1) Com2 matrix
        let pf_rand_stmt_com2 = vec_to_col_vec(&crs.v).left_mul(&pf_rand_stmt, is_parallel);

        let pi = col_vec_to_vec(&x_rand_lin_b.add(&x_rand_stmt_lin_y).add(&pf_rand_stmt_com2));
        assert_eq!(pi.len(), 1);

        // (2 x 1) Com1 matrix
        let y_rand_lin_a = vec_to_col_vec(&Com1::<E>::batch_scalar_linear_map(&self.a_consts, &crs)).left_mul(&y_rand_trans, is_parallel);

        // (2 x m') field matrix
        let y_rand_stmt = y_rand_trans.right_mul(&self.gamma.transpose(), is_parallel);
        // (2 x 1) Com1 matrix
        let y_rand_stmt_lin_x = vec_to_col_vec(&Com1::<E>::batch_scalar_linear_map(&scalar_x_vars, &crs)).left_mul(&y_rand_stmt, is_parallel);

        // (2 x 1) Com1 matrix
        let u1: Matrix<Com1<E>> = vec![vec![crs.u[0]]];
        let pf_rand_com1 = u1.left_mul(&pf_rand, is_parallel);

        let theta = col_vec_to_vec(&y_rand_lin_a.add(&y_rand_stmt_lin_x).add(&pf_rand_com1));
        assert_eq!(theta.len(), 2);

        EquProof::<E> {
            pi,
            theta,
            equ_type: EquType::MultiScalarG2
        }
    }
}

impl<E: PairingEngine> Provable<E, E::Fr, E::Fr, E::Fr> for QuadEqu<E> {

    fn prove<CR>(&self, scalar_x_vars: &Vec<E::Fr>, scalar_y_vars: &Vec<E::Fr>, scalar_x_coms: &Commit1<E>, scalar_y_coms: &Commit2<E>, crs: &CRS<E>, rng: &mut CR) -> EquProof<E> 
    where
        CR: Rng + CryptoRng
    {
        // Gamma is an (m' x n') matrix with m' x variables and n' y variables
        // x's commit randomness (i.e. r) is a (m' x 1) matrix (i.e. column vector)
        assert_eq!(scalar_x_vars.len(), scalar_x_coms.rand.len());
        assert_eq!(self.gamma.len(), scalar_x_coms.rand.len());
        assert_eq!(scalar_x_coms.rand[0].len(), 1);
        let _m_prime = scalar_x_vars.len();
        // y's commit randomness (i.e. s) is a (n' x 1) matrix (i.e. column vector)
        assert_eq!(scalar_y_vars.len(), scalar_y_coms.rand.len());
        assert_eq!(self.gamma[0].len(), scalar_y_coms.rand.len());
        assert_eq!(scalar_y_coms.rand[0].len(), 1);
        let _n_prime = scalar_y_vars.len();

        let is_parallel = true;

        // (1 x m') field matrix r^T, in GS parlance
        let x_rand_trans = scalar_x_coms.rand.transpose();
        // (1 x n') field matrix s^T, in GS parlance
        let y_rand_trans = scalar_y_coms.rand.transpose();
        // field element T, in GS parlance
        let pf_rand: Matrix<E::Fr> = vec![vec![E::Fr::rand(rng)]];

        let x_rand_lin_b = vec_to_col_vec(&Com2::<E>::batch_scalar_linear_map(&self.b_consts, &crs)).left_mul(&x_rand_trans, is_parallel);

        // (1 x n') field matrix
        let x_rand_stmt = x_rand_trans.right_mul(&self.gamma, is_parallel);
        // (1 x 1) Com2 matrix
        let x_rand_stmt_lin_y = vec_to_col_vec(&Com2::<E>::batch_scalar_linear_map(&scalar_y_vars, &crs)).left_mul(&x_rand_stmt, is_parallel);

        // (1 x 2) field matrix
        let pf_rand_stmt = x_rand_trans.right_mul(&self.gamma, is_parallel).right_mul(&scalar_y_coms.rand, is_parallel).add(&pf_rand.transpose().neg());
        let v1: Matrix<Com2<E>> = vec![vec![crs.v[0]]];
        // (1 x 1) Com2 matrix
        let pf_rand_stmt_com2 = v1.left_mul(&pf_rand_stmt, is_parallel);

        let pi = col_vec_to_vec(&x_rand_lin_b.add(&x_rand_stmt_lin_y).add(&pf_rand_stmt_com2));
        assert_eq!(pi.len(), 1);

        // (1 x 1) Com1 matrix
        let y_rand_lin_a = vec_to_col_vec(&Com1::<E>::batch_scalar_linear_map(&self.a_consts, &crs)).left_mul(&y_rand_trans, is_parallel);

        // (1 x m') field matrix
        let y_rand_stmt = y_rand_trans.right_mul(&self.gamma.transpose(), is_parallel);
        // (1 x 1) Com1 matrix
        let y_rand_stmt_lin_x = vec_to_col_vec(&Com1::<E>::batch_scalar_linear_map(&scalar_x_vars, &crs)).left_mul(&y_rand_stmt, is_parallel);

        // (1 x 1) Com1 matrix
        let u1: Matrix<Com1<E>> = vec![vec![crs.u[0]]];
        let pf_rand_com1 = u1.left_mul(&pf_rand, is_parallel);

        let theta = col_vec_to_vec(&y_rand_lin_a.add(&y_rand_stmt_lin_x).add(&pf_rand_com1));
        assert_eq!(theta.len(), 1);

        EquProof::<E> {
            pi,
            theta,
            equ_type: EquType::Quadratic
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use ark_bls12_381::{Bls12_381 as F};
    use ark_ec::{PairingEngine, AffineCurve, ProjectiveCurve};
    use ark_ff::{UniformRand, Zero, One};
    use ark_std::test_rng;

    use super::*;

    type G1Affine = <F as PairingEngine>::G1Affine;
    type G2Affine = <F as PairingEngine>::G2Affine;
    type Fr = <F as PairingEngine>::Fr;
    type Fqk = <F as PairingEngine>::Fqk;

    #[test]
    fn test_PPE_proof_type() {

        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let xvars: Vec<G1Affine> = vec![
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()
        ];
        let yvars: Vec<G2Affine> = vec![
            crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()
        ];
        let xcoms: Commit1<F> = batch_commit_G1(&xvars, &crs, &mut rng);
        let ycoms: Commit2<F> = batch_commit_G2(&yvars, &crs, &mut rng);

        let equ: PPE<F> = PPE::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(), crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: Fqk::rand(&mut rng)
        };
        let proof: EquProof<F> = equ.prove(&xvars, &yvars, &xcoms, &ycoms, &crs, &mut rng);

        assert_eq!(proof.equ_type, EquType::PairingProduct);
    }

    #[test]
    fn test_MSMEG1_proof_type() {

        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let xvars: Vec<G1Affine> = vec![
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
            crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()
        ];
        let scalar_yvars: Vec<Fr> = vec![
            Fr::rand(&mut rng)
        ];
        let xcoms: Commit1<F> = batch_commit_G1(&xvars, &crs, &mut rng);
        let scalar_ycoms: Commit2<F> = batch_commit_scalar_to_B2(&scalar_yvars, &crs, &mut rng);

        let equ: MSMEG1<F> = MSMEG1::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![Fr::rand(&mut rng), Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()
        };
        let proof: EquProof<F> = equ.prove(&xvars, &scalar_yvars, &xcoms, &scalar_ycoms, &crs, &mut rng);

        assert_eq!(proof.equ_type, EquType::MultiScalarG1);
    }

    #[test]
    fn test_MSMEG2_proof_type() {

        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let scalar_xvars: Vec<Fr> = vec![
            Fr::rand(&mut rng),
            Fr::rand(&mut rng)
        ];
        let yvars: Vec<G2Affine> = vec![
            crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()
        ];
        let scalar_xcoms: Commit1<F> = batch_commit_scalar_to_B1(&scalar_xvars, &crs, &mut rng);
        let ycoms: Commit2<F> = batch_commit_G2(&yvars, &crs, &mut rng);

        let equ: MSMEG2<F> = MSMEG2::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
                crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()
            ],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()
        };
        let proof: EquProof<F> = equ.prove(&scalar_xvars, &yvars, &scalar_xcoms, &ycoms, &crs, &mut rng);

        assert_eq!(proof.equ_type, EquType::MultiScalarG2);
    }

    #[test]
    fn test_quadratic_proof_type() {

        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let scalar_xvars: Vec<Fr> = vec![
            Fr::rand(&mut rng),
            Fr::rand(&mut rng)
        ];
        let scalar_yvars: Vec<Fr> = vec![
            Fr::rand(&mut rng)
        ];
        let scalar_xcoms: Commit1<F> = batch_commit_scalar_to_B1(&scalar_xvars, &crs, &mut rng);
        let scalar_ycoms: Commit2<F> = batch_commit_scalar_to_B2(&scalar_yvars, &crs, &mut rng);

        let equ: QuadEqu<F> = QuadEqu::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![
                Fr::rand(&mut rng),
                Fr::rand(&mut rng)
            ],
            gamma: vec![vec![Fr::one()], vec![Fr::zero()]],
            target: Fr::rand(&mut rng)
        };
        let proof: EquProof<F> = equ.prove(&scalar_xvars, &scalar_yvars, &scalar_xcoms, &scalar_ycoms, &crs, &mut rng);

        assert_eq!(proof.equ_type, EquType::Quadratic);
    }
}

/*
 * NOTE:
 *
 * Proof verification tests are considered integration tests for the Groth-Sahai proof system.
 *
 * See tests/prover.rs for more details.
 */
