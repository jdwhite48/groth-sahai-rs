//! Contains the functionality for using Groth-Sahai proofs about the satisfiability of bilinear equations.
//!
//! A Groth-Sahai statement is a list of [`Equations`](self::Equation) with the abstract form `(A * Y)(X * B)(X * Γ Y) = t`, where:
//!
//! - `A` and `B` are vectors representing public constants in the equation,
//! - `X` and `Y` are vectors representing private variables in the equation (introduced on prove),
//! - `Γ` is a matrix of public [scalar](ark_ec::PairingEngine::Fr) constants defining how to scalar multiply
//!     the corresponding variables being paired together,
//! - `t` is a public constant representing the RHS of the equation, and
//! - `*` is the specified pairing, applied entry-wise to the corresponding elements in each vector.
//!
//! Each [`Equation`](self::Equation) contains the public components of the equation to be proven
//! and must be one of the following four types, each defined over a bilinear group:
//!
//! 1) **Pairing-product equation** ([`PPE`](self::PPE)):&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `(G1, G2, GT)` with
//!     [`e`](ark_ec::PairingEngine::pairing)` : G1 x G2 -> GT` as the equipped pairing.
//! 2) **Multi-scalar mult. equation in G1** ([`MSMEG1`](self::MSMEG1)):&emsp;`(G1, Fr, G1)`
//!     with [point-scalar multiplication](ark_ec::AffineCurve::mul) as the equipped pairing.
//! 3) **Multi-scalar mult. equation in G2** ([`MSMEG2`](self::MSMEG2)):&emsp;`(Fr, G2, G2)`
//!     with [point-scalar multiplication](ark_ec::AffineCurve::mul) as the equipped pairing.
//! 4) **Quadratic equation** ([`Quad`](self::Quad)):&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;`(Fr, Fr, Fr)`
//!     with [scalar](ark_ec::PairingEngine::Fr) multiplication as the equipped pairing.
//!
//! The Groth-Sahai proof system expects that **each** equation is defined with respect to the list of variables
//! that span across **ALL** equations being proven about. For example, if one wishes to prove
//! about 1 PPE equation and 2 MSMEG2 equations collectively containing `m` `X` variables in `G1`,
//! `n` `Y` variables in `G2`, and `m'` `x` variables in `Fr`, then the PPE equation would need
//! `Γ` to be a `m` by `n` matrix and the MSMEG2 equations would need `Γ` to be `m'` by `n` matrices.
//!
//! **NOTE**: The bilinear equation may need to be re-arranged using the properties
//! of bilinear group arithmetic and pairings in order to form a valid Groth-Sahai statement.
//! This API does not provide such functionality.

use ark_ec::{PairingEngine};
use ark_std::{
    UniformRand,
    rand::{CryptoRng, Rng}
};

use crate::data_structures::*;
use crate::GSType;
use crate::generator::CRS;
use crate::commit::*;

/// A marker trait for an arbitrary Groth-Sahai [`Equation`](self::Equation).
pub trait Equ {}

/// A single equation, defined over an arbitrary bilinear group `(A1, A2, AT)`, that forms
/// the atomic unit for a [`Statement`](self::Statement).
pub trait Equation<E: PairingEngine, A1, A2, AT>:
    Equ
{

    /// Produce a proof `(π, θ)` that the x and y variables satisfy a single Groth-Sahai statement / equation.
    fn prove<CR>(&self, x_vars: &Vec<A1>, y_vars: &Vec<A2>, x_coms: &Commit1<E>, y_coms: &Commit2<E>, crs: &CRS<E>, rng: &mut CR) -> EquProof<E>
        where
            CR: Rng + CryptoRng;
    /// Verify that a single Groth-Sahai equation is satisfied by the prover's variables.
    fn verify(&self, proof: &EquProof<E>, x_coms: &Commit1<E>, y_coms: &Commit2<E>, crs: &CRS<E>) -> bool;
    fn get_type(&self) -> GSType;
}

/// A collection of Groth-Sahai compatible bilinear [`Equations`](self::Equation).
pub type Statement = Vec<dyn Equ>;

/// A witness-indistinguishable proof for a single [`Equation`](self::Equation).
pub struct EquProof<E: PairingEngine> {
    pub pi: Vec<Com2<E>>,
    pub theta: Vec<Com1<E>>,
    pub equ_type: GSType
}

/// A collection of proofs for Groth-Sahai compatible bilinear equations.
pub type Proof<E> = Vec<EquProof<E>>;

/// A pairing-product equation, equipped with the bilinear group pairing
/// [`e`](ark_ec::PairingEngine::pairing)` : G1 x G2 -> GT`.
///
/// For example, the equation `e(W, N) * e(U, V)^5 = t_T` can be expressed by the following
/// (private) witness variables `X = [U, W]`, `Y = [V]`, (public) constants `A = [0]`, `B = [0, N]`,
/// pairing exponent matrix `Γ = [[5], [0]]`, and `target = t_T` in `GT`.
pub struct PPE<E: PairingEngine> {
    pub a_consts: Vec<E::G1Affine>,
    pub b_consts: Vec<E::G2Affine>,
    pub gamma: Matrix<E::Fr>,
    pub target: E::Fqk
}

impl<E: PairingEngine> Equ for PPE<E> {}
impl<E: PairingEngine> Equation<E, E::G1Affine, E::G2Affine, E::Fqk> for PPE<E> {

    #[inline(always)]
    fn get_type(&self) -> GSType {
        GSType::PairingProduct
    }

    fn prove<CR>(&self, x_vars: &Vec<E::G1Affine>, y_vars: &Vec<E::G2Affine>, x_coms: &Commit1<E>, y_coms: &Commit2<E>, crs: &CRS<E>, rng: &mut CR) -> EquProof<E> 
    where
        CR: Rng + CryptoRng
    {
        // Gamma is an (m x n) matrix with m x variables and n y variables
        // x's commit randomness (i.e. R) is a (m x 2) matrix
        assert_eq!(x_vars.len(), x_coms.rand.len());
        assert_eq!(self.gamma.len(), x_coms.rand.len());
        let _m = x_vars.len();
        // y's commit randomness (i.e. S) is a (n x 2) matrix
        assert_eq!(y_vars.len(), y_coms.rand.len());
        assert_eq!(self.gamma[0].len(), y_coms.rand.len());
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
            equ_type: self.get_type()
        }
    }

    fn verify(&self, proof: &EquProof<E>, x_coms: &Commit1<E>, y_coms: &Commit2<E>, crs: &CRS<E>) -> bool {

        let is_parallel = true;

        let lin_a_com_y = ComT::<E>::pairing_sum(&Com1::<E>::batch_linear_map(&self.a_consts), &y_coms.coms);

        let com_x_lin_b = ComT::<E>::pairing_sum(&x_coms.coms, &Com2::<E>::batch_linear_map(&self.b_consts));

        let stmt_com_y: Matrix<Com2<E>> = vec_to_col_vec(&y_coms.coms).left_mul(&self.gamma, is_parallel);
        let com_x_stmt_com_y = ComT::<E>::pairing_sum(&x_coms.coms, &col_vec_to_vec(&stmt_com_y));

        let lin_t = ComT::<E>::linear_map_PPE(&self.target);

        let com1_pf2 = ComT::<E>::pairing_sum(&crs.u, &proof.pi);

        let pf1_com2 = ComT::<E>::pairing_sum(&proof.theta, &crs.v);

        let lhs: ComT<E> = lin_a_com_y + com_x_lin_b + com_x_stmt_com_y;
        let rhs: ComT<E> = lin_t + com1_pf2 + pf1_com2;

        lhs == rhs
    }
}

/// A multi-scalar multiplication equation in [`G1`](ark_ec::PairingEngine::G1Affine), equipped with point-scalar multiplication as pairing.
///
/// For example, the equation `n * W + (v * U)^5 = t_1` can be expressed by the following
/// (private) witness variables `X = [U, W]`, `Y = [v]`, (public) constants `A = [0]`, `B = [0, n]`,
/// pairing exponent matrix `Γ = [[5], [0]]`, and `target = t_1` in `G1`.
pub struct MSMEG1<E: PairingEngine> {
    pub a_consts: Vec<E::G1Affine>,
    pub b_consts: Vec<E::Fr>,
    pub gamma: Matrix<E::Fr>,
    pub target: E::G1Affine
}
impl<E: PairingEngine> Equ for MSMEG1<E> {}
impl<E: PairingEngine> Equation<E, E::G1Affine, E::Fr, E::G1Affine> for MSMEG1<E> {

    #[inline(always)]
    fn get_type(&self) -> GSType {
        GSType::MultiScalarG1
    }

    fn prove<CR>(&self, x_vars: &Vec<E::G1Affine>, scalar_y_vars: &Vec<E::Fr>, x_coms: &Commit1<E>, scalar_y_coms: &Commit2<E>, crs: &CRS<E>, rng: &mut CR) -> EquProof<E> 
    where
        CR: Rng + CryptoRng
    {
        // Gamma is an (m x n') matrix with m x variables and n' scalar y variables
        // x's commit randomness (i.e. R) is a (m x 2) matrix
        assert_eq!(x_vars.len(), x_coms.rand.len());
        assert_eq!(self.gamma.len(), x_coms.rand.len());
        let _m = x_vars.len();
        // scalar y's commit randomness (i.e. s) is a (n' x 1) matrix (i.e. column vector)
        assert_eq!(scalar_y_vars.len(), scalar_y_coms.rand.len());
        assert_eq!(self.gamma[0].len(), scalar_y_coms.rand.len());
        let _n_prime = scalar_y_vars.len();

        let is_parallel = true;

        // (2 x m) field matrix R^T, in GS parlance
        let x_rand_trans = x_coms.rand.transpose();
        // (2 x n') field matrix s^T, in GS parlance
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
            equ_type: self.get_type()
        }
    }

    fn verify(&self, proof: &EquProof<E>, x_coms: &Commit1<E>, scalar_y_coms: &Commit2<E>, crs: &CRS<E>) -> bool {

        let is_parallel = true;

        let lin_a_com_y = ComT::<E>::pairing_sum(&Com1::<E>::batch_linear_map(&self.a_consts), &scalar_y_coms.coms);

        let com_x_lin_b = ComT::<E>::pairing_sum(&x_coms.coms, &Com2::<E>::batch_scalar_linear_map(&self.b_consts, &crs));

        let stmt_com_y: Matrix<Com2<E>> = vec_to_col_vec(&scalar_y_coms.coms).left_mul(&self.gamma, is_parallel);
        let com_x_stmt_com_y = ComT::<E>::pairing_sum(&x_coms.coms, &col_vec_to_vec(&stmt_com_y));

        let lin_t = ComT::<E>::linear_map_MSMEG1(&self.target, &crs);

        let com1_pf2 = ComT::<E>::pairing_sum(&crs.u, &proof.pi);

        let pf1_com2 = ComT::<E>::pairing(proof.theta[0].clone(), crs.v[0].clone());

        let lhs: ComT<E> = lin_a_com_y + com_x_lin_b + com_x_stmt_com_y;
        let rhs: ComT<E> = lin_t + com1_pf2 + pf1_com2;

        lhs == rhs
    }
}

/*
/// A multi-scalar multiplication equation in [`G2`](ark_ec::PairingEngine::G2Affine), equipped with point-scalar multiplication as pairing.
///
/// For example, the equation `w * N + (u * V)^5 = t_2` can be expressed by the following
/// (private) witness variables `X = [u, w]`, `Y = [V]`, (public) constants `A = [0]`, `B = [0, N]`,
/// pairing exponent matrix `Γ = [[5], [0]]`, and `target = t_2` in `G2`.
pub struct MSMEG2<E: PairingEngine> {
    a_consts: Vec<E::Fr>,
    b_consts: Vec<E::G2Affine>,
    gamma: Matrix<E::Fr>,
    target: E::G2Affine
}
impl<E: PairingEngine> Equ for MSMEG2<E> {}
impl<E: PairingEngine> Equation<E, E::Fr, E::G2Affine, E::G2Affine> for MSMEG2<E> {}

/// A quadratic equation in the [scalar field](ark_ec::PairingEngine::Fr), equipped with field multiplication as pairing.
///
/// For example, the equation `w * n + (u * v)^5 = t_p` can be expressed by the following
/// (private) witness variables `X = [u, w]`, `Y = [v]`, (public) constants `A = [0]`, `B = [0, n]`,
/// pairing exponent matrix `Γ = [[5], [0]]`, and `target = t_p` in `Fr`.
pub struct QuadEqu<E: PairingEngine> {
    a_consts: Vec<E::Fr>,
    b_consts: Vec<E::Fr>,
    gamma: Matrix<E::Fr>,
    target: E::Fr
}
impl<E: PairingEngine> Equ for QuadEqu<E> {}
impl<E: PairingEngine> Equation<E, E::Fr, E::Fr, E::Fr> for QuadEqu<E> {}
*/


#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use ark_bls12_381::{Bls12_381 as F};
    use ark_ec::{PairingEngine, AffineCurve, ProjectiveCurve};
    use ark_ff::{UniformRand, Zero, One};
    use ark_std::test_rng;

    use super::*;
    use crate::GSType;

    type G1Affine = <F as PairingEngine>::G1Affine;
    type G2Affine = <F as PairingEngine>::G2Affine;
    type Fr = <F as PairingEngine>::Fr;
    type Fqk = <F as PairingEngine>::Fqk;

    #[test]
    fn test_PPE_equation_type() {

        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let equ: PPE<F> = PPE::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()],
            gamma: vec![vec![Fr::rand(&mut rng)]],
            target: Fqk::rand(&mut rng)
        };

        assert_eq!(equ.get_type(), GSType::PairingProduct);
    }

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

        assert_eq!(proof.equ_type, GSType::PairingProduct);
    }
}
