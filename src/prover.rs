//! Contains types and functions for proving about the satisfiability of bilinear equations.
//!
//! An equation has the abstract form `(A * Y)(X * B)(X * Γ Y) = target`
//! where `A` and `B` are vectors representing public constants in the equation,
//! `X` and `Y` are are vectors representing private variables in the equation (see
//! [`Witness`](self::Witness)),
//! `Γ` is a matrix of public constants defining how the pairing applies to the variables, and
//! `target` is a public constant representing the RHS of the equation.
//!
//! A Groth-Sahai proof consists of a set of individual proofs, one for each bilinear equation.
//!
//! NOTE: The bilinear equation may need to be re-arranged using the properties
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

pub trait Equ {}
/// A single Groth-Sahai statement.
pub trait Equation<E: PairingEngine, A1, A2, AT>: Equ {

    // TODO: Consider binding the variables together into a witness struct
    // TODO: Consider binding the commitment and variables together in API
    // TODO: Expose option to parallelize proof computations in API
    /// Produce a proof `(π, θ)` that the x and y variables satisfy a single Groth-Sahai statement / equation.
    fn prove<CR>(&self, x_vars: &Vec<A1>, y_vars: &Vec<A2>, x_coms: &Commit1<E>, y_coms: &Commit2<E>, crs: &CRS<E>, rng: &mut CR) -> EquProof<E> where CR: Rng + CryptoRng;
    fn get_type(&self) -> GSType;
}
/// A collection of Groth-Sahai compatible bilinear equations.

pub type Statement = Vec<dyn Equ>;

/// A Groth-Sahai witness, expressed as variables across one or more [`Equation`](self::Equ)s.
pub struct Witness<A1, A2> {
    x_vars: Vec<A1>,
    y_vars: Vec<A2>
}
/// A witness-indistinguishable proof for a single [`Equation`](self::Equ).
pub struct EquProof<E: PairingEngine> {
    pub pi: Vec<Com2<E>>,
    pub theta: Vec<Com1<E>>,
    pub equ_type: GSType
}

/// A collection of proofs for Groth-Sahai compatible bilinear equations.
pub type Proof<E> = Vec<EquProof<E>>;

/// A pairing product equation, equipped with the bilinear group pairing as pairing.
///
/// For example, the equation `e(W, N) * e(U, V)^5 = t_T` can be expressed by the following
/// (private) witness variables `X = [U, W]`, `Y = [V]`, (public) constants `A = [0]`, `B = [0, N]`,
/// pairing exponent matrix `gamma = [[5], [0]]`, and `target = t_T` in `GT`.
pub struct PPE<E: PairingEngine> {
    pub a_consts: Vec<E::G1Affine>,
    pub b_consts: Vec<E::G2Affine>,
    pub gamma: Matrix<E::Fr>,
    pub target: E::Fqk
}

// TODO: Batch prove for any kind of equation, more efficiently than just appending Vec's of
// individual vars / commitments (would require the user restructuring their Gamma's and A/B
// constants to consider list of variables for all equations

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
        let pf_rand_stmt_com2 = crs.v.left_mul(&pf_rand_stmt, is_parallel);

        let pi = col_vec_to_vec(&x_rand_lin_b.add(&x_rand_stmt_lin_y).add(&pf_rand_stmt_com2));
        assert_eq!(pi.len(), 2);

        // (2 x 1) Com1 matrix
        let y_rand_lin_a = vec_to_col_vec(&Com1::<E>::batch_linear_map(&self.a_consts)).left_mul(&y_rand_trans, is_parallel);

        // (2 x m) field matrix
        let y_rand_stmt = y_rand_trans.right_mul(&self.gamma.transpose(), is_parallel);
        // (2 x 1) Com1 matrix
        let y_rand_stmt_lin_x = vec_to_col_vec(&Com1::<E>::batch_linear_map(&x_vars)).left_mul(&y_rand_stmt, is_parallel);

        // (2 x 1) Com1 matrix
        let pf_rand_com1 = crs.u.left_mul(&pf_rand, is_parallel);

        let theta = col_vec_to_vec(&y_rand_lin_a.add(&y_rand_stmt_lin_x).add(&pf_rand_com1));
        assert_eq!(theta.len(), 2);

        EquProof::<E> {
            pi,
            theta,
            equ_type: self.get_type()
        }
    }
}

/*
/// A multi-scalar multiplication equation in [`G1`](ark_ec::PairingEngine::G1Affine), equipped with point-scalar multiplication as pairing.
///
/// For example, the equation `n * W + (v * U)^5 = t_1` can be expressed by the following
/// (private) witness variables `X = [U, W]`, `Y = [v]`, (public) constants `A = [0]`, `B = [0, n]`,
/// pairing exponent matrix `gamma = [[5], [0]]`, and `target = t_1` in `G1`.
pub struct MSMEG1<E: PairingEngine> {
    a_consts: Vec<E::G1Affine>,
    b_consts: Vec<E::Fr>,
    gamma: Matrix<E::Fr>,
    target: E::G1Affine
}
impl<E: PairingEngine> Equ for MSMEG1<E> {}
impl<E: PairingEngine> Equation<E, E::G1Affine, E::Fr, E::G1Affine> for MSMEG1<E> {}

/// A multi-scalar multiplication equation in [`G2`](ark_ec::PairingEngine::G2Affine), equipped with point-scalar multiplication as pairing.
///
/// For example, the equation `w * N + (u * V)^5 = t_2` can be expressed by the following
/// (private) witness variables `X = [u, w]`, `Y = [V]`, (public) constants `A = [0]`, `B = [0, N]`,
/// pairing exponent matrix `gamma = [[5], [0]]`, and `target = t_2` in `G2`.
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
/// pairing exponent matrix `gamma = [[5], [0]]`, and `target = t_p` in `Fr`.
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
    use ark_ff::{Zero, One};
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
