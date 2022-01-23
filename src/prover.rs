//! Contains types and functions for proving about the satisfiability of bilinear equations.
//!
//! An equation has the abstract form `(A * Y)(X * B)(X * Γ Y) = target`
//! where `A` and `B` are vectors representing public constants in the equation,
//! `X` and `Y` are are vectors representing private variables in the equation (see
//! [`Witness`](self::Witness)),
//! `Γ` is a matrix of public constants defining how the pairing applies to the variables, and
//! `target` is a public constant representing the RHS of the equation.
//!
//! A proof consists of 
//!
//! NOTE: The bilinear equation may need to be re-arranged using the properties
//! of bilinear group arithmetic and pairings in order to form a valid Groth-Sahai statement.
//! This API does not provide such functionality.

use ark_ec::{PairingEngine};

use crate::data_structures::*;
//use crate::generator::CRS;
//use crate::commit::*;

/// A single Groth-Sahai statement, expressed as the public components of an arbitrary bilinear equation.
/*
pub struct Equation<E: PairingEngine, A1, A2, AT> {
    a_consts: Vec<A1>,
    b_consts: Vec<A2>,
    gamma: Matrix<E::Fr>,
    target: AT
}
*/
pub trait Equ {}

/// A collection of Groth-Sahai compatible bilinear equations.
pub type Statement = Vec<dyn Equ>;

/// A Groth-Sahai witness, expressed as variables in a corresponding [`Equation`](self::Equ).
pub struct Witness<A1, A2> {
    x_vars: Vec<A1>,
    y_vars: Vec<A2>
}
/// A witness-indistinguishable proof for a single [`Equation`](self::Equ).
pub struct EquProof<E: PairingEngine> {
    pi: Vec<Com2<E>>,
    theta: Vec<Com1<E>>
}

// TODO: Express the combination of proofs at a finer-grained level.
/// A collection of proofs for Groth-Sahai compatible bilinear equations.
pub type Proof<E> = Vec<EquProof<E>>;

/// A pairing product equation, equipped with the bilinear group pairing as pairing.
///
/// For example, the equation `e(W, N) * e(U, V)^5 = t_T` can be expressed by the following
/// (private) witness variables `X = [U, W]`, `Y = [V]`, (public) constants `A = [0]`, `B = [0, N]`,
/// pairing exponent matrix `gamma = [[5], [0]]`, and `target = t_T` in `GT`.
pub struct PPE<E: PairingEngine> {
    a_consts: Vec<E::G1Affine>,
    b_consts: Vec<E::G2Affine>,
    gamma: Matrix<E::Fr>,
    target: E::Fqk
}
impl<E: PairingEngine> Equ for PPE<E> {}

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
