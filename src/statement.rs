//! Contains the data structures that define Groth-Sahai statements.
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
//! 4) **Quadratic equation** ([`QuadEqu`](self::QuadEqu)):&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;`(Fr, Fr, Fr)`
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

use ark_ec::PairingEngine;

use crate::data_structures::Matrix;
use crate::prover::Provable;
use crate::verifier::Verifiable;

/// Groth-Sahai statement (i.e. bilinear equation) types.
#[derive(Debug, PartialEq, Eq)]
pub enum EquType {
    PairingProduct,
    MultiScalarG1,
    MultiScalarG2,
    Quadratic
}

/// A marker trait for an arbitrary Groth-Sahai [`Equation`](self::Equation).
pub trait Equ {}

/// A single equation, defined over an arbitrary bilinear group `(A1, A2, AT)`, that forms
/// the atomic unit for a Groth-Sahai [`Statement`](self::Statement).
pub trait Equation<E: PairingEngine, A1, A2, AT>:
    Equ
    + Provable<E, A1, A2, AT>
    + Verifiable<E>
{
    fn get_type(&self) -> EquType;
}

/// A collection of Groth-Sahai compatible bilinear [`Equations`](self::Equation).
pub type Statement = Vec<dyn Equ>;

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
    fn get_type(&self) -> EquType {
        EquType::PairingProduct
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
    fn get_type(&self) -> EquType {
        EquType::MultiScalarG1
    }
}

/// A multi-scalar multiplication equation in [`G2`](ark_ec::PairingEngine::G2Affine), equipped with point-scalar multiplication as pairing.
///
/// For example, the equation `w * N + (u * V)^5 = t_2` can be expressed by the following
/// (private) witness variables `X = [u, w]`, `Y = [V]`, (public) constants `A = [0]`, `B = [0, N]`,
/// pairing exponent matrix `Γ = [[5], [0]]`, and `target = t_2` in `G2`.
pub struct MSMEG2<E: PairingEngine> {
    pub a_consts: Vec<E::Fr>,
    pub b_consts: Vec<E::G2Affine>,
    pub gamma: Matrix<E::Fr>,
    pub target: E::G2Affine
}
impl<E: PairingEngine> Equ for MSMEG2<E> {}
impl<E: PairingEngine> Equation<E, E::Fr, E::G2Affine, E::G2Affine> for MSMEG2<E> {
    #[inline(always)]
    fn get_type(&self) -> EquType {
        EquType::MultiScalarG2
    }
}


/// A quadratic equation in the [scalar field](ark_ec::PairingEngine::Fr), equipped with field multiplication as pairing.
///
/// For example, the equation `w * n + (u * v)^5 = t_p` can be expressed by the following
/// (private) witness variables `X = [u, w]`, `Y = [v]`, (public) constants `A = [0]`, `B = [0, n]`,
/// pairing exponent matrix `Γ = [[5], [0]]`, and `target = t_p` in `Fr`.
pub struct QuadEqu<E: PairingEngine> {
    pub a_consts: Vec<E::Fr>,
    pub b_consts: Vec<E::Fr>,
    pub gamma: Matrix<E::Fr>,
    pub target: E::Fr
}
impl<E: PairingEngine> Equ for QuadEqu<E> {}
impl<E: PairingEngine> Equation<E, E::Fr, E::Fr, E::Fr> for QuadEqu<E> {
    #[inline(always)]
    fn get_type(&self) -> EquType {
        EquType::Quadratic
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use ark_bls12_381::{Bls12_381 as F};
    use ark_ec::{PairingEngine, AffineCurve, ProjectiveCurve};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use super::*;
    use crate::CRS;

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

        assert_eq!(equ.get_type(), EquType::PairingProduct);
    }

    #[test]
    fn test_MSMEG1_equation_type() {

        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let equ: MSMEG1<F> = MSMEG1::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::rand(&mut rng)]],
            target: crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()
        };

        assert_eq!(equ.get_type(), EquType::MultiScalarG1);
    }

    #[test]
    fn test_MSMEG2_equation_type() {

        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let equ: MSMEG2<F> = MSMEG2::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()],
            gamma: vec![vec![Fr::rand(&mut rng)]],
            target: crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()
        };

        assert_eq!(equ.get_type(), EquType::MultiScalarG2);
    }

    #[test]
    fn test_quadratic_equation_type() {

        let mut rng = test_rng();

        let equ: QuadEqu<F> = QuadEqu::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::rand(&mut rng)]],
            target: Fr::rand(&mut rng)
        };

        assert_eq!(equ.get_type(), EquType::Quadratic);
    }
}
