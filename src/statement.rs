//! Contains the data structures that define Groth-Sahai statements.
//!
//! A Groth-Sahai statement is a list of [`Equations`](self::Equation) with the abstract form `(A * Y)(X * B)(X * Γ Y) = t`, where:
//!
//! - `A` and `B` are vectors representing public constants in the equation,
//! - `X` and `Y` are vectors representing private variables in the equation (introduced on prove),
//! - `Γ` is a matrix of public [scalar](ark_ec::Pairing::Fr) constants defining how to scalar multiply
//!     the corresponding variables being paired together,
//! - `t` is a public constant representing the RHS of the equation, and
//! - `*` is the specified pairing, applied entry-wise to the corresponding elements in each vector.
//!
//! Each [`Equation`](self::Equation) contains the public components of the equation to be proven
//! and must be one of the following four types, each defined over a bilinear group:
//!
//! 1) **Pairing-product equation** ([`PPE`](self::PPE)):&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `(G1, G2, GT)` with
//!     [`e`](ark_ec::Pairing::pairing)` : G1 x G2 -> GT` as the equipped pairing.
//! 2) **Multi-scalar mult. equation in G1** ([`MSMEG1`](self::MSMEG1)):&emsp;`(G1, Fr, G1)`
//!     with [point-scalar multiplication](ark_ec::AffineCurve::mul) as the equipped pairing.
//! 3) **Multi-scalar mult. equation in G2** ([`MSMEG2`](self::MSMEG2)):&emsp;`(Fr, G2, G2)`
//!     with [point-scalar multiplication](ark_ec::AffineCurve::mul) as the equipped pairing.
//! 4) **Quadratic equation** ([`QuadEqu`](self::QuadEqu)):&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;`(Fr, Fr, Fr)`
//!     with [scalar](ark_ec::Pairing::Fr) multiplication as the equipped pairing.
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

use ark_ec::pairing::{Pairing, PairingOutput};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};

use crate::data_structures::Matrix;
use crate::prover::Provable;
use crate::verifier::Verifiable;

/// Groth-Sahai statement (i.e. bilinear equation) types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EquType {
    // The order of the variants must be preserved for serialization.
    PairingProduct,
    MultiScalarG1,
    MultiScalarG2,
    Quadratic,
}

// Implement the `Valid` trait required for implementing `CanonicalDeserialize`.
impl Valid for EquType {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        Ok(())
    }
}

// Although `CanonicalSerialize` and `CanonicalDeerialize` are typically used for elements
// in arkworks ecosystem, here we implement it for the `EquType` enum so that the it can
// be used as a field in structs that support canonical (de)serialization.
impl CanonicalSerialize for EquType {
    #[inline]
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        let b = match self {
            EquType::PairingProduct => 0u8,
            EquType::MultiScalarG1 => 1,
            EquType::MultiScalarG2 => 2,
            EquType::Quadratic => 3,
        };
        u8::serialize_compressed(&b, writer)
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        1 // 1 byte
    }
}

impl CanonicalDeserialize for EquType {
    #[inline]
    fn deserialize_with_mode<R: ark_serialize::Read>(
        reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        match u8::deserialize_compressed(reader)? {
            0 => Ok(EquType::PairingProduct),
            1 => Ok(EquType::MultiScalarG1),
            2 => Ok(EquType::MultiScalarG2),
            3 => Ok(EquType::Quadratic),
            _ => Err(ark_serialize::SerializationError::InvalidData),
        }
    }
}

/// A marker trait for an arbitrary Groth-Sahai [`Equation`](self::Equation).
pub trait Equ {}

/// A single equation, defined over an arbitrary bilinear group `(A1, A2, AT)`, that forms
/// the atomic unit for a Groth-Sahai [`Statement`](self::Statement).
pub trait Equation<E: Pairing, A1, A2, AT>: Equ + Provable<E, A1, A2, AT> + Verifiable<E> {
    fn get_type(&self) -> EquType;
}

/// A collection of Groth-Sahai compatible bilinear [`Equations`](self::Equation).
pub type Statement = Vec<dyn Equ>;

/// A pairing-product equation, equipped with the bilinear group pairing
/// [`e`](ark_ec::Pairing::pairing)` : G1 x G2 -> GT`.
///
/// For example, the equation `e(W, N) * e(U, V)^5 = t_T` can be expressed by the following
/// (private) witness variables `X = [U, W]`, `Y = [V]`, (public) constants `A = [0]`, `B = [0, N]`,
/// pairing exponent matrix `Γ = [[5], [0]]`, and `target = t_T` in `GT`.
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct PPE<E: Pairing> {
    pub a_consts: Vec<E::G1Affine>,
    pub b_consts: Vec<E::G2Affine>,
    pub gamma: Matrix<E::ScalarField>,
    pub target: PairingOutput<E>,
}

impl<E: Pairing> Equ for PPE<E> {}
impl<E: Pairing> Equation<E, E::G1Affine, E::G2Affine, PairingOutput<E>> for PPE<E> {
    #[inline(always)]
    fn get_type(&self) -> EquType {
        EquType::PairingProduct
    }
}

/// A multi-scalar multiplication equation in [`G1`](ark_ec::Pairing::G1Affine), equipped with point-scalar multiplication as pairing.
///
/// For example, the equation `n * W + (v * U)^5 = t_1` can be expressed by the following
/// (private) witness variables `X = [U, W]`, `Y = [v]`, (public) constants `A = [0]`, `B = [0, n]`,
/// pairing exponent matrix `Γ = [[5], [0]]`, and `target = t_1` in `G1`.
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct MSMEG1<E: Pairing> {
    pub a_consts: Vec<E::G1Affine>,
    pub b_consts: Vec<E::ScalarField>,
    pub gamma: Matrix<E::ScalarField>,
    pub target: E::G1Affine,
}

impl<E: Pairing> Equ for MSMEG1<E> {}
impl<E: Pairing> Equation<E, E::G1Affine, E::ScalarField, E::G1Affine> for MSMEG1<E> {
    #[inline(always)]
    fn get_type(&self) -> EquType {
        EquType::MultiScalarG1
    }
}

/// A multi-scalar multiplication equation in [`G2`](ark_ec::Pairing::G2Affine), equipped with point-scalar multiplication as pairing.
///
/// For example, the equation `w * N + (u * V)^5 = t_2` can be expressed by the following
/// (private) witness variables `X = [u, w]`, `Y = [V]`, (public) constants `A = [0]`, `B = [0, N]`,
/// pairing exponent matrix `Γ = [[5], [0]]`, and `target = t_2` in `G2`.
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct MSMEG2<E: Pairing> {
    pub a_consts: Vec<E::ScalarField>,
    pub b_consts: Vec<E::G2Affine>,
    pub gamma: Matrix<E::ScalarField>,
    pub target: E::G2Affine,
}
impl<E: Pairing> Equ for MSMEG2<E> {}
impl<E: Pairing> Equation<E, E::ScalarField, E::G2Affine, E::G2Affine> for MSMEG2<E> {
    #[inline(always)]
    fn get_type(&self) -> EquType {
        EquType::MultiScalarG2
    }
}

/// A quadratic equation in the [scalar field](ark_ec::Pairing::Fr), equipped with field multiplication as pairing.
///
/// For example, the equation `w * n + (u * v)^5 = t_p` can be expressed by the following
/// (private) witness variables `X = [u, w]`, `Y = [v]`, (public) constants `A = [0]`, `B = [0, n]`,
/// pairing exponent matrix `Γ = [[5], [0]]`, and `target = t_p` in `Fr`.
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct QuadEqu<E: Pairing> {
    pub a_consts: Vec<E::ScalarField>,
    pub b_consts: Vec<E::ScalarField>,
    pub gamma: Matrix<E::ScalarField>,
    pub target: E::ScalarField,
}
impl<E: Pairing> Equ for QuadEqu<E> {}
impl<E: Pairing> Equation<E, E::ScalarField, E::ScalarField, E::ScalarField> for QuadEqu<E> {
    #[inline(always)]
    fn get_type(&self) -> EquType {
        EquType::Quadratic
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use ark_bls12_381::Bls12_381 as F;
    use ark_ec::CurveGroup;
    use ark_ff::UniformRand;
    use ark_std::ops::Mul;
    use ark_std::test_rng;

    use super::*;
    use crate::generator::*;

    type Fr = <F as Pairing>::ScalarField;
    type GT = PairingOutput<F>;

    #[test]
    fn test_equtypes_serde() {
        for equ_type in [
            EquType::PairingProduct,
            EquType::MultiScalarG1,
            EquType::MultiScalarG2,
            EquType::Quadratic,
        ] {
            assert_eq!(equ_type.serialized_size(ark_serialize::Compress::Yes), 1);
            assert_eq!(equ_type.serialized_size(ark_serialize::Compress::No), 1);

            let mut c_bytes = Vec::new();
            equ_type.serialize_compressed(&mut c_bytes).unwrap();
            assert_eq!(c_bytes.len(), 1);
            let equ_type_de = EquType::deserialize_compressed(&c_bytes[..]).unwrap();
            assert_eq!(equ_type, equ_type_de);

            let mut u_bytes = Vec::new();
            equ_type.serialize_uncompressed(&mut u_bytes).unwrap();
            assert_eq!(u_bytes.len(), 1);
            let equ_type_de = EquType::deserialize_uncompressed(&u_bytes[..]).unwrap();
            assert_eq!(equ_type, equ_type_de);
        }
    }

    #[test]
    fn test_PPE_equation_type() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let equ: PPE<F> = PPE::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()],
            gamma: vec![vec![Fr::rand(&mut rng)]],
            target: GT::rand(&mut rng),
        };

        assert_eq!(equ.get_type(), EquType::PairingProduct);
    }

    #[test]
    fn test_PPE_equation_serde() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let equ: PPE<F> = PPE::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()],
            gamma: vec![vec![Fr::rand(&mut rng)]],
            target: GT::rand(&mut rng),
        };

        // Serialize and deserialize the equation.

        let mut c_bytes = Vec::new();
        equ.serialize_compressed(&mut c_bytes).unwrap();
        let equ_de = PPE::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(equ, equ_de);

        let mut u_bytes = Vec::new();
        equ.serialize_uncompressed(&mut u_bytes).unwrap();
        let equ_de = PPE::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(equ, equ_de);
    }

    #[test]
    fn test_MSMEG1_equation_type() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let equ: MSMEG1<F> = MSMEG1::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::rand(&mut rng)]],
            target: crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
        };

        assert_eq!(equ.get_type(), EquType::MultiScalarG1);
    }

    #[test]
    fn test_MSMEG1_equation_serde() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let equ: MSMEG1<F> = MSMEG1::<F> {
            a_consts: vec![crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine()],
            b_consts: vec![Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::rand(&mut rng)]],
            target: crs.g1_gen.mul(Fr::rand(&mut rng)).into_affine(),
        };

        // Serialize and deserialize the equation.

        let mut c_bytes = Vec::new();
        equ.serialize_compressed(&mut c_bytes).unwrap();
        let equ_de = MSMEG1::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(equ, equ_de);

        let mut u_bytes = Vec::new();
        equ.serialize_uncompressed(&mut u_bytes).unwrap();
        let equ_de = MSMEG1::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(equ, equ_de);
    }

    #[test]
    fn test_MSMEG2_equation_type() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let equ: MSMEG2<F> = MSMEG2::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()],
            gamma: vec![vec![Fr::rand(&mut rng)]],
            target: crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
        };

        assert_eq!(equ.get_type(), EquType::MultiScalarG2);
    }

    #[test]
    fn test_MSMEG2_equation_serde() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);

        let equ: MSMEG2<F> = MSMEG2::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine()],
            gamma: vec![vec![Fr::rand(&mut rng)]],
            target: crs.g2_gen.mul(Fr::rand(&mut rng)).into_affine(),
        };

        // Serialize and deserialize the equation.

        let mut c_bytes = Vec::new();
        equ.serialize_compressed(&mut c_bytes).unwrap();
        let equ_de = MSMEG2::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(equ, equ_de);

        let mut u_bytes = Vec::new();
        equ.serialize_uncompressed(&mut u_bytes).unwrap();
        let equ_de = MSMEG2::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(equ, equ_de);
    }

    #[test]
    fn test_quadratic_equation_type() {
        let mut rng = test_rng();

        let equ: QuadEqu<F> = QuadEqu::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::rand(&mut rng)]],
            target: Fr::rand(&mut rng),
        };

        assert_eq!(equ.get_type(), EquType::Quadratic);
    }

    #[test]
    fn test_quadratic_equation_serde() {
        let mut rng = test_rng();

        let equ: QuadEqu<F> = QuadEqu::<F> {
            a_consts: vec![Fr::rand(&mut rng)],
            b_consts: vec![Fr::rand(&mut rng)],
            gamma: vec![vec![Fr::rand(&mut rng)]],
            target: Fr::rand(&mut rng),
        };

        // Serialize and deserialize the equation.

        let mut c_bytes = Vec::new();
        equ.serialize_compressed(&mut c_bytes).unwrap();
        let equ_de = QuadEqu::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(equ, equ_de);

        let mut u_bytes = Vec::new();
        equ.serialize_uncompressed(&mut u_bytes).unwrap();
        let equ_de = QuadEqu::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(equ, equ_de);
    }
}
