
// TODO: For some reason inner docs didn't work, using //! or /*! inside data_structures.rs. Move
// it back into there for more clarity...
/// Contains the abstract Groth-Sahai commitment group `(B1, B2, BT)`, its concrete representation within the SXDH instantiation `(Com1, Com2, ComT)`, as well as an implementation of matrix arithmetic. 
///
/// # Properties
///
/// In a Type-III pairing setting, the Groth-Sahai instantiation requires the SXDH assumption,
/// implementing the commitment group using elements of the bilinear group over an elliptic curve.
/// [`Com1`](crate::data_structures::Com1) and [`Com2`](crate::data_structures:::Com2) are represented by 2 x 1 vectors of elements in the corresponding groups [`G1Affine`](ark_ec::PairingEngine::G1Affine) and [`G2Affine`](ark_ec::PairingEngine::G2Affine).
/// [`ComT`](crate::data_structures::ComT) represents a 2 x 2 matrix of elements in [`Fqk`](ark_ec::PairingEngine::Fqk) (aka `GT`).
///
/// All of `Com1`, `Com2`, `ComT` are expressed as follows:
/// * Addition in `Com1`, `Com2` is defined by entry-wise addition of elements in `G1Affine`, `G2Affine`:
///     * The equality of `Com1`, `Com2` is equality of all elements
///     * The zero element of `Com1`, `Com2` is the zero vector
///     * The negation of `Com1`, `Com2` is the additive inverse (i.e. negation) of all elements
///
/// * Addition in `ComT` is defined by entry-wise multiplication of elements in `Fqk`:
///     * The equality of `ComT` is equality of all elements
///     * The zero element of `ComT` is the all-ones vector
///     * The negation of `ComT` is the multiplicative inverse (i.e. reciprocal) of all elements
///
/// The Groth-Sahai proof system uses matrices of commitment group elements in its computations as
/// well. 
pub mod data_structures;
pub mod generator;
//pub mod commit;

/// Groth-Sahai statement (i.e. bilinear equation) types.
pub enum GSType {
    PairingProduct,
    MultiScalarG1,
    MultiScalarG2,
    Quadratic
}

pub enum Role {
    Prover,
    Verifier
}

pub use crate::generator::CRS;
pub use crate::data_structures::*;
//pub use crate::commit::*;
