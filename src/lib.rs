pub mod data_structures;
pub mod generator;
pub mod commit;

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
