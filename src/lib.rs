pub mod data_structures;
pub mod generator;

/// Groth-Sahai statement (equation) types
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

pub use generator::CRS;
pub use data_structures::{Com1, Com2, ComT};
