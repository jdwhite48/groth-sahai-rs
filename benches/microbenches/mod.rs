mod bls12_381;
mod bn254;

pub use bls12_381::*;//bls12_bilinear_group_arith, bls12_commit};
pub use bn254::*;//{bn254_bilinear_group_arith, bn254_commit};

//criterion_main!(bls12_bilinear_group_arith, bls12_commit, bn254_bilinear_group_arith, bn254_commit);
