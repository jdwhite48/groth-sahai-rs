mod utils;

use ark_ec::{
    CurveGroup,
    pairing::Pairing,
};
use ark_bls12_381::Bls12_381;

use groth_sahai::{
    CRS,
    prover::{Commit1, commit_G1}
};
use ark_std::{test_rng, UniformRand};
use crate::utils::*;

pub fn main() {
        let mut rng = test_rng();
        //let crs = CRS::<Bls12_381>::generate_crs(&mut rng);
        let crs = get_value::<CRS<Bls12_381>>("../examples/testdata/test_crs");
        let a1 = <Bls12_381 as Pairing>::G1::rand(&mut rng).into_affine();
        let com_g1 = commit_G1(&a1, &crs, &mut rng);
        record_value::<Commit1<Bls12_381>>("../examples/testdata/test_commit_G1", &com_g1);
}
