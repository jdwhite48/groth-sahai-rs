mod utils;

use ark_ec::pairing::Pairing;
use ark_bls12_381::Bls12_381;

use groth_sahai::{
    CRS,
    prover::{Commit2, commit_scalar_to_B2}
};
use ark_std::{test_rng, UniformRand};
use crate::utils::*;

pub fn main() {
        let mut rng = test_rng();
        //let crs = CRS::<Bls12_381>::generate_crs(&mut rng);
        let crs = get_value::<CRS<Bls12_381>>("../examples/testdata/test_crs");
        let s2 = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
        let com_s2 = commit_scalar_to_B2(&s2, &crs, &mut rng);
        record_value::<Commit2<Bls12_381>>("../examples/testdata/test_commit_scalar_to_G2", &com_s2);
}
