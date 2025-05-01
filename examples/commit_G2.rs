mod utils;

use ark_ec::{
    CurveGroup,
    pairing::Pairing,
};
use ark_bls12_381::Bls12_381;

use groth_sahai::{
    CRS,
    prover::{Commit2, commit_G2}
};
use ark_std::{test_rng, UniformRand};
use crate::utils::*;

pub fn main() {
        let mut rng = test_rng();
        //let crs = CRS::<Bls12_381>::generate_crs(&mut rng);
        let crs = get_value::<CRS<Bls12_381>>("../examples/testdata/test_crs");
        let a2 = <Bls12_381 as Pairing>::G2::rand(&mut rng).into_affine();
        let com_g2 = commit_G2(&a2, &crs, &mut rng);
        record_value::<Commit2<Bls12_381>>("../examples/testdata/test_commit_G2", &com_g2);
}
