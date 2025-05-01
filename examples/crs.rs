mod utils;

use ark_bls12_381::Bls12_381;
use groth_sahai::{AbstractCrs, CRS};
use ark_std::test_rng;
use crate::utils::record_value;

pub fn main() {
    let mut rng = test_rng();
    let crs = CRS::<Bls12_381>::generate_crs(&mut rng);
    record_value::<CRS<Bls12_381>>("../examples/testdata/test_crs", &crs);
}
