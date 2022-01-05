use ark_ec::{PairingEngine, AffineCurve, ProjectiveCurve};
use ark_std::{
    UniformRand,
    rand::{CryptoRng, Rng}
};

use crate::data_structures::*;
use crate::generator::CRS;

#[allow(non_snake_case)]
pub fn commit_G1<R, E>(x: &Vec<E::G1Affine>, key: &CRS<E>, rng: &mut R) -> Vec<Com1<E>> 
where
    E: PairingEngine,
    R: Rng + CryptoRng
{
    
    // R is a random scalar m x 2 matrix
    let m = x.len();
    let mut rand_mat: Matrix<E::Fr> = Vec::with_capacity(m);
    for _ in 0..m { 
        rand_mat.push(vec![E::Fr::rand(&mut rng); 2]);
    }

    // i_1(X) = [ (O, X_1), ..., (O, X_m) ]
    let b1_mat: Matrix<Com1<E>> = Com1::<E>::batch_linear_map(x)
        .into_iter()
        .map( |com1| com1.as_vec() )
        .collect::<Matrix<E::G1Affine>>();
    // Ru = [ r_11 u_1 + r_12 u_2, ..., r_m1 u_1 + r_m2 u_2 ]
    let com_mat: Matrix<Com1<E>> = rand_mat.
        .into_iter()
        // TODO: move matrices to be over B1/B2/BT before continuing....
        .map( |rand_row| group_matrix_scalar_mul::<E, E::G1Affine>(&rand_row[0], &key.u

    // c := = i_1(X) + Ru
    vec![
        matrix_add(&b1_mat, &group_right_matrix_mul::<E, E::G1Affine>(&rand_mat, &key.u.0.as_col_vec(), false)),
        matrix_add(&b1_mat, &group_right_matrix_mul::<E, E::G1Affine>(&rand_mat, &key.u.1.as_col_vec(), false))
    ]
    .into_iter()
    .map( |vec| 
}
