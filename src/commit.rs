
#![allow(non_snake_case)]

use ark_ec::{PairingEngine, AffineCurve, ProjectiveCurve};
use ark_std::{
    UniformRand,
    rand::{CryptoRng, Rng}
};

use crate::data_structures::*;
use crate::generator::CRS;

// TODO: Perform individual commitments as well

/// Commit all G1 elements in list to corresponding element in B1.
pub fn batch_commit_G1<CR, E>(xvars: &Vec<E::G1Affine>, key: &CRS<E>, rng: &mut CR) -> Vec<Com1<E>> 
where
    E: PairingEngine,
    CR: Rng + CryptoRng
{
    
    // R is a random scalar m x 2 matrix
    let m = xvars.len();
    let mut R: Matrix<E::Fr> = Vec::with_capacity(m);
    for _ in 0..m {
        R.push(vec![E::Fr::rand(rng); 2]);
    }

    // i_1(X) = [ (O, X_1), ..., (O, X_m) ] (m x 1 matrix)
    let lin_x: Matrix<Com1<E>> = vec_to_col_vec(&Com1::<E>::batch_linear_map(xvars));

    // c := i_1(X) + Ru (m x 1 matrix)
    let coms = lin_x.add(&key.u.left_mul(&R, false));

    col_vec_to_vec(&coms)
}

/// Commit all scalar field elements in list to corresponding element in B1.
pub fn batch_commit_scalar_to_B1<CR, E>(scalar_xvars: &Vec<E::Fr>, key: &CRS<E>, rng: &mut CR) -> Vec<Com1<E>>
where
    E: PairingEngine,
    CR: Rng + CryptoRng
{
    let mprime = scalar_xvars.len();
    let mut r: Matrix<E::Fr> = Vec::with_capacity(mprime);
    for _ in 0..mprime {
        r.push(vec![E::Fr::rand(rng)]);
    }

    let slin_x: Matrix<Com1<E>> = vec_to_col_vec(&Com1::<E>::batch_scalar_linear_map(scalar_xvars, key));
    let ru: Matrix<Com1<E>> = vec_to_col_vec(
        &col_vec_to_vec(&r).into_iter().map( |sca| {
            key.u[0][0].scalar_mul(&sca)
        }).collect::<Vec<Com1<E>>>()
    );

    // c := i_1'(x) + r u_1 (mprime x 1 matrix)
    let coms: Matrix<Com1<E>> = slin_x.add(&ru);

    col_vec_to_vec(&coms)
}

/// Commit all G2 elements in list to corresponding element in B2.
pub fn batch_commit_G2<CR, E>(yvars: &Vec<E::G2Affine>, key: &CRS<E>, rng: &mut CR) -> Vec<Com2<E>> 
where
    E: PairingEngine,
    CR: Rng + CryptoRng
{

    // S is a random scalar n x 2 matrix
    let n = yvars.len();
    let mut S: Matrix<E::Fr> = Vec::with_capacity(n);
    for _ in 0..n {
        S.push(vec![E::Fr::rand(rng); 2]);
    }

    // i_2(Y) = [ (O, Y_1), ..., (O, Y_m) ] (n x 1 matrix)
    let lin_y: Matrix<Com2<E>> = vec_to_col_vec(&Com2::<E>::batch_linear_map(yvars));

    // c := i_2(Y) + Sv (n x 1 matrix)
    let coms = lin_y.add(&key.v.left_mul(&S, false));

    col_vec_to_vec(&coms)
}

/// Commit all scalar field elements in list to corresponding element in B2.
pub fn batch_commit_scalar_to_B2<CR, E>(scalar_yvars: &Vec<E::Fr>, key: &CRS<E>, rng: &mut CR) -> Vec<Com2<E>>
where
    E: PairingEngine,
    CR: Rng + CryptoRng
{
    let nprime = scalar_yvars.len();
    let mut s: Matrix<E::Fr> = Vec::with_capacity(nprime);
    for _ in 0..nprime {
        s.push(vec![E::Fr::rand(rng)]);
    }

    let slin_y: Matrix<Com2<E>> = vec_to_col_vec(&Com2::<E>::batch_scalar_linear_map(scalar_yvars, key));
    let sv: Matrix<Com2<E>> = vec_to_col_vec(
        &col_vec_to_vec(&s).into_iter().map( |sca| {
            key.v[0][0].scalar_mul(&sca)
        }).collect::<Vec<Com2<E>>>()
    );

    // d := i_2'(y) + s v_1 (nprime x 1 matrix)
    let coms: Matrix<Com2<E>> = slin_y.add(&sv);

    col_vec_to_vec(&coms)
}
