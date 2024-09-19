//! Commit from scalar field [`Fr`](ark_ec::Pairing::Fr) or bilinear group `G1, G2`
//! into the Groth-Sahai commitment group `B1, B2` for the SXDH instantiation.
#![allow(non_snake_case)]

use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{fmt::Debug, rand::Rng, UniformRand};

use crate::data_structures::{col_vec_to_vec, vec_to_col_vec, Com1, Com2, Mat, Matrix, B1, B2};
use crate::generator::CRS;

pub trait Commit: Eq + Debug {
    /// Append together two lists of commits to obtain single list of commits.
    fn append(&mut self, other: &mut Self);
}

/// Contains both the commitment's values (as [`Com1`](crate::data_structures::Com1)) and its randomness.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Commit1<E: Pairing> {
    pub coms: Vec<Com1<E>>,
    pub(super) rand: Matrix<E::ScalarField>,
}
/// Contains both the commitment's values (as [`Com2`](crate::data_structures::Com2)) and its randomness.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Commit2<E: Pairing> {
    pub coms: Vec<Com2<E>>,
    pub(super) rand: Matrix<E::ScalarField>,
}

macro_rules! impl_com {
    ($( $commit:ident ),*) => {
        $(
            impl<E: Pairing> PartialEq for $commit<E> {

                #[inline]
                fn eq(&self, other: &Self) -> bool {
                    self.coms == other.coms && self.rand == other.rand
                }
            }
            impl<E: Pairing> Eq for $commit<E> {}

            impl<E: Pairing> Commit for $commit<E> {
                fn append(&mut self, other: &mut Self) {
                    // One row of random values per committed value
                    assert_eq!(self.coms.len(), self.rand.len());
                    assert_eq!(other.coms.len(), other.rand.len());
                    let mut otherComs: Vec<_> = other.coms.drain(..).collect();
                    let mut otherRand: Vec<_> = other.rand.drain(..).collect();
                    self.coms.append(&mut otherComs);
                    self.rand.append(&mut otherRand);
                }
            }
        )*
    }
}
impl_com!(Commit1, Commit2);

/// Commit a single [`G1`](ark_ec::Pairing::G1Affine) element to [`B1`](crate::data_structures::Com1).
pub fn commit_G1<CR, E>(xvar: &E::G1Affine, key: &CRS<E>, rng: &mut CR) -> Commit1<E>
where
    E: Pairing,
    CR: Rng,
{
    let (r1, r2) = (E::ScalarField::rand(rng), E::ScalarField::rand(rng));

    // c := i_1(x) + r_1 u_1 + r_2 u_2
    Commit1::<E> {
        coms: vec![
            Com1::<E>::linear_map(xvar)
                + vec_to_col_vec(&key.u)[0][0].scalar_mul(&r1)
                + vec_to_col_vec(&key.u)[1][0].scalar_mul(&r2),
        ],
        rand: vec![vec![r1, r2]],
    }
}

/// Commit all [`G1`](ark_ec::Pairing::G1Affine) elements in list to corresponding element in [`B1`](crate::data_structures::Com1).
pub fn batch_commit_G1<CR, E>(xvars: &[E::G1Affine], key: &CRS<E>, rng: &mut CR) -> Commit1<E>
where
    E: Pairing,
    CR: Rng,
{
    // R is a random scalar m x 2 matrix
    let m = xvars.len();
    let mut R: Matrix<E::ScalarField> = Vec::with_capacity(m);
    for _ in 0..m {
        R.push(vec![E::ScalarField::rand(rng), E::ScalarField::rand(rng)]);
    }

    // i_1(X) = [ (O, X_1), ..., (O, X_m) ] (m x 1 matrix)
    let lin_x: Matrix<Com1<E>> = vec_to_col_vec(&Com1::<E>::batch_linear_map(xvars));

    // c := i_1(X) + Ru (m x 1 matrix)
    let coms = lin_x.add(&vec_to_col_vec(&key.u).left_mul(&R, false));

    Commit1::<E> {
        coms: col_vec_to_vec(&coms),
        rand: R,
    }
}

/// Commit a single [scalar field](ark_ec::Pairing::Fr) element to [`B1`](crate::data_structures::Com1).
pub fn commit_scalar_to_B1<CR, E>(
    scalar_xvar: &E::ScalarField,
    key: &CRS<E>,
    rng: &mut CR,
) -> Commit1<E>
where
    E: Pairing,
    CR: Rng,
{
    let r: E::ScalarField = E::ScalarField::rand(rng);

    // c := i_1'(x) + r u_1
    Commit1::<E> {
        coms: vec![
            Com1::<E>::scalar_linear_map(scalar_xvar, key)
                + vec_to_col_vec(&key.u)[0][0].scalar_mul(&r),
        ],
        rand: vec![vec![r]],
    }
}

/// Commit all [scalar field](ark_ec::Pairing::Fr) elements in list to corresponding element in [`B1`](crate::data_structures::Com1).
pub fn batch_commit_scalar_to_B1<CR, E>(
    scalar_xvars: &[E::ScalarField],
    key: &CRS<E>,
    rng: &mut CR,
) -> Commit1<E>
where
    E: Pairing,
    CR: Rng,
{
    let mprime = scalar_xvars.len();
    let mut r: Matrix<E::ScalarField> = Vec::with_capacity(mprime);
    for _ in 0..mprime {
        r.push(vec![E::ScalarField::rand(rng)]);
    }

    let slin_x: Matrix<Com1<E>> =
        vec_to_col_vec(&Com1::<E>::batch_scalar_linear_map(scalar_xvars, key));
    let ru: Matrix<Com1<E>> = vec_to_col_vec(
        &col_vec_to_vec(&r)
            .into_iter()
            .map(|sca| vec_to_col_vec(&key.u)[0][0].scalar_mul(&sca))
            .collect::<Vec<Com1<E>>>(),
    );

    // c := i_1'(x) + r u_1 (mprime x 1 matrix)
    let coms: Matrix<Com1<E>> = slin_x.add(&ru);

    Commit1::<E> {
        coms: col_vec_to_vec(&coms),
        rand: r,
    }
}

/// Commit a single [`G2`](ark_ec::Pairing::G2Affine) element to [`B2`](crate::data_structures::Com2).
pub fn commit_G2<CR, E>(yvar: &E::G2Affine, key: &CRS<E>, rng: &mut CR) -> Commit2<E>
where
    E: Pairing,
    CR: Rng,
{
    let (s1, s2) = (E::ScalarField::rand(rng), E::ScalarField::rand(rng));

    // d := i_2(y) + s_1 v_1 + s_2 v_2
    Commit2::<E> {
        coms: vec![
            Com2::<E>::linear_map(yvar)
                + vec_to_col_vec(&key.v)[0][0].scalar_mul(&s1)
                + vec_to_col_vec(&key.v)[1][0].scalar_mul(&s2),
        ],
        rand: vec![vec![s1, s2]],
    }
}

/// Commit all [`G2`](ark_ec::Pairing::G2Affine) elements in list to corresponding element in [`B2`](crate::data_structures::Com2).
pub fn batch_commit_G2<CR, E>(yvars: &[E::G2Affine], key: &CRS<E>, rng: &mut CR) -> Commit2<E>
where
    E: Pairing,
    CR: Rng,
{
    // S is a random scalar n x 2 matrix
    let n = yvars.len();
    let mut S: Matrix<E::ScalarField> = Vec::with_capacity(n);
    for _ in 0..n {
        S.push(vec![E::ScalarField::rand(rng), E::ScalarField::rand(rng)]);
    }

    // i_2(Y) = [ (O, Y_1), ..., (O, Y_m) ] (n x 1 matrix)
    let lin_y: Matrix<Com2<E>> = vec_to_col_vec(&Com2::<E>::batch_linear_map(yvars));

    // c := i_2(Y) + Sv (n x 1 matrix)
    let coms = lin_y.add(&vec_to_col_vec(&key.v).left_mul(&S, false));

    Commit2::<E> {
        coms: col_vec_to_vec(&coms),
        rand: S,
    }
}

/// Commit a single [scalar field](ark_ec::Pairing::Fr) element to [`B2`](crate::data_structures::Com2).
pub fn commit_scalar_to_B2<CR, E>(
    scalar_yvar: &E::ScalarField,
    key: &CRS<E>,
    rng: &mut CR,
) -> Commit2<E>
where
    E: Pairing,
    CR: Rng,
{
    let s: E::ScalarField = E::ScalarField::rand(rng);

    // d := i_2'(y) + s v_1
    Commit2::<E> {
        coms: vec![
            Com2::<E>::scalar_linear_map(scalar_yvar, key)
                + vec_to_col_vec(&key.v)[0][0].scalar_mul(&s),
        ],
        rand: vec![vec![s]],
    }
}

/// Commit all [scalar field](ark_ec::Pairing::Fr) elements in list to corresponding element in [`B2`](crate::data_structures::Com2).
pub fn batch_commit_scalar_to_B2<CR, E>(
    scalar_yvars: &[E::ScalarField],
    key: &CRS<E>,
    rng: &mut CR,
) -> Commit2<E>
where
    E: Pairing,
    CR: Rng,
{
    let nprime = scalar_yvars.len();
    let mut s: Matrix<E::ScalarField> = Vec::with_capacity(nprime);
    for _ in 0..nprime {
        s.push(vec![E::ScalarField::rand(rng)]);
    }

    let slin_y: Matrix<Com2<E>> =
        vec_to_col_vec(&Com2::<E>::batch_scalar_linear_map(scalar_yvars, key));
    let sv: Matrix<Com2<E>> = vec_to_col_vec(
        &col_vec_to_vec(&s)
            .into_iter()
            .map(|sca| vec_to_col_vec(&key.v)[0][0].scalar_mul(&sca))
            .collect::<Vec<Com2<E>>>(),
    );

    // d := i_2'(y) + s v_1 (nprime x 1 matrix)
    let coms: Matrix<Com2<E>> = slin_y.add(&sv);

    Commit2::<E> {
        coms: col_vec_to_vec(&coms),
        rand: s,
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use std::ops::Mul;
    use std::str::FromStr;

    use ark_bls12_381::Bls12_381 as F;
    use ark_ec::CurveGroup;
    use ark_ff::One;
    use ark_std::test_rng;

    use crate::AbstractCrs;

    use super::*;

    type G1Affine = <F as Pairing>::G1Affine;
    type G2Affine = <F as Pairing>::G2Affine;
    type Fr = <F as Pairing>::ScalarField;

    // Uses an affine group generator to produce an affine group element represented by the numeric string.
    macro_rules! affine_group_new {
        ($gen:expr, $strnum:tt) => {
            $gen.mul(Fr::from_str($strnum).unwrap()).into_affine()
        };
    }

    // Uses an affine group generator to produce a projective group element represented by the numeric string.
    #[allow(unused_macros)]
    macro_rules! projective_group_new {
        ($gen:expr, $strnum:tt) => {
            $gen.mul(Fr::from_str($strnum).unwrap())
        };
    }

    #[test]
    fn test_commit_serde() {
        let mut rng = test_rng();
        let crs = CRS::<F>::generate_crs(&mut rng);
        let r1 = Fr::rand(&mut rng);
        let r2 = Fr::rand(&mut rng);
        let com1 = Commit1::<F> {
            coms: vec![Com1::<F>(
                crs.g1_gen.mul(r1).into_affine(),
                crs.g1_gen.mul(r2).into_affine(),
            )],
            rand: vec![vec![r1, r2]],
        };
        let com2 = Commit2::<F> {
            coms: vec![Com2::<F>(
                crs.g2_gen.mul(r1).into_affine(),
                crs.g2_gen.mul(r2).into_affine(),
            )],
            rand: vec![vec![r1, r2]],
        };

        // Serialize and deserialize the commitment 1
        let mut c_bytes = Vec::new();
        com1.serialize_compressed(&mut c_bytes).unwrap();
        let com1_de = Commit1::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(com1, com1_de);

        let mut u_bytes = Vec::new();
        com1.serialize_uncompressed(&mut u_bytes).unwrap();
        let com1_de = Commit1::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(com1, com1_de);

        // Serialize and deserialize the commitment 2
        let mut c_bytes = Vec::new();
        com2.serialize_compressed(&mut c_bytes).unwrap();
        let com2_de = Commit2::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
        assert_eq!(com2, com2_de);

        let mut u_bytes = Vec::new();
        com2.serialize_uncompressed(&mut u_bytes).unwrap();
        let com2_de = Commit2::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
        assert_eq!(com2, com2_de);
    }

    #[test]
    fn test_commit_append_com1() {
        std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
        let mut rng = test_rng();

        let crs = CRS::<F>::generate_crs(&mut rng);
        let r11 = Fr::rand(&mut rng);
        let r12 = Fr::rand(&mut rng);
        let r21 = Fr::rand(&mut rng);
        let r22 = Fr::rand(&mut rng);

        // Create a fake commit value
        let mut com1 = Commit1::<F> {
            coms: vec![Com1::<F>(
                crs.g1_gen.mul(r11).into_affine(),
                crs.g1_gen.mul(r12).into_affine(),
            )],
            rand: vec![vec![r11, r12]],
        };
        let mut com2 = Commit1::<F> {
            coms: vec![Com1::<F>(
                crs.g1_gen.mul(r21).into_affine(),
                crs.g1_gen.mul(r22).into_affine(),
            )],
            rand: vec![vec![r21, r22]],
        };

        // Append should append each of the internal vectors
        let com1_exp = Commit1::<F> {
            coms: vec![
                Com1::<F>(
                    crs.g1_gen.mul(r11).into_affine(),
                    crs.g1_gen.mul(r12).into_affine(),
                ),
                Com1::<F>(
                    crs.g1_gen.mul(r21).into_affine(),
                    crs.g1_gen.mul(r22).into_affine(),
                ),
            ],
            rand: vec![vec![r11, r12], vec![r21, r22]],
        };
        let com2_exp = Commit1::<F> {
            coms: vec![],
            rand: vec![],
        };

        com1.append(&mut com2);
        assert_eq!(com1, com1_exp);
        assert_eq!(com2, com2_exp);
    }

    #[test]
    fn test_commit_append_com2() {
        std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
        let mut rng = test_rng();

        let crs = CRS::<F>::generate_crs(&mut rng);
        let r11 = Fr::rand(&mut rng);
        let r12 = Fr::rand(&mut rng);
        let r21 = Fr::rand(&mut rng);
        let r22 = Fr::rand(&mut rng);

        // Create a fake commit value
        let mut com1 = Commit2::<F> {
            coms: vec![Com2::<F>(
                crs.g2_gen.mul(r11).into_affine(),
                crs.g2_gen.mul(r12).into_affine(),
            )],
            rand: vec![vec![r11, r12]],
        };
        let mut com2 = Commit2::<F> {
            coms: vec![Com2::<F>(
                crs.g2_gen.mul(r21).into_affine(),
                crs.g2_gen.mul(r22).into_affine(),
            )],
            rand: vec![vec![r21, r22]],
        };

        // Append should append each of the internal vectors
        let com1_exp = Commit2::<F> {
            coms: vec![
                Com2::<F>(
                    crs.g2_gen.mul(r11).into_affine(),
                    crs.g2_gen.mul(r12).into_affine(),
                ),
                Com2::<F>(
                    crs.g2_gen.mul(r21).into_affine(),
                    crs.g2_gen.mul(r22).into_affine(),
                ),
            ],
            rand: vec![vec![r11, r12], vec![r21, r22]],
        };
        let com2_exp = Commit2::<F> {
            coms: vec![],
            rand: vec![],
        };

        com1.append(&mut com2);
        assert_eq!(com1, com1_exp);
        assert_eq!(com2, com2_exp);
    }

    #[test]
    fn test_commit_G1_batching() {
        std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
        let mut rng = test_rng();
        let mut rng2 = test_rng();

        let crs = CRS::<F>::generate_crs(&mut rng);
        let rngsync1 = Fr::rand(&mut rng);

        let xvars: Vec<G1Affine> = vec![
            crs.g1_gen,
            affine_group_new!(crs.g1_gen, "2"),
            affine_group_new!(crs.g1_gen, "3"),
        ];
        let mut exp: Commit1<F> = commit_G1(&xvars[0], &crs, &mut rng);
        exp.append(&mut commit_G1(&xvars[1], &crs, &mut rng));
        exp.append(&mut commit_G1(&xvars[2], &crs, &mut rng));

        // Mock the use of CRS so both RNGs are at the same point
        let _ = CRS::<F>::generate_crs(&mut rng2);
        let rngsync2 = Fr::rand(&mut rng2);
        assert_eq!(rngsync1, rngsync2);

        let res: Commit1<F> = batch_commit_G1(&xvars, &crs, &mut rng2);
        assert_eq!(exp, res);
    }

    #[test]
    fn test_commit_G2_batching() {
        std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
        let mut rng = test_rng();
        let mut rng2 = test_rng();

        let crs = CRS::<F>::generate_crs(&mut rng);
        let rngsync1 = Fr::rand(&mut rng);

        let yvars: Vec<G2Affine> = vec![
            crs.g2_gen,
            affine_group_new!(crs.g2_gen, "2"),
            affine_group_new!(crs.g2_gen, "3"),
        ];
        let mut exp: Commit2<F> = commit_G2(&yvars[0], &crs, &mut rng);
        exp.append(&mut commit_G2(&yvars[1], &crs, &mut rng));
        exp.append(&mut commit_G2(&yvars[2], &crs, &mut rng));

        // Mock the use of CRS so both RNGs are at the same point
        let _ = CRS::<F>::generate_crs(&mut rng2);
        let rngsync2 = Fr::rand(&mut rng2);
        assert_eq!(rngsync1, rngsync2);

        let res: Commit2<F> = batch_commit_G2(&yvars, &crs, &mut rng2);

        assert_eq!(exp, res);
    }

    #[test]
    fn test_commit_scalar_B1_batching() {
        std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
        let mut rng = test_rng();
        let mut rng2 = test_rng();

        let crs = CRS::<F>::generate_crs(&mut rng);
        let rngsync1 = Fr::rand(&mut rng);

        let scalar_xvars: Vec<Fr> = vec![
            Fr::one(),
            Fr::from_str("2").unwrap(),
            Fr::from_str("3").unwrap(),
        ];
        let mut exp: Commit1<F> = commit_scalar_to_B1(&scalar_xvars[0], &crs, &mut rng);
        exp.append(&mut commit_scalar_to_B1(&scalar_xvars[1], &crs, &mut rng));
        exp.append(&mut commit_scalar_to_B1(&scalar_xvars[2], &crs, &mut rng));

        // Mock the use of CRS so both RNGs are at the same point
        let _ = CRS::<F>::generate_crs(&mut rng2);
        let rngsync2 = Fr::rand(&mut rng2);
        assert_eq!(rngsync1, rngsync2);

        let res: Commit1<F> = batch_commit_scalar_to_B1(&scalar_xvars, &crs, &mut rng2);

        assert_eq!(exp, res);
    }

    #[test]
    fn test_commit_scalar_B2_batching() {
        std::env::set_var("DETERMINISTIC_TEST_RNG", "1");
        let mut rng = test_rng();
        let mut rng2 = test_rng();

        let crs = CRS::<F>::generate_crs(&mut rng);
        let rngsync1 = Fr::rand(&mut rng);

        let scalar_yvars: Vec<Fr> = vec![
            Fr::one(),
            Fr::from_str("2").unwrap(),
            Fr::from_str("3").unwrap(),
        ];
        let mut exp: Commit2<F> = commit_scalar_to_B2(&scalar_yvars[0], &crs, &mut rng);
        exp.append(&mut commit_scalar_to_B2(&scalar_yvars[1], &crs, &mut rng));
        exp.append(&mut commit_scalar_to_B2(&scalar_yvars[2], &crs, &mut rng));

        // Mock the use of CRS so both RNGs are at the same point
        let _ = CRS::<F>::generate_crs(&mut rng2);
        let rngsync2 = Fr::rand(&mut rng2);
        assert_eq!(rngsync1, rngsync2);

        let res: Commit2<F> = batch_commit_scalar_to_B2(&scalar_yvars, &crs, &mut rng2);

        assert_eq!(exp, res);
    }
}
