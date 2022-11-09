//! Contains the abstract Groth-Sahai commitment group `(B1, B2, BT)`, its concrete representation
//! within the SXDH instantiation `(Com1, Com2, ComT)`, as well as an implementation of matrix
//! arithmetic.
//!
//! # Properties
//!
//! In a Type-III pairing setting, the Groth-Sahai instantiation requires the SXDH assumption,
//! implementing the commitment group using elements of the bilinear group over an elliptic curve.
//! [`Com1`](crate::data_structures::Com1) and [`Com2`](crate::data_structures::Com2) are represented by 2 x 1 vectors of elements
//! in the corresponding groups [`G1Affine`](ark_ec::PairingEngine::G1Affine) and [`G2Affine`](ark_ec::PairingEngine::G2Affine).
//! [`ComT`](crate::data_structures::ComT) represents a 2 x 2 matrix of elements in [`Fqk`](ark_ec::PairingEngine::Fqk) (aka `GT`).
//!
//! All of `Com1`, `Com2`, `ComT` are expressed as follows:
//! * Addition in `Com1`, `Com2` is defined by entry-wise addition of elements in `G1Affine`, `G2Affine`:
//!     * The equality of `Com1`, `Com2` is equality of all elements
//!     * The zero element of `Com1`, `Com2` is the zero vector
//!     * The negation of `Com1`, `Com2` is the additive inverse (i.e. negation) of all elements
//!
//! * Addition in `ComT` is defined by entry-wise multiplication of elements in `Fqk`:
//!     * The equality of `ComT` is equality of all elements
//!     * The zero element of `ComT` is the all-ones vector
//!     * The negation of `ComT` is the multiplicative inverse (i.e. reciprocal) of all elements
//!
//! The Groth-Sahai proof system uses matrices of commitment group elements in its computations as
//! well.

use ark_ec::{PairingEngine, AffineCurve, ProjectiveCurve};
#[allow(unused_imports)]
use ark_ff::{Zero, One, Field, field_new};
use core::{
    ops::{Add, AddAssign, Neg, Sub, SubAssign, Mul, MulAssign},
    iter::Sum
};
use ark_std::{
    fmt::Debug
};
use rayon::prelude::*;
extern crate nalgebra as na;
use na::{ClosedAdd, ClosedMul};
use na::{Matrix2x1, Matrix2, Vector2};

use crate::generator::*;

// Matrices in Fqk/GT and Fr are automatically amenable with nalgebra because, as a field,
// they're closed under both addition and multiplication. *However*, we also need group matrix
// "multiplication" over (acting on) matrices of the underlying scalar Fr; however, this results
// in a type mismatch e.g. `Fr = Com1` when `impl Mul<E::Fr, Output=Com1<E>> for Com1<E>`
// is necessarily implemented for nalgebra::ClosedMul.
// TODO: Figure out how to specialize such that we can allow for Fr^{m x n} * G1^{n x p} and
// G1^{m x n} * Fr^{n x p} matrix multiplication using nalgebra, or else see if it's even worth it.
/*
pub trait Mat<Elem: Clone>:
    Eq
    + Clone
    + Debug
{
    type Other;

    fn add(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
    // TODO: replace with Mul/ScalarMul
    fn scalar_mul(&self, other: &Self::Other) -> Self;
    fn transpose(&self) -> Self;
    fn left_mul(&self, lhs: &Matrix<Self::Other>, is_parallel: bool) -> Self;
    fn right_mul(&self, rhs: &Matrix<Self::Other>, is_parallel: bool) -> Self;
}

pub type Matrix<E> = Vec<Vec<E>>;
*/

/// Encapsulates arithmetic traits for Groth-Sahai's bilinear group for commitments
pub trait B<E: PairingEngine>:
    Eq
    + Copy
    + Clone
    + Sized
    + Debug
    + Zero
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Neg<Output = Self>
    + Sum
{}

/// Provides linear maps and vector conversions for the base of the GS commitment group.
pub trait B1<E: PairingEngine>:
    B<E>
    + From<Vec<E::G1Affine>>
//    + Into<Matrix2x1<E::G1Affine>>
    + Mul<E::Fr, Output = Self>
    + MulAssign<E::Fr>
    + ClosedAdd
    // NOTE: Not closed under itself, but closed under group action with E::Fr
//    + ClosedMul
{
    // TODO: Into
    fn as_mat(&self) -> Matrix2x1<E::G1Affine>;
    fn as_vector(&self) -> Vector2<E::G1Affine>;
    /// The linear map from G1 to B1 for pairing-product and multi-scalar multiplication equations.
    fn linear_map(x: &E::G1Affine) -> Self;
    fn batch_linear_map(x_vec: &Vec<E::G1Affine>) -> Vec<Self>;
    /// The linear map from scalar field to B1 for multi-scalar multiplication and quadratic equations.
    fn scalar_linear_map(x: &E::Fr, key: &CRS<E>) -> Self;
    fn batch_scalar_linear_map(x_vec: &Vec<E::Fr>, key: &CRS<E>) -> Vec<Self>;
}

/// Provides linear maps and vector conversions for the extension of the GS commitment group.
pub trait B2<E: PairingEngine>:
    B<E>
//    + MulAssign<E::Fr>
    + From<Vec<E::G2Affine>>
//    + Into<Matrix2x1<E::G2Affine>>
    + Mul<E::Fr, Output = Self>
    + MulAssign<E::Fr>
    + ClosedAdd
    // NOTE: Not closed under itself, but closed under group action with E::Fr
//    + ClosedMul
{
    // TODO: Into
    fn as_mat(&self) -> Matrix2x1<E::G2Affine>;
    fn as_vector(&self) -> Vector2<E::G2Affine>;
    /// The linear map from G2 to B2 for pairing-product and multi-scalar multiplication equations.
    fn linear_map(y: &E::G2Affine) -> Self;
    fn batch_linear_map(y_vec: &Vec<E::G2Affine>) -> Vec<Self>;
    /// The linear map from scalar field to B2 for multi-scalar multiplication and quadratic equations.
    fn scalar_linear_map(y: &E::Fr, key: &CRS<E>) -> Self;
    fn batch_scalar_linear_map(y_vec: &Vec<E::Fr>, key: &CRS<E>) -> Vec<Self>;
}

/// Provides linear maps and matrix conversions for the target of the GS commitment group, as well as the equipped pairing.
pub trait BT<E: PairingEngine, C1: B1<E>, C2: B2<E>>:
    B<E>
    + From<Vec<E::Fqk>>
//  + Into<Matrix2<E::Fqk>>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + ClosedAdd
    + ClosedMul
{
    // TODO: Into
    fn as_mat(&self) -> Matrix2<E::Fqk>;

    /// The bilinear pairing over the GS commitment group (B1, B2, BT) is the tensor product.
    /// with respect to the bilinear pairing over the bilinear group (G1, G2, GT).
    fn pairing(x: C1, y: C2) -> Self;
    /// The entry-wise sum of bilinear pairings over the GS commitment group.
    fn pairing_sum(x_vec: &Vec<C1>, y_vec: &Vec<C2>) -> Self;

    /// The linear map from GT to BT for pairing-product equations.
    #[allow(non_snake_case)]
    fn linear_map_PPE(z: &E::Fqk) -> Self;
    /// The linear map from G1 to BT for multi-scalar multiplication equations.
    #[allow(non_snake_case)]
    fn linear_map_MSMEG1(z: &E::G1Affine, key: &CRS<E>) -> Self;
    /// The linear map from G2 to BT for multi-scalar multiplication equations.
    #[allow(non_snake_case)]
    fn linear_map_MSMEG2(z: &E::G2Affine, key: &CRS<E>) -> Self;
    /// The linear map from Fr to BT for quadratic equations.
    fn linear_map_quad(z: &E::Fr, key: &CRS<E>) -> Self;
}

// SXDH instantiation's bilinear group for commitments

/// Base [`B1`](crate::data_structures::B1) for the commitment group in the SXDH instantiation.
#[derive(Copy, Clone, Debug)]
pub struct Com1<E: PairingEngine>(pub E::G1Affine, pub E::G1Affine);

/// Extension [`B2`](crate::data_structures::B2) for the commitment group in the SXDH instantiation.
#[derive(Copy, Clone, Debug)]
pub struct Com2<E: PairingEngine>(pub E::G2Affine, pub E::G2Affine);

/// Target [`BT`](crate::data_structures::BT) for the commitment group in the SXDH instantiation.
#[derive(Copy, Clone, Debug)]
pub struct ComT<E: PairingEngine>(pub E::Fqk, pub E::Fqk, pub E::Fqk, pub E::Fqk);

macro_rules! impl_base_commit_groups {
    (
        $(
            $com:ident
        ),*
    ) => {
        // Repeat for each $com
        $(
            // Equality for Com group
            impl<E: PairingEngine> PartialEq for $com<E> {

                #[inline]
                fn eq(&self, other: &Self) -> bool {
                    self.0 == other.0 && self.1 == other.1
                }
            }
            impl<E: PairingEngine> Eq for $com<E> {}

            // Addition for Com group
            impl<E: PairingEngine> Add<$com<E>> for $com<E> {
                type Output = Self;

                #[inline]
                fn add(self, other: Self) -> Self {
                    Self (
                        self.0 + other.0,
                        self.1 + other.1
                    )
                }
            }
            impl<E: PairingEngine> AddAssign<$com<E>> for $com<E> {

                #[inline]
                fn add_assign(&mut self, other: Self) {
                    *self = Self (
                        self.0 + other.0,
                        self.1 + other.1
                    );
                }
            }
            impl<E: PairingEngine> Neg for $com<E> {
                type Output = Self;

                #[inline]
                fn neg(self) -> Self::Output {
                    Self (
                        -self.0,
                        -self.1
                    )
                }
            }
            impl<E: PairingEngine> Sub<$com<E>> for $com<E> {
                type Output = Self;

                #[inline]
                fn sub(self, other: Self) -> Self {
                    Self (
                        self.0 + -other.0,
                        self.1 + -other.1
                    )
                }
            }
            impl<E: PairingEngine> SubAssign<$com<E>> for $com<E> {

                #[inline]
                fn sub_assign(&mut self, other: Self) {
                    *self = Self (
                        self.0 + -other.0,
                        self.1 + -other.1
                    );
                }
            }
            /*
            // Entry-wise scalar point-multiplication
            impl <E: PairingEngine> MulAssign<E::Fr> for $com<E> {
                fn mul_assign(&mut self, rhs: E::Fr) {

                    let mut s1p = self.0.into_projective();
                    let mut s2p = self.1.into_projective();
                    s1p *= rhs;
                    s2p *= rhs;
                    *self = Self (
                        s1p.into_affine().clone(),
                        s2p.into_affine().clone()
                    )
                }
            }
            */
            impl<E: PairingEngine> Sum for $com<E> {
                fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
                    iter.fold(
                        Self::zero(),
                        |a,b| a + b,
                    )
                }
            }

            // Scalar multiplication
            impl<E: PairingEngine> Mul<E::Fr> for $com<E> {
                type Output = Self;

                #[inline]
                fn mul(self, other: E::Fr) -> Self {
                    let mut s1p = self.0.clone().into_projective();
                    let mut s2p = self.1.clone().into_projective();
                    s1p *= other;
                    s2p *= other;
                    Self (
                        s1p.into_affine().clone(),
                        s2p.into_affine().clone()
                    )
                }
            }

            impl<E: PairingEngine> MulAssign<E::Fr> for $com<E> {
                #[inline]
                fn mul_assign(&mut self, other: E::Fr) {
                    let mut s1p = self.0.clone().into_projective();
                    let mut s2p = self.1.clone().into_projective();
                    s1p *= other;
                    s2p *= other;
                    *self = Self (
                        s1p.into_affine().clone(),
                        s2p.into_affine().clone()
                    );
                }
            }
        )*
    }
}
impl_base_commit_groups!(Com1, Com2);

impl<E: PairingEngine> Zero for Com1<E> {
    #[inline]
    fn zero() -> Self {
        Self (
            E::G1Affine::zero(),
            E::G1Affine::zero()
        )
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}
impl<E: PairingEngine> Zero for Com2<E> {
    #[inline]
    fn zero() -> Self {
        Self (
            E::G2Affine::zero(),
            E::G2Affine::zero()
        )
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<E: PairingEngine> B<E> for Com1<E> {}
impl<E: PairingEngine> B<E> for Com2<E> {}

impl<E: PairingEngine> From<Vec<E::G1Affine>> for Com1<E> {
    fn from(bs: Vec<E::G1Affine>) -> Self {
        assert_eq!(bs.len(), 2);
        Self (
            bs[0],
            bs[1]
        )
    }
}
impl<E: PairingEngine> From<Vec<E::G2Affine>> for Com2<E> {
    fn from(bs: Vec<E::G2Affine>) -> Self {
        assert_eq!(bs.len(), 2);
        Self (
            bs[0],
            bs[1]
        )
    }
}

impl<E: PairingEngine> B1<E> for Com1<E> {
    fn as_mat(&self) -> Matrix2x1<E::G1Affine> {
        Matrix2x1::new(self.0, self.1)
    }

    fn as_vector(&self) -> Vector2<E::G1Affine> {
        Vector2::new(self.0, self.1)
    }

    #[inline]
    fn linear_map(x: &E::G1Affine) -> Self {
        Self (
            E::G1Affine::zero(),
            x.clone()
        )
    }

    #[inline]
    fn batch_linear_map(x_vec: &Vec<E::G1Affine>) -> Vec<Self> {
        x_vec
            .into_iter()
            .map( |elem| Self::linear_map(&elem))
            .collect::<Vec<Self>>()
    }

    #[inline]
    fn scalar_linear_map(x: &E::Fr, key: &CRS<E>) -> Self {
        // = xu, where u = u_2 + (O, P) is a commitment group element
        ( key.u[1] + Com1::<E>::linear_map(&key.g1_gen) ) * *x
    }

    #[inline]
    fn batch_scalar_linear_map(x_vec: &Vec<E::Fr>, key: &CRS<E>) -> Vec<Self> {
        x_vec
            .into_iter()
            .map( |elem| Self::scalar_linear_map(&elem, key))
            .collect::<Vec<Self>>()
    }
}

impl<E: PairingEngine> B2<E> for Com2<E> {

    fn as_mat(&self) -> Matrix2x1<E::G2Affine> {
        Matrix2x1::new(self.0, self.1)
    }

    fn as_vector(&self) -> Vector2<E::G2Affine> {
        Vector2::new(self.0, self.1)
    }

    #[inline]
    fn linear_map(y: &E::G2Affine) -> Self {
        Self (
            E::G2Affine::zero(),
            y.clone()
        )
    }

    #[inline]
    fn batch_linear_map(y_vec: &Vec<E::G2Affine>) -> Vec<Self> {
        y_vec
            .into_iter()
            .map( |elem| Self::linear_map(&elem))
            .collect::<Vec<Self>>()
    }

    #[inline]
    fn scalar_linear_map(y: &E::Fr, key: &CRS<E>) -> Self {
        // = yv, where v = v_2 + (O, P) is a commitment group element
        ( key.v[1] + Com2::<E>::linear_map(&key.g2_gen) ) * *y
    }

    #[inline]
    fn batch_scalar_linear_map(y_vec: &Vec<E::Fr>, key: &CRS<E>) -> Vec<Self> {
        y_vec
            .into_iter()
            .map( |elem| Self::scalar_linear_map(&elem, key))
            .collect::<Vec<Self>>()
    }
}

// ComT<Com1, Com2> is an instantiation of BT<B1, B2>
impl<E: PairingEngine> PartialEq for ComT<E> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1 && self.2 == other.2 && self.3 == other.3
    }
}
impl<E: PairingEngine> Eq for ComT<E> {}

impl<E: PairingEngine> Add<ComT<E>> for ComT<E> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self (
            self.0 * other.0,
            self.1 * other.1,
            self.2 * other.2,
            self.3 * other.3,
        )
    }
}
impl<E: PairingEngine> Zero for ComT<E> {
    #[inline]
    fn zero() -> Self {
        Self (
            E::Fqk::one(),
            E::Fqk::one(),
            E::Fqk::one(),
            E::Fqk::one()
        )
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}
impl<E: PairingEngine> AddAssign<ComT<E>> for ComT<E> {

    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = Self (
            self.0 * other.0,
            self.1 * other.1,
            self.2 * other.2,
            self.3 * other.3
        );
    }
}
impl<E: PairingEngine> Neg for ComT<E> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self (
            E::Fqk::one() / self.0,
            E::Fqk::one() / self.1,
            E::Fqk::one() / self.2,
            E::Fqk::one() / self.3
        )
    }
}
impl<E: PairingEngine> Sub<ComT<E>> for ComT<E> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self (
            self.0 / other.0,
            self.1 / other.1,
            self.2 / other.2,
            self.3 / other.3
        )
    }
}
impl<E: PairingEngine> SubAssign<ComT<E>> for ComT<E> {

    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = Self (
            self.0 / other.0,
            self.1 / other.1,
            self.2 / other.2,
            self.3 / other.3
        );
    }
}
impl<E: PairingEngine> From<Vec<E::Fqk>> for ComT<E> {
    fn from(bts: Vec<E::Fqk>) -> Self {
        assert_eq!(bts.len(), 4);
        Self (
            bts[0],
            bts[1],
            bts[2],
            bts[3]
        )
    }
}
impl<E: PairingEngine> Sum for ComT<E> {
    fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
        iter.fold(
            Self::zero(),
            |a,b| a + b,
        )
    }
}

impl<E: PairingEngine> Mul<ComT<E>> for ComT<E> {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self (
            self.0 * other.0,
            self.1 * other.1,
            self.2 * other.2,
            self.3 * other.3,
        )
    }
}

impl<E: PairingEngine> MulAssign<ComT<E>> for ComT<E> {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = Self (
            self.0 * other.0,
            self.1 * other.1,
            self.2 * other.2,
            self.3 * other.3,
        );
    }
}

impl<E: PairingEngine> B<E> for ComT<E> {}
impl<E: PairingEngine> BT<E, Com1<E>, Com2<E>> for ComT<E> {

    #[inline]
    fn pairing(x: Com1<E>, y: Com2<E>) -> ComT<E> {
        ComT::<E>(
            E::pairing::<E::G1Affine, E::G2Affine>(x.0.clone(), y.0.clone()),
            E::pairing::<E::G1Affine, E::G2Affine>(x.0.clone(), y.1.clone()),
            E::pairing::<E::G1Affine, E::G2Affine>(x.1.clone(), y.0.clone()),
            E::pairing::<E::G1Affine, E::G2Affine>(x.1.clone(), y.1.clone()),
        )
    }

    #[inline]
    fn pairing_sum(x_vec: &Vec<Com1<E>>, y_vec: &Vec<Com2<E>>) -> ComT<E> {

        assert_eq!(x_vec.len(), y_vec.len());
        let xy_vec = x_vec.into_iter().zip(y_vec).collect::<Vec<(&Com1<E>, &Com2<E>)>>();

        xy_vec.into_iter().map(|(&x, &y)| {
            Self::pairing(x, y)
        }).sum()
    }

    fn as_mat(&self) -> Matrix2<E::Fqk> {
        Matrix2::new(
            self.0, self.1,
            self.2, self.3
        )
    }

    #[inline]
    fn linear_map_PPE(z: &E::Fqk) -> Self {
        Self (
            E::Fqk::one(),
            E::Fqk::one(),
            E::Fqk::one(),
            z.clone()
        )
    }

    #[inline]
    fn linear_map_MSMEG1(z: &E::G1Affine, key: &CRS<E>) -> Self {
        Self::pairing(Com1::<E>::linear_map(z), Com2::<E>::scalar_linear_map(&E::Fr::one(), key))
    }

    #[inline]
    fn linear_map_MSMEG2(z: &E::G2Affine, key: &CRS<E>) -> Self {
        Self::pairing(Com1::<E>::scalar_linear_map(&E::Fr::one(), key), Com2::<E>::linear_map(z))
    }

    #[inline]
    fn linear_map_quad(z: &E::Fr, key: &CRS<E>) -> Self {
        Self::pairing(Com1::<E>::scalar_linear_map(&E::Fr::one(), key), Com2::<E>::scalar_linear_map(&E::Fr::one(), key) * *z)
    }
}

/*
// Matrix multiplication algorithm based on source: https://boydjohnson.dev/blog/concurrency-matrix-multiplication/

macro_rules! impl_base_commit_mats {
    (
        $(
            $com:ident
        ),*
    ) => {
        // Repeat for each $com
        $(

            /*
            // Implements scalar point-multiplication for matrices of commitment group elements
            impl<E: PairingEngine> MulAssign<E::Fr> for Matrix<$com<E>> {
                fn mul_assign(&mut self, other: E::Fr) {
                    let m = self.len();
                    let n = self[0].len();
                    let mut smul = Vec::with_capacity(m);
                    for i in 0..m {
                        smul.push(Vec::with_capacity(n));
                        for j in 0..n {
                            let mut elem = self[i][j];
                            elem *= other;
                            smul[i].push(elem.clone());
                        }
                    }
                    *self = smul;
                }
            }
            impl<E: PairingEngine> Neg<Output = Self> for Matrix<$com<E>> {

                #[inline]
                fn neg(self) -> Self::Output {
                   (0..self.len()).map( |i| {
                       let row = &self[i];
                       (0..row.len()).map( |j| {
                           -row[j]
                       })
                       .collect::<Vec<$com<E>>>()
                   })
                   .collect::<Vec<Vec<$com<E>>>>()
                }
            }
            */
            impl<E: PairingEngine> Mat<$com<E>> for Matrix<$com<E>> {
                type Other = E::Fr;

                fn add(&self, other: &Self) -> Self {
                    assert_eq!(self.len(), other.len());
                    assert_eq!(self[0].len(), other[0].len());
                    let m = self.len();
                    let n = self[0].len();
                    let mut add = Vec::with_capacity(m);
                    for i in 0..m {
                        add.push(Vec::with_capacity(n));
                        for j in 0..n {
                            add[i].push(self[i][j].clone() + other[i][j].clone());
                        }
                    }
                    add
                }

                #[inline]
                fn neg(&self) -> Self {
                   (0..self.len()).map( |i| {
                       let row = &self[i];
                       (0..row.len()).map( |j| {
                           -row[j]
                       })
                       .collect::<Vec<$com<E>>>()
                   })
                   .collect::<Vec<Vec<$com<E>>>>()
                }

                fn scalar_mul(&self, other: &Self::Other) -> Self {
                    let m = self.len();
                    let n = self[0].len();
                    let mut smul: Matrix<$com<E>> = Vec::with_capacity(m);
                    for i in 0..m {
                        smul.push(Vec::with_capacity(n));
                        for j in 0..n {
                            smul[i].push(self[i][j] * other);
                        }
                    }
                    smul
                }

                fn transpose(&self) -> Self {
                    let mut trans = Vec::with_capacity(self[0].len());
                    for _ in 0..self[0].len() {
                        trans.push(Vec::with_capacity(self.len()));
                    }

                    for row in self {
                        for i in 0..row.len() {
                            // Push rows onto columns
                            trans[i].push(row[i].clone());
                        }
                    }
                    trans
                }

                fn right_mul(&self, rhs: &Matrix<Self::Other>, is_parallel: bool) -> Self {
                    if self.len() == 0 || self[0].len() == 0 {
                        return vec![];
                    }
                    if rhs.len() == 0 || rhs[0].len() == 0 {
                        return vec![];
                    }

                    // Check that every row in a and column in b has the same length
                    assert_eq!(self[0].len(), rhs.len());
                    let row_dim = self.len();

                    if is_parallel {
                        let mut rows = (0..row_dim)
                            .into_par_iter()
                            .map( |i| {
                                let row = &self[i];
                                let dim = rhs.len();

                                // Perform multiplication for single row
                                // Assuming every column in b has the same length
                                let mut cols = (0..rhs[0].len())
                                    .into_par_iter()
                                    .map( |j| {
                                        (j, (0..dim).map( |k| row[k] * rhs[k][j] ).sum())
                                    })
                                    .collect::<Vec<(usize, $com<E>)>>();

                                // After computing concurrently, sort by index
                                cols.par_sort_by(|left, right| left.0.cmp(&right.0));

                                // Strip off index and return Vec<F>
                                let final_row = cols.into_iter()
                                    .map( |(_, elem)| elem)
                                    .collect();

                                (i, final_row)
                            })
                            .collect::<Vec<(usize, Vec<$com<E>>)>>();

                        // After computing concurrently, sort by index
                        rows.par_sort_by(|left, right| left.0.cmp(&right.0));

                        // Strip off index and return Vec<Vec<F>> (i.e. Matrix<F>)
                        rows.into_iter()
                            .map( |(_, row)| row)
                            .collect()
                    }
                    else {

                        (0..row_dim)
                            .map( |i| {
                                let row = &self[i];
                                let dim = rhs.len();

                                // Perform matrix multiplication for single row
                                // Assuming every column in b has the same length
                                (0..rhs[0].len())
                                    .map( |j| {
                                        (0..dim).map( |k| row[k] * rhs[k][j) ).sum()
                                    })
                                    .collect::<Vec<$com<E>>>()
                            })
                            .collect::<Vec<Vec<$com<E>>>>()
                    }
                }

                fn left_mul(&self, lhs: &Matrix<Self::Other>, is_parallel: bool) -> Self {
                    if lhs.len() == 0 || lhs[0].len() == 0 {
                        return vec![];
                    }
                    if self.len() == 0 || self[0].len() == 0 {
                        return vec![];
                    }

                    // Check that every row in a and column in b has the same length
                    assert_eq!(lhs[0].len(), self.len());
                    let row_dim = lhs.len();

                    if is_parallel {
                        let mut rows = (0..row_dim)
                            .into_par_iter()
                            .map( |i| {
                                let row = &lhs[i];
                                let dim = self.len();

                                // Perform matrix multiplication for single row
                                let mut cols = (0..self[0].len())
                                    .into_par_iter()
                                    .map( |j| {
                                        (j, (0..dim).map( |k| self[k][j] * row[k] ).sum())
                                    })
                                    .collect::<Vec<(usize, $com<E>)>>();

                                // After computing concurrently, sort by index
                                cols.par_sort_by(|left, right| left.0.cmp(&right.0));

                                // Strip off index and return Vec<F>
                                let final_row = cols.into_iter()
                                    .map( |(_, elem)| elem)
                                    .collect();

                                (i, final_row)
                            })
                            .collect::<Vec<(usize, Vec<$com<E>>)>>();

                        // After computing concurrently, sort by index
                        rows.par_sort_by(|left, right| left.0.cmp(&right.0));

                        // Strip off index and return Vec<Vec<F>> (i.e. Matrix<F>)
                        rows.into_iter()
                            .map( |(_, row)| row)
                            .collect()
                    }
                    else {
                        (0..row_dim)
                            .map( |i| {
                                let row = &lhs[i];
                                let dim = self.len();
                                (0..self[0].len())
                                    .map( |j| {
                                        (0..dim).map( |k| self[k][j] * row[k] ).sum()
                                    })
                                    .collect::<Vec<$com<E>>>()
                            })
                            .collect::<Vec<Vec<$com<E>>>>()
                    }
                }
            }
        )*
    }
}
impl_base_commit_mats![Com1, Com2];

/*
// Implements scalar point-multiplication for matrices of commitment group elements
impl MulAssign<F> for Matrix<F> {
    fn mul_assign(&mut self, other: F) {
        let m = self.len();
        let n = self[0].len();
        let mut smul = Vec::with_capacity(m);
        for i in 0..m {
            smul.push(Vec::with_capacity(n));
            for j in 0..n {
                let mut elem = self[i][j];
                elem *= other;
                smul[i].push(elem.clone());
            }
        }
        *self = smul;
    }
}
*/
impl<F: Field> Mat<F> for Matrix<F> {
    type Other = F;

    fn add(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        assert_eq!(self[0].len(), other[0].len());
        let m = self.len();
        let n = self[0].len();
        let mut add = Vec::with_capacity(m);
        for i in 0..m {
            add.push(Vec::with_capacity(n));
            for j in 0..n {
                add[i].push(self[i][j].clone() + other[i][j].clone());
            }
        }
        add
    }

    #[inline]
    fn neg(&self) -> Self {
       (0..self.len()).map( |i| {
           let row = &self[i];
           (0..row.len()).map( |j| {
               -row[j]
           })
           .collect::<Vec<F>>()
       })
       .collect::<Vec<Vec<F>>>()
    }

    fn scalar_mul(&self, other: &Self::Other) -> Self {
        let m = self.len();
        let n = self[0].len();
        let mut smul: Matrix<F> = Vec::with_capacity(m);
        for i in 0..m {
            smul.push(Vec::with_capacity(n));
            for j in 0..n {
                smul[i].push(self[i][j] * other);
            }
        }
        smul
    }

    fn transpose(&self) -> Self {
        let mut trans = Vec::with_capacity(self[0].len());
        for _ in 0..self[0].len() {
            trans.push(Vec::with_capacity(self.len()));
        }

        for row in self {
            for i in 0..row.len() {
                // Push rows onto columns
                trans[i].push(row[i].clone());
            }
        }
        trans
    }

    fn right_mul(&self, rhs: &Matrix<Self::Other>, is_parallel: bool) -> Self {
        if self.len() == 0 || self[0].len() == 0 {
            return vec![];
        }
        if rhs.len() == 0 || rhs[0].len() == 0 {
            return vec![];
        }

        // Check that every row in a and column in b has the same length
        assert_eq!(self[0].len(), rhs.len());
        let row_dim = self.len();

        if is_parallel {
            let mut rows = (0..row_dim)
                .into_par_iter()
                .map( |i| {
                    let row = &self[i];
                    let dim = rhs.len();

                    // Perform multiplication for single row
                    // Assuming every column in b has the same length
                    let mut cols = (0..rhs[0].len())
                        .into_par_iter()
                        .map( |j| {
                            (j, (0..dim).map( |k| row[k] * rhs[k][j] ).sum())
                        })
                        .collect::<Vec<(usize, F)>>();

                    // After computing concurrently, sort by index
                    cols.par_sort_by(|left, right| left.0.cmp(&right.0));

                    // Strip off index and return Vec<F>
                    let final_row = cols.into_iter()
                        .map( |(_, elem)| elem)
                        .collect();

                    (i, final_row)
                })
                .collect::<Vec<(usize, Vec<F>)>>();

            // After computing concurrently, sort by index
            rows.par_sort_by(|left, right| left.0.cmp(&right.0));

            // Strip off index and return Vec<Vec<F>> (i.e. Matrix<F>)
            rows.into_iter()
                .map( |(_, row)| row)
                .collect()
        }
        else {

            (0..row_dim)
                .map( |i| {
                    let row = &self[i];
                    let dim = rhs.len();

                    // Perform matrix multiplication for single row
                    // Assuming every column in b has the same length
                    (0..rhs[0].len())
                        .map( |j| {
                            (0..dim).map( |k| row[k] * rhs[k][j] ).sum()
                        })
                        .collect::<Vec<F>>()
                })
                .collect::<Vec<Vec<F>>>()
        }
    }

    fn left_mul(&self, lhs: &Matrix<Self::Other>, is_parallel: bool) -> Self {
        if lhs.len() == 0 || lhs[0].len() == 0 {
            return vec![];
        }
        if self.len() == 0 || self[0].len() == 0 {
            return vec![];
        }

        // Check that every row in a and column in b has the same length
        assert_eq!(lhs[0].len(), self.len());
        let row_dim = lhs.len();

        if is_parallel {
            let mut rows = (0..row_dim)
                .into_par_iter()
                .map( |i| {
                    let row = &lhs[i];
                    let dim = self.len();

                    // Perform matrix multiplication for single row
                    let mut cols = (0..self[0].len())
                        .into_par_iter()
                        .map( |j| {
                            (j, (0..dim).map( |k| self[k][j] * row[k] ).sum())
                        })
                        .collect::<Vec<(usize, F)>>();

                    // After computing concurrently, sort by index
                    cols.par_sort_by(|left, right| left.0.cmp(&right.0));

                    // Strip off index and return Vec<F>
                    let final_row = cols.into_iter()
                        .map( |(_, elem)| elem)
                        .collect();

                    (i, final_row)
                })
                .collect::<Vec<(usize, Vec<F>)>>();

            // After computing concurrently, sort by index
            rows.par_sort_by(|left, right| left.0.cmp(&right.0));

            // Strip off index and return Vec<Vec<F>> (i.e. Matrix<F>)
            rows.into_iter()
                .map( |(_, row)| row)
                .collect()
        }
        else {
            (0..row_dim)
                .map( |i| {
                    let row = &lhs[i];
                    let dim = self.len();
                    (0..self[0].len())
                        .map( |j| {
                            (0..dim).map( |k| self[k][j] * row[k] ).sum()
                        })
                        .collect::<Vec<F>>()
                })
                .collect::<Vec<Vec<F>>>()
        }
    }
}
*/

#[cfg(test)]
mod tests {

    #![allow(non_snake_case)]
    mod SXDH_com_group {

        use ark_bls12_381::{Bls12_381 as F};
        use ark_ff::UniformRand;
        use ark_ec::ProjectiveCurve;
        use ark_std::test_rng;

        use na::{Matrix3, SMatrix};

        use crate::mat::*;

        // G_1
        type G1Affine = <F as PairingEngine>::G1Affine;
        type G1Projective = <F as PairingEngine>::G1Projective;
        // G_2
        type G2Affine = <F as PairingEngine>::G2Affine;
        type G2Projective = <F as PairingEngine>::G2Projective;
        // Target group (aka G_T)
        type Fqk = <F as PairingEngine>::Fqk;
        // Scalar for elliptic curves
        type Fr = <F as PairingEngine>::Fr;

        // TODO: move/remove
        #[test]
        fn test_na_field_matrix_add() {

            // 3 x 3 matrices
            let one = Fr::one();
            let lhs: SMatrix<Fr, 3, 3> = Matrix3::new(
                one, field_new!(Fr, "2"), field_new!(Fr, "3"),
                field_new!(Fr, "4"), field_new!(Fr, "5"), field_new!(Fr, "6"),
                field_new!(Fr, "7"), field_new!(Fr, "8"), field_new!(Fr, "9")
            );
            let rhs: SMatrix<Fr, 3, 3> = Matrix3::new(
                field_new!(Fr, "10"), field_new!(Fr, "11"), field_new!(Fr, "12"),
                field_new!(Fr, "13"), field_new!(Fr, "14"), field_new!(Fr, "15"),
                field_new!(Fr, "16"), field_new!(Fr, "17"), field_new!(Fr, "18")
            );

            let exp: SMatrix<Fr, 3, 3> = Matrix3::new(
                field_new!(Fr, "11"), field_new!(Fr, "13"), field_new!(Fr, "15"),
                field_new!(Fr, "17"), field_new!(Fr, "19"), field_new!(Fr, "21"),
                field_new!(Fr, "23"), field_new!(Fr, "25"), field_new!(Fr, "27")
            );
            let lr = lhs + rhs;
            let rl = rhs + lhs;

            assert!(lr.shape().0 == 3 && lr.shape().1 == 3);
            assert_eq!(exp, lr);
            assert_eq!(lr, rl);
        }

        #[test]
        fn test_B1_add_zero() {
            let mut rng = test_rng();
            let a = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let zero = Com1::<F>(
                G1Affine::zero(),
                G1Affine::zero()
            );
            let asub = a + zero;

            assert_eq!(zero, Com1::<F>::zero());
            assert!(zero.is_zero());
            assert_eq!(a, asub);
        }

        #[test]
        fn test_B2_add_zero() {
            let mut rng = test_rng();
            let a = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            );
            let zero = Com2::<F>(
                G2Affine::zero(),
                G2Affine::zero()
            );
            let asub = a + zero;

            assert_eq!(zero, Com2::<F>::zero());
            assert!(zero.is_zero());
            assert_eq!(a, asub);
        }

        #[test]
        fn test_BT_add_zero() {
            let mut rng = test_rng();
            let a = ComT::<F>(
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng)
            );
            let zero = ComT::<F>(
                Fqk::one(),
                Fqk::one(),
                Fqk::one(),
                Fqk::one()
            );
            let asub = a + zero;

            assert_eq!(zero, ComT::<F>::zero());
            assert!(zero.is_zero());
            assert_eq!(a, asub);
        }

        #[test]
        fn test_B1_add() {
            let mut rng = test_rng();
            let a = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let b = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let ab = a + b;
            let ba = b + a;

            assert_eq!(ab, Com1::<F>(a.0 + b.0, a.1 + b.1));
            assert_eq!(ab, ba);
        }

        #[test]
        fn test_B2_add() {
            let mut rng = test_rng();
            let a = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            );
            let b = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            );
            let ab = a + b;
            let ba = b + a;

            assert_eq!(ab, Com2::<F>(a.0 + b.0, a.1 + b.1));
            assert_eq!(ab, ba);
        }

        #[test]
        fn test_BT_add() {
            let mut rng = test_rng();
            let a = ComT::<F>(
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng)
            );
            let b = ComT::<F>(
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng)
            );
            let ab = a + b;
            let ba = b + a;

            assert_eq!(ab, ComT::<F>(a.0 * b.0, a.1 * b.1, a.2 * b.2, a.3 * b.3));
            assert_eq!(ab, ba);
        }


        #[test]
        fn test_B1_sum() {
            let mut rng = test_rng();
            let a = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let b = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let c = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );

            let abc_vec = vec![a, b, c];
            let abc: Com1<F> = abc_vec.into_iter().sum();

            assert_eq!(abc, a + b + c);
        }

        #[test]
        fn test_B2_sum() {
            let mut rng = test_rng();
            let a = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            );
            let b = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            );
            let c = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            );

            let abc_vec = vec![a, b, c];
            let abc: Com2<F> = abc_vec.into_iter().sum();

            assert_eq!(abc, a + b + c);
        }

        #[test]
        fn test_BT_sum() {
            let mut rng = test_rng();
            let a = ComT::<F>(
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng)
            );
            let b = ComT::<F>(
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng)
            );
            let c = ComT::<F>(
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng)
            );

            let abc_vec = vec![a, b, c];
            let abc: ComT<F> = abc_vec.into_iter().sum();

            assert_eq!(abc, a + b + c);
        }

        #[test]
        fn test_B1_neg() {
            let mut rng = test_rng();
            let b = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let bneg = -b;
            let zero = b + bneg;

            assert!(zero.is_zero());
        }

        #[test]
        fn test_B2_neg() {
            let mut rng = test_rng();
            let b = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            );
            let bneg = -b;
            let zero = b + bneg;

            assert!(zero.is_zero());
        }

        #[test]
        fn test_BT_neg() {
            let mut rng = test_rng();
            let b = ComT::<F>(
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
            );
            let bneg = -b;
            let zero = b + bneg;

            assert!(zero.is_zero());
        }

        #[test]
        fn test_B1_sub() {
            let mut rng = test_rng();
            let a = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let b = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let ab = a - b;
            let ba = b - a;

            assert_eq!(ab, Com1::<F>(a.0 + -b.0, a.1 + -b.1));
            assert_eq!(ab, -ba);
        }

        #[test]
        fn test_B2_sub() {
            let mut rng = test_rng();
            let a = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            );
            let b = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            );
            let ab = a - b;
            let ba = b - a;

            assert_eq!(ab, Com2::<F>(a.0 + -b.0, a.1 + -b.1));
            assert_eq!(ab, -ba);
        }

        #[test]
        fn test_BT_sub() {
            let mut rng = test_rng();
            let a = ComT::<F>(
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng)
            );
            let b = ComT::<F>(
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng),
                Fqk::rand(&mut rng)
            );
            let ab = a - b;
            let ba = b - a;

            assert_eq!(ab, ComT::<F>(a.0 / b.0, a.1 / b.1, a.2 / b.2, a.3 / b.3));
            assert_eq!(ab, -ba);
        }

        #[test]
        fn test_B1_scalar_mul() {
            let mut rng = test_rng();
            let b = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let scalar = Fr::rand(&mut rng);
            let b0 = b.0.mul(scalar);
            let b1 = b.1.mul(scalar);
            let bres = b * scalar;
            let bexp = Com1::<F>(b0.into_affine(), b1.into_affine());

            assert_eq!(bres, bexp);
        }

        #[test]
        fn test_B2_scalar_mul() {
            let mut rng = test_rng();
            let b = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let scalar = Fr::rand(&mut rng);
            let b0 = b.0.mul(scalar);
            let b1 = b.1.mul(scalar);
            let bres = b * scalar;
            let bexp = Com1::<F>(b0.into_affine(), b1.into_affine());

            assert_eq!(bres, bexp);
        }

        #[test]
        fn test_B_pairing_zero_G1() {
            let mut rng = test_rng();
            let b1 = Com1::<F>(
                G1Affine::zero(),
                G1Affine::zero()
            );
            let b2 = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            );
            let bt = ComT::pairing(b1.clone(), b2.clone());

            assert_eq!(bt.0, Fqk::one());
            assert_eq!(bt.1, Fqk::one());
            assert_eq!(bt.2, Fqk::one());
            assert_eq!(bt.3, Fqk::one());
        }

        #[test]
        fn test_B_pairing_zero_G2() {
            let mut rng = test_rng();
            let b1 = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let b2 = Com2::<F>(
                G2Affine::zero(),
                G2Affine::zero()
            );
            let bt = ComT::pairing(b1.clone(), b2.clone());

            assert_eq!(bt.0, Fqk::one());
            assert_eq!(bt.1, Fqk::one());
            assert_eq!(bt.2, Fqk::one());
            assert_eq!(bt.3, Fqk::one());
        }

        #[test]
        fn test_B_pairing_commit() {
            let mut rng = test_rng();
            let b1 = Com1::<F>(
                G1Affine::zero(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let b2 = Com2::<F>(
                G2Affine::zero(),
                G2Projective::rand(&mut rng).into_affine()
            );
            let bt = ComT::pairing(b1.clone(), b2.clone());

            assert_eq!(bt.0, Fqk::one());
            assert_eq!(bt.1, Fqk::one());
            assert_eq!(bt.2, Fqk::one());
            assert_eq!(bt.3, F::pairing::<G1Affine, G2Affine>(b1.1.clone(), b2.1.clone()));
        }

        #[test]
        fn test_B_pairing_rand() {
            let mut rng = test_rng();
            let b1 = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let b2 = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            );
            let bt = ComT::pairing(b1.clone(), b2.clone());

            assert_eq!(bt.0, F::pairing::<G1Affine, G2Affine>(b1.0.clone(), b2.0.clone()));
            assert_eq!(bt.1, F::pairing::<G1Affine, G2Affine>(b1.0.clone(), b2.1.clone()));
            assert_eq!(bt.2, F::pairing::<G1Affine, G2Affine>(b1.1.clone(), b2.0.clone()));
            assert_eq!(bt.3, F::pairing::<G1Affine, G2Affine>(b1.1.clone(), b2.1.clone()));
        }

        #[test]
        fn test_B_pairing_sum() {
            let mut rng = test_rng();
            let x1 = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
            );
            let x2 = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
            );
            let y1 = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            );
            let y2 = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            );
            let x = vec![x1, x2];
            let y = vec![y1, y2];
            let exp: ComT<F> = vec![ ComT::<F>::pairing(x1, y1), ComT::<F>::pairing(x2, y2) ].into_iter().sum();
            let res: ComT<F> = ComT::<F>::pairing_sum(&x, &y);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_B_into_mat() {

            let mut rng = test_rng();
            let b1 = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            );
            let b2 = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            );
            let bt = ComT::pairing(b1.clone(), b2.clone());

            // B1 and B2 can be representing as 2-dim column vectors
            assert_eq!(b1.as_mat(), Matrix2x1::new(b1.0, b1.1));
            assert_eq!(b2.as_mat(), Matrix2x1::new(b2.0, b2.1));
            // BT can be represented as a 2 x 2 matrix
            assert_eq!(bt.as_mat(), Matrix2::new(bt.0, bt.1, bt.2, bt.3));
        }

        #[test]
        fn test_B_from_mat() {

            let mut rng = test_rng();
            let b1_vec = vec![
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine()
            ];
            let b2_vec = vec![
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine()
            ];
            let bt_vec = vec![
                F::pairing::<G1Affine, G2Affine>(b1_vec[0].clone(), b2_vec[0].clone()),
                F::pairing::<G1Affine, G2Affine>(b1_vec[0].clone(), b2_vec[1].clone()),
                F::pairing::<G1Affine, G2Affine>(b1_vec[1].clone(), b2_vec[0].clone()),
                F::pairing::<G1Affine, G2Affine>(b1_vec[1].clone(), b2_vec[1].clone())
            ];

            let b1 = Com1::<F>::from(b1_vec.clone());
            let b2 = Com2::<F>::from(b2_vec.clone());
            let bt = ComT::<F>::from(bt_vec.clone());

            assert_eq!(b1.0, b1_vec[0]);
            assert_eq!(b1.1, b1_vec[1]);
            assert_eq!(b2.0, b2_vec[0]);
            assert_eq!(b2.1, b2_vec[1]);
            assert_eq!(bt.0, bt_vec[0]);
            assert_eq!(bt.1, bt_vec[1]);
            assert_eq!(bt.2, bt_vec[2]);
            assert_eq!(bt.3, bt_vec[3]);
        }

        #[test]
        fn test_batched_linear_maps() {
            let mut rng = test_rng();
            let vec_g1 = vec![G1Projective::rand(&mut rng).into_affine(), G1Projective::rand(&mut rng).into_affine()];
            let vec_g2 = vec![G2Projective::rand(&mut rng).into_affine(), G2Projective::rand(&mut rng).into_affine()];
            let vec_b1 = Com1::<F>::batch_linear_map(&vec_g1);
            let vec_b2 = Com2::<F>::batch_linear_map(&vec_g2);

            assert_eq!(vec_b1[0], Com1::<F>::linear_map(&vec_g1[0]));
            assert_eq!(vec_b1[1], Com1::<F>::linear_map(&vec_g1[1]));
            assert_eq!(vec_b2[0], Com2::<F>::linear_map(&vec_g2[0]));
            assert_eq!(vec_b2[1], Com2::<F>::linear_map(&vec_g2[1]));
        }

        #[test]
        fn test_batched_scalar_linear_maps() {
            let mut rng = test_rng();
            let key = CRS::<F>::generate_crs(&mut rng);

            let vec_scalar = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];
            let vec_b1 = Com1::<F>::batch_scalar_linear_map(&vec_scalar, &key);
            let vec_b2 = Com2::<F>::batch_scalar_linear_map(&vec_scalar, &key);

            assert_eq!(vec_b1[0], Com1::<F>::scalar_linear_map(&vec_scalar[0], &key));
            assert_eq!(vec_b1[1], Com1::<F>::scalar_linear_map(&vec_scalar[1], &key));
            assert_eq!(vec_b2[0], Com2::<F>::scalar_linear_map(&vec_scalar[0], &key));
            assert_eq!(vec_b2[1], Com2::<F>::scalar_linear_map(&vec_scalar[1], &key));
        }

        #[test]
        fn test_PPE_linear_maps() {

            let mut rng = test_rng();
            let a1 = G1Projective::rand(&mut rng).into_affine();
            let a2 = G2Projective::rand(&mut rng).into_affine();
            let at = F::pairing(a1.clone(), a2.clone());
            let b1 = Com1::<F>::linear_map(&a1);
            let b2 = Com2::<F>::linear_map(&a2);
            let bt = ComT::<F>::linear_map_PPE(&at);

            assert_eq!(b1.0, G1Affine::zero());
            assert_eq!(b1.1, a1);
            assert_eq!(b2.0, G2Affine::zero());
            assert_eq!(b2.1, a2);
            assert_eq!(bt.0, Fqk::one());
            assert_eq!(bt.1, Fqk::one());
            assert_eq!(bt.2, Fqk::one());
            assert_eq!(bt.3, F::pairing(a1.clone(), a2.clone()));
        }

        // Test that we're using the linear map that preserves witness-indistinguishability (see Ghadafi et al. 2010)
        #[test]
        fn test_MSMEG1_linear_maps() {

            let mut rng = test_rng();
            let key = CRS::<F>::generate_crs(&mut rng);

            let a1 = G1Projective::rand(&mut rng).into_affine();
            let a2 = Fr::rand(&mut rng);
            let at = a1.mul(a2).into_affine();
            let b1 = Com1::<F>::linear_map(&a1);
            let b2 = Com2::<F>::scalar_linear_map(&a2, &key);
            let bt = ComT::<F>::linear_map_MSMEG1(&at, &key);

            assert_eq!(b1.0, G1Affine::zero());
            assert_eq!(b1.1, a1);
            assert_eq!(b2.0, key.v[1].0.mul(a2));
            assert_eq!(b2.1, (key.v[1].1 + key.g2_gen).mul(a2));
            assert_eq!(bt.0, Fqk::one());
            assert_eq!(bt.1, Fqk::one());
            assert_eq!(bt.2, F::pairing(at.clone(), key.v[1].0.clone()));
            assert_eq!(bt.3, F::pairing(at.clone(), key.v[1].1.clone() + key.g2_gen.clone()));
        }

        // Test that we're using the linear map that preserves witness-indistinguishability (see Ghadafi et al. 2010)
        #[test]
        fn test_MSMEG2_linear_maps() {

            let mut rng = test_rng();
            let key = CRS::<F>::generate_crs(&mut rng);

            let a1 = Fr::rand(&mut rng);
            let a2 = G2Projective::rand(&mut rng).into_affine();
            let at = a2.mul(a1).into_affine();
            let b1 = Com1::<F>::scalar_linear_map(&a1, &key);
            let b2 = Com2::<F>::linear_map(&a2);
            let bt = ComT::<F>::linear_map_MSMEG2(&at, &key);

            assert_eq!(b1.0, key.u[1].0.mul(a1));
            assert_eq!(b1.1, (key.u[1].1 + key.g1_gen).mul(a1));
            assert_eq!(b2.0, G2Affine::zero());
            assert_eq!(b2.1, a2);
            assert_eq!(bt.0, Fqk::one());
            assert_eq!(bt.1, F::pairing(key.u[1].0.clone(), at.clone()));
            assert_eq!(bt.2, Fqk::one());
            assert_eq!(bt.3, F::pairing(key.u[1].1.clone() + key.g1_gen.clone(), at.clone()));
        }

        // Test that we're using the linear map that preserves witness-indistinguishability (see Ghadafi et al. 2010)
        #[test]
        fn test_QuadEqu_linear_maps() {

            let mut rng = test_rng();
            let key = CRS::<F>::generate_crs(&mut rng);

            let a1 = Fr::rand(&mut rng);
            let a2 = Fr::rand(&mut rng);
            let at = a1 * a2;
            let b1 = Com1::<F>::scalar_linear_map(&a1, &key);
            let b2 = Com2::<F>::scalar_linear_map(&a2, &key);
            let bt = ComT::<F>::linear_map_quad(&at, &key);
            let W1 = Com1::<F>(
                key.u[1].0,
                key.u[1].1 + key.g1_gen
            );
            let W2 = Com2::<F>(
                key.v[1].0,
                key.v[1].1 + key.g2_gen
            );
            assert_eq!(b1.0, W1.0.mul(a1));
            assert_eq!(b1.1, W1.1.mul(a1));
            assert_eq!(b2.0, W2.0.mul(a2));
            assert_eq!(b2.1, W2.1.mul(a2));
            assert_eq!(bt, ComT::<F>::pairing(W1 * a1, W2 * a2));
            assert_eq!(bt, ComT::<F>::pairing(W1, W2 * at));
        }
    }

/*
    mod matrix {

        use ark_bls12_381::{Bls12_381 as F};
        use ark_ff::{UniformRand, field_new};
        use ark_ec::ProjectiveCurve;
        use ark_std::test_rng;
        use na::{Matrix};

        use crate::data_structures::*;

        type G1Affine = <F as PairingEngine>::G1Affine;
        type G1Projective = <F as PairingEngine>::G1Projective;
        type G2Affine = <F as PairingEngine>::G2Affine;
        type G2Projective = <F as PairingEngine>::G2Projective;
        type Fr = <F as PairingEngine>::Fr;

        // Uses an affine group generator to produce an affine group element represented by the numeric string.
        #[allow(unused_macros)]
        macro_rules! affine_group_new {
            ($gen:expr, $strnum:tt) => {
                $gen.mul(field_new!(Fr, $strnum)).into_affine()
            }
        }

        // Uses an affine group generator to produce a projective group element represented by the numeric string.
        #[allow(unused_macros)]
        macro_rules! projective_group_new {
            ($gen:expr, $strnum:tt) => {
                $gen.mul(field_new!(Fr, $strnum))
            }
        }

        #[test]
        fn test_field_matrix_left_mul_entry() {

            // 1 x 3 (row) vector
            let one = Fr::one();
            let lhs: Matrix<Fr> = vec![vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")]];
            // 3 x 1 (column) vector
            let rhs: Matrix<Fr> = vec![
                vec![field_new!(Fr, "4")],
                vec![field_new!(Fr, "5")],
                vec![field_new!(Fr, "6")]
            ];
            let exp: Matrix<Fr> = vec![vec![field_new!(Fr, "32")]];
            let res: Matrix<Fr> = rhs.left_mul(&lhs, false);

            // 1 x 1 resulting matrix
            assert_eq!(res.len(), 1);
            assert_eq!(res[0].len(), 1);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_right_mul_entry() {

            // 1 x 3 (row) vector
            let one = Fr::one();
            let lhs: Matrix<Fr> = vec![vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")]];
            // 3 x 1 (column) vector
            let rhs: Matrix<Fr> = vec![
                vec![field_new!(Fr, "4")],
                vec![field_new!(Fr, "5")],
                vec![field_new!(Fr, "6")]
            ];
            let exp: Matrix<Fr> = vec![vec![field_new!(Fr, "32")]];
            let res: Matrix<Fr> = lhs.right_mul(&rhs, false);

            // 1 x 1 resulting matrix
            assert_eq!(res.len(), 1);
            assert_eq!(res[0].len(), 1);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_left_mul() {

            // 2 x 3 matrix
            let one = Fr::one();
            let lhs: Matrix<Fr> = vec![
                vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")],
                vec![field_new!(Fr, "4"), field_new!(Fr, "5"), field_new!(Fr, "6")]
            ];
            // 3 x 4 matrix
            let rhs: Matrix<Fr> = vec![
                vec![field_new!(Fr, "7"), field_new!(Fr, "8"), field_new!(Fr, "9"), field_new!(Fr, "10")],
                vec![field_new!(Fr, "11"), field_new!(Fr, "12"), field_new!(Fr, "13"), field_new!(Fr, "14")],
                vec![field_new!(Fr, "15"), field_new!(Fr, "16"), field_new!(Fr, "17"), field_new!(Fr, "18")]
            ];
            let exp: Matrix<Fr> = vec![
                vec![field_new!(Fr, "74"), field_new!(Fr, "80"), field_new!(Fr, "86"), field_new!(Fr, "92")],
                vec![field_new!(Fr, "173"), field_new!(Fr, "188"), field_new!(Fr, "203"), field_new!(Fr, "218")]
            ];
            let res: Matrix<Fr> = rhs.left_mul(&lhs, false);

            // 2 x 4 resulting matrix
            assert_eq!(res.len(), 2);
            assert_eq!(res[0].len(), 4);
            assert_eq!(res[1].len(), 4);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_right_mul() {

            // 2 x 3 matrix
            let one = Fr::one();
            let lhs: Matrix<Fr> = vec![
                vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")],
                vec![field_new!(Fr, "4"), field_new!(Fr, "5"), field_new!(Fr, "6")]
            ];
            // 3 x 4 matrix
            let rhs: Matrix<Fr> = vec![
                vec![field_new!(Fr, "7"), field_new!(Fr, "8"), field_new!(Fr, "9"), field_new!(Fr, "10")],
                vec![field_new!(Fr, "11"), field_new!(Fr, "12"), field_new!(Fr, "13"), field_new!(Fr, "14")],
                vec![field_new!(Fr, "15"), field_new!(Fr, "16"), field_new!(Fr, "17"), field_new!(Fr, "18")]
            ];
            let exp: Matrix<Fr> = vec![
                vec![field_new!(Fr, "74"), field_new!(Fr, "80"), field_new!(Fr, "86"), field_new!(Fr, "92")],
                vec![field_new!(Fr, "173"), field_new!(Fr, "188"), field_new!(Fr, "203"), field_new!(Fr, "218")]
            ];
            let res: Matrix<Fr> = lhs.right_mul(&rhs, false);

            // 2 x 4 resulting matrix
            assert_eq!(res.len(), 2);
            assert_eq!(res[0].len(), 4);
            assert_eq!(res[1].len(), 4);

            assert_eq!(exp, res);
        }


        #[test]
        fn test_field_matrix_left_mul_par() {

            // 2 x 3 matrix
            let one = Fr::one();
            let lhs: Matrix<Fr> = vec![
                vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")],
                vec![field_new!(Fr, "4"), field_new!(Fr, "5"), field_new!(Fr, "6")]
            ];
            // 3 x 4 matrix
            let rhs: Matrix<Fr> = vec![
                vec![field_new!(Fr, "7"), field_new!(Fr, "8"), field_new!(Fr, "9"), field_new!(Fr, "10")],
                vec![field_new!(Fr, "11"), field_new!(Fr, "12"), field_new!(Fr, "13"), field_new!(Fr, "14")],
                vec![field_new!(Fr, "15"), field_new!(Fr, "16"), field_new!(Fr, "17"), field_new!(Fr, "18")]
            ];
            let exp: Matrix<Fr> = vec![
                vec![field_new!(Fr, "74"), field_new!(Fr, "80"), field_new!(Fr, "86"), field_new!(Fr, "92")],
                vec![field_new!(Fr, "173"), field_new!(Fr, "188"), field_new!(Fr, "203"), field_new!(Fr, "218")]
            ];
            let res: Matrix<Fr> = rhs.left_mul(&lhs, true);

            // 2 x 4 resulting matrix
            assert_eq!(res.len(), 2);
            assert_eq!(res[0].len(), 4);
            assert_eq!(res[1].len(), 4);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_right_mul_par() {

            // 2 x 3 matrix
            let one = Fr::one();
            let lhs: Matrix<Fr> = vec![
                vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")],
                vec![field_new!(Fr, "4"), field_new!(Fr, "5"), field_new!(Fr, "6")]
            ];
            // 3 x 4 matrix
            let rhs: Matrix<Fr> = vec![
                vec![field_new!(Fr, "7"), field_new!(Fr, "8"), field_new!(Fr, "9"), field_new!(Fr, "10")],
                vec![field_new!(Fr, "11"), field_new!(Fr, "12"), field_new!(Fr, "13"), field_new!(Fr, "14")],
                vec![field_new!(Fr, "15"), field_new!(Fr, "16"), field_new!(Fr, "17"), field_new!(Fr, "18")]
            ];
            let exp: Matrix<Fr> = vec![
                vec![field_new!(Fr, "74"), field_new!(Fr, "80"), field_new!(Fr, "86"), field_new!(Fr, "92")],
                vec![field_new!(Fr, "173"), field_new!(Fr, "188"), field_new!(Fr, "203"), field_new!(Fr, "218")]
            ];
            let res: Matrix<Fr> = lhs.right_mul(&rhs, true);

            // 2 x 4 resulting matrix
            assert_eq!(res.len(), 2);
            assert_eq!(res[0].len(), 4);
            assert_eq!(res[1].len(), 4);

            assert_eq!(exp, res);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B1_matrix_left_mul_entry() {

            // 1 x 3 (row) vector
            let one = Fr::one();
            let mut rng = test_rng();
            let g1gen = G1Projective::rand(&mut rng).into_affine();

            let lhs: Matrix<Fr> = vec![vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")]];
            // 3 x 1 (column) vector
            let rhs: Matrix<Com1<F>> = vec![
                vec![ Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "4") ) ],
                vec![ Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "5") ) ],
                vec![ Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "6") ) ]
            ];
            let exp: Matrix<Com1<F>> = vec![vec![ Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "32") )]];
            let res: Matrix<Com1<F>> = rhs.left_mul(&lhs, false);

            // 1 x 1 resulting matrix
            assert_eq!(res.len(), 1);
            assert_eq!(res[0].len(), 1);

            assert_eq!(exp, res);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B1_matrix_right_mul_entry() {

            // 1 x 3 (row) vector
            let mut rng = test_rng();
            let g1gen = G1Projective::rand(&mut rng).into_affine();
            let lhs: Matrix<Com1<F>> = vec![vec![
                Com1::<F>( G1Affine::zero(), g1gen),
                Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "2")),
                Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "3"))
            ]];
            // 3 x 1 (column) vector
            let rhs: Matrix<Fr> = vec![
                vec![field_new!(Fr, "4")],
                vec![field_new!(Fr, "5")],
                vec![field_new!(Fr, "6")]
            ];
            let exp: Matrix<Com1<F>> = vec![vec![ Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "32") ) ]];
            let res: Matrix<Com1<F>> = lhs.right_mul(&rhs, false);

            // 1 x 1 resulting matrix
            assert_eq!(res.len(), 1);
            assert_eq!(res[0].len(), 1);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_scalar_mul() {

            // 3 x 3 matrices
            let one = Fr::one();
            let scalar: Fr = field_new!(Fr, "3");
            let mat: Matrix<Fr> = vec![
                vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")],
                vec![field_new!(Fr, "4"), field_new!(Fr, "5"), field_new!(Fr, "6")],
                vec![field_new!(Fr, "7"), field_new!(Fr, "8"), field_new!(Fr, "9")]
            ];

            let exp: Matrix<Fr> = vec![
                vec![field_new!(Fr, "3"), field_new!(Fr, "6"), field_new!(Fr, "9")],
                vec![field_new!(Fr, "12"), field_new!(Fr, "15"), field_new!(Fr, "18")],
                vec![field_new!(Fr, "21"), field_new!(Fr, "24"), field_new!(Fr, "27")]
            ];
            let res: Matrix<Fr> = mat.scalar_mul(&scalar);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_B1_matrix_scalar_mul() {

            let scalar: Fr = field_new!(Fr, "3");

            // 3 x 3 matrix of Com1 elements (0, 3)
            let mut rng = test_rng();
            let g1gen = G1Projective::rand(&mut rng).into_affine();
            let mut mat: Matrix<Com1<F>> = Vec::with_capacity(3);

            for i in 0..3 {
                mat.push(Vec::with_capacity(3));
                for _ in 0..3 {

                    mat[i].push( Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "1") ) );
                }
            }

            let mut exp: Matrix<Com1<F>> = Vec::with_capacity(3);
            for i in 0..3 {
                exp.push(Vec::with_capacity(3));
                for _ in 0..3 {
                    exp[i].push( Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "3") ) );
                }
            }

            let res: Matrix<Com1<F>> = mat.scalar_mul(&scalar);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_B2_matrix_scalar_mul() {

            let scalar: Fr = field_new!(Fr, "3");

            // 3 x 3 matrix of Com1 elements (0, 3)
            let mut rng = test_rng();
            let g2gen = G2Projective::rand(&mut rng).into_affine();
            let mut mat: Matrix<Com2<F>> = Vec::with_capacity(3);

            for i in 0..3 {
                mat.push(Vec::with_capacity(3));
                for _ in 0..3 {

                    mat[i].push( Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "1") ) );
                }
            }

            let mut exp: Matrix<Com2<F>> = Vec::with_capacity(3);
            for i in 0..3 {
                exp.push(Vec::with_capacity(3));
                for _ in 0..3 {
                    exp[i].push( Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "3") ) );
                }
            }

            let res: Matrix<Com2<F>> = mat.scalar_mul(&scalar);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_B1_transpose_vec() {
            let mut rng = test_rng();
            let g1gen = G1Projective::rand(&mut rng).into_affine();
            // 1 x 3 (row) vector
            let mat: Matrix<Com1<F>> = vec![vec![
                Com1::<F>( G1Affine::zero(), g1gen),
                Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "2")),
                Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "3"))
            ]];
            // 3 x 1 transpose (column) vector
            let exp: Matrix<Com1<F>> = vec![
                vec![Com1::<F>( G1Affine::zero(), g1gen)],
                vec![Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "2"))],
                vec![Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "3"))]
            ];
            let res: Matrix<Com1<F>> = mat.transpose();

            assert_eq!(res.len(), 3);
            for i in 0..res.len() {
                assert_eq!(res[i].len(), 1);
            }
            assert_eq!(exp, res);
        }

        #[test]
        fn test_B1_matrix_transpose() {

            // 3 x 3 matrix
            let mut rng = test_rng();
            let g1gen = G1Projective::rand(&mut rng).into_affine();
            let mat: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>( G1Affine::zero(), g1gen ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "2") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "3") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "4") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "5") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "6") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "7") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "8") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "9") )
                ]

            ];
            // 3 x 3 transpose matrix
            let exp: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>( G1Affine::zero(), g1gen ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "4") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "7") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "2") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "5") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "8") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "3") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "6") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "9") )
                ]
            ];
            let res: Matrix<Com1<F>> = mat.transpose();

            assert_eq!(res.len(), 3);
            for i in 0..res.len() {
                assert_eq!(res[i].len(), 3);
            }

            assert_eq!(exp, res);
        }

        #[test]
        fn test_B2_transpose_vec() {
            let mut rng = test_rng();
            let g2gen = G2Projective::rand(&mut rng).into_affine();
            // 1 x 3 (row) vector
            let mat: Matrix<Com2<F>> = vec![vec![
                Com2::<F>( G2Affine::zero(), g2gen),
                Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "2")),
                Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "3"))
            ]];
            // 3 x 1 transpose (column) vector
            let exp: Matrix<Com2<F>> = vec![
                vec![Com2::<F>( G2Affine::zero(), g2gen)],
                vec![Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "2"))],
                vec![Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "3"))]
            ];
            let res: Matrix<Com2<F>> = mat.transpose();

            assert_eq!(res.len(), 3);
            for i in 0..res.len() {
                assert_eq!(res[i].len(), 1);
            }
            assert_eq!(exp, res);
        }

        #[test]
        fn test_B2_matrix_transpose() {

            // 3 x 3 matrix
            let mut rng = test_rng();
            let g2gen = G2Projective::rand(&mut rng).into_affine();
            let mat: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>( G2Affine::zero(), g2gen ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "2") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "3") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "4") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "5") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "6") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "7") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "8") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "9") )
                ]

            ];
            // 3 x 3 transpose matrix
            let exp: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>( G2Affine::zero(), g2gen ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "4") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "7") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "2") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "5") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "8") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "3") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "6") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "9") )
                ]
            ];
            let res: Matrix<Com2<F>> = mat.transpose();

            assert_eq!(res.len(), 3);
            for i in 0..res.len() {
                assert_eq!(res[i].len(), 3);
            }

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_transpose_vec() {

            // 1 x 3 (row) vector
            let one = Fr::one();
            let mat: Matrix<Fr> = vec![vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")]];

            // 3 x 1 transpose (column) vector
            let exp: Matrix<Fr> = vec![
                vec![one],
                vec![field_new!(Fr, "2")],
                vec![field_new!(Fr, "3")]
            ];
            let res: Matrix<Fr> = mat.transpose();

            assert_eq!(res.len(), 3);
            for i in 0..res.len() {
                assert_eq!(res[i].len(), 1);
            }

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_transpose() {

            // 3 x 3 matrix
            let one = Fr::one();
            let mat: Matrix<Fr> = vec![
                vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")],
                vec![field_new!(Fr, "4"), field_new!(Fr, "5"), field_new!(Fr, "6")],
                vec![field_new!(Fr, "7"), field_new!(Fr, "8"), field_new!(Fr, "9")]

            ];
            // 3 x 3 transpose matrix
            let exp: Matrix<Fr> = vec![
                vec![one, field_new!(Fr, "4"), field_new!(Fr, "7")],
                vec![field_new!(Fr, "2"), field_new!(Fr, "5"), field_new!(Fr, "8")],
                vec![field_new!(Fr, "3"), field_new!(Fr, "6"), field_new!(Fr, "9")]
            ];
            let res: Matrix<Fr> = mat.transpose();

            assert_eq!(res.len(), 3);
            for i in 0..res.len() {
                assert_eq!(res[i].len(), 3);
            }

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_neg() {

            // 3 x 3 matrix
            let one = Fr::one();
            let mat: Matrix<Fr> = vec![
                vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")],
                vec![field_new!(Fr, "4"), field_new!(Fr, "5"), field_new!(Fr, "6")],
                vec![field_new!(Fr, "7"), field_new!(Fr, "8"), field_new!(Fr, "9")]

            ];
            // 3 x 3 transpose matrix
            let exp: Matrix<Fr> = vec![
                vec![-one, field_new!(Fr, "-2"), field_new!(Fr, "-3")],
                vec![field_new!(Fr, "-4"), field_new!(Fr, "-5"), field_new!(Fr, "-6")],
                vec![field_new!(Fr, "-7"), field_new!(Fr, "-8"), field_new!(Fr, "-9")]
            ];
            let res: Matrix<Fr> = mat.neg();

            assert_eq!(res.len(), 3);
            for i in 0..res.len() {
                assert_eq!(res[i].len(), 3);
            }

            assert_eq!(exp, res);
        }


        #[test]
        fn test_B1_matrix_neg() {

            // 3 x 3 matrix
            let mut rng = test_rng();
            let g1gen = G1Projective::rand(&mut rng).into_affine();
            let mat: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>( G1Affine::zero(), g1gen ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "2") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "3") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "4") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "5") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "6") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "7") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "8") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "9") )
                ]

            ];
            // 3 x 3 transpose matrix
            let exp: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>( G1Affine::zero(), -g1gen ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "-2") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "-3") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "-4") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "-5") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "-6") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "-7") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "-8") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "-9") )
                ]
            ];
            let res: Matrix<Com1<F>> = mat.neg();

            assert_eq!(res.len(), 3);
            for i in 0..res.len() {
                assert_eq!(res[i].len(), 3);
            }

            assert_eq!(exp, res);
        }

        #[test]
        fn test_B2_matrix_neg() {

            // 3 x 3 matrix
            let mut rng = test_rng();
            let g2gen = G2Projective::rand(&mut rng).into_affine();
            let mat: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>( G2Affine::zero(), g2gen ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "2") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "3") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "4") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "5") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "6") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "7") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "8") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "9") )
                ]

            ];
            // 3 x 3 transpose matrix
            let exp: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>( G2Affine::zero(), -g2gen ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "-2") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "-3") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "-4") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "-5") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "-6") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "-7") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "-8") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "-9") )
                ]
            ];
            let res: Matrix<Com2<F>> = mat.neg();

            assert_eq!(res.len(), 3);
            for i in 0..res.len() {
                assert_eq!(res[i].len(), 3);
            }

            assert_eq!(exp, res);
        }
        #[test]
        fn test_field_matrix_add() {

            // 3 x 3 matrices
            let one = Fr::one();
            let lhs: Matrix<Fr> = vec![
                vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")],
                vec![field_new!(Fr, "4"), field_new!(Fr, "5"), field_new!(Fr, "6")],
                vec![field_new!(Fr, "7"), field_new!(Fr, "8"), field_new!(Fr, "9")]
            ];
            let rhs: Matrix<Fr> = vec![
                vec![field_new!(Fr, "10"), field_new!(Fr, "11"), field_new!(Fr, "12")],
                vec![field_new!(Fr, "13"), field_new!(Fr, "14"), field_new!(Fr, "15")],
                vec![field_new!(Fr, "16"), field_new!(Fr, "17"), field_new!(Fr, "18")]
            ];

            let exp: Matrix<Fr> = vec![
                vec![field_new!(Fr, "11"), field_new!(Fr, "13"), field_new!(Fr, "15")],
                vec![field_new!(Fr, "17"), field_new!(Fr, "19"), field_new!(Fr, "21")],
                vec![field_new!(Fr, "23"), field_new!(Fr, "25"), field_new!(Fr, "27")]
            ];
            let lr: Matrix<Fr> = lhs.add(&rhs);
            let rl: Matrix<Fr> = rhs.add(&lhs);

            assert_eq!(lr.len(), 3);
            for i in 0..lr.len() {
                assert_eq!(lr[i].len(), 3);
            }

            assert_eq!(exp, lr);
            assert_eq!(lr, rl);
        }

        #[test]
        fn test_na_field_matrix_add() {

            // 3 x 3 matrices
            let one = Fr::one();
//            let lhs: SMatrix<Fr, 3, 3> = vec![
//                vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")],
//               vec![field_new!(Fr, "4"), field_new!(Fr, "5"), field_new!(Fr, "6")],
//                vec![field_new!(Fr, "7"), field_new!(Fr, "8"), field_new!(Fr, "9")]
//            ];
            let rhs: Matrix<Fr> = vec![
                vec![field_new!(Fr, "10"), field_new!(Fr, "11"), field_new!(Fr, "12")],
                vec![field_new!(Fr, "13"), field_new!(Fr, "14"), field_new!(Fr, "15")],
                vec![field_new!(Fr, "16"), field_new!(Fr, "17"), field_new!(Fr, "18")]
            ];

            let exp: Matrix<Fr> = vec![
                vec![field_new!(Fr, "11"), field_new!(Fr, "13"), field_new!(Fr, "15")],
                vec![field_new!(Fr, "17"), field_new!(Fr, "19"), field_new!(Fr, "21")],
                vec![field_new!(Fr, "23"), field_new!(Fr, "25"), field_new!(Fr, "27")]
            ];
            let lr: Matrix<Fr> = lhs.add(&rhs);
            let rl: Matrix<Fr> = rhs.add(&lhs);

            assert_eq!(lr.len(), 3);
            for i in 0..lr.len() {
                assert_eq!(lr[i].len(), 3);
            }

            assert_eq!(exp, lr);
            assert_eq!(lr, rl);
        }

        #[test]
        fn test_B1_matrix_add() {

            // 3 x 3 matrices
            let mut rng = test_rng();
            let g1gen = G1Projective::rand(&mut rng).into_affine();
            let lhs: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>( G1Affine::zero(), g1gen ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "2") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "3") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "4") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "5") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "6") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "7") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "8") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "9") )
                ]
            ];
            let rhs: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "10") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "11") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "12") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "13") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "14") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "15") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "16") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "17") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "18") )
                ]
            ];

            let exp: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "11") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "13") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "15") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "17") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "19") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "21") )
                ],
                vec![
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "23") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "25") ),
                    Com1::<F>( G1Affine::zero(), affine_group_new!(g1gen, "27") )
                ]
            ];
            let lr: Matrix<Com1<F>> = lhs.add(&rhs);
            let rl: Matrix<Com1<F>> = rhs.add(&lhs);

            assert_eq!(lr.len(), 3);
            for i in 0..lr.len() {
                assert_eq!(lr[i].len(), 3);
            }

            assert_eq!(exp, lr);
            assert_eq!(lr, rl);
        }

        #[test]
        fn test_B2_matrix_add() {

            // 3 x 3 matrices
            let mut rng = test_rng();
            let g2gen = G2Projective::rand(&mut rng).into_affine();
            let lhs: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>( G2Affine::zero(), g2gen ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "2") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "3") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "4") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "5") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "6") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "7") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "8") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "9") )
                ]
            ];
            let rhs: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "10") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "11") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "12") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "13") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "14") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "15") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "16") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "17") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "18") )
                ]
            ];

            let exp: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "11") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "13") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "15") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "17") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "19") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "21") )
                ],
                vec![
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "23") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "25") ),
                    Com2::<F>( G2Affine::zero(), affine_group_new!(g2gen, "27") )
                ]
            ];
            let lr: Matrix<Com2<F>> = lhs.add(&rhs);
            let rl: Matrix<Com2<F>> = rhs.add(&lhs);

            assert_eq!(lr.len(), 3);
            for i in 0..lr.len() {
                assert_eq!(lr[i].len(), 3);
            }

            assert_eq!(exp, lr);
            assert_eq!(lr, rl);
        }
    }
*/
}
