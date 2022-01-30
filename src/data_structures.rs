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
use ark_std::{
    fmt::Debug,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
    iter::Sum
};
use rayon::prelude::*;

use crate::generator::CRS;

// NOTE: Include traits Add<Self>, Neg<<Output = Self>, Mul<Self>, MulAssign<Scalar>?
// (but there is no single standard Zero or One element for arbitrary-dimension matrices)
// (and since Vec is not in crate, can't implement Matrix with trait Mat)
//
// TODO: Include helpful functions for identity matrix, zero matrix, inverse (though probably not needed for GS)
pub trait Mat<Elem: Clone>:
    Eq
    + Clone
    + Debug
{
    type Other;

    fn add(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
    fn scalar_mul(&self, other: &Self::Other) -> Self;
    fn transpose(&self) -> Self;
    fn left_mul(&self, lhs: &Matrix<Self::Other>, is_parallel: bool) -> Self;
    fn right_mul(&self, rhs: &Matrix<Self::Other>, is_parallel: bool) -> Self;
}

pub type Matrix<E> = Vec<Vec<E>>;

/// Encapsulates arithmetic traits for Groth-Sahai's bilinear group for commitments.
pub trait B<E: PairingEngine>:
    Eq
    + Copy
    + Clone
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
// TODO: Convert as_* into Into for B1/B2/BT, and figure out if multiple From<> / Into<> is possible.
pub trait B1<E: PairingEngine>:
    B<E>
//    + MulAssign<E::Fr>
    + From<Matrix<E::G1Affine>>
{
    fn as_col_vec(&self) -> Matrix<E::G1Affine>;
    fn as_vec(&self) -> Vec<E::G1Affine>;
    /// The linear map from G1 to B1 for pairing-product and multi-scalar multiplication equations.
    fn linear_map(x: &E::G1Affine) -> Self;
    fn batch_linear_map(x_vec: &Vec<E::G1Affine>) -> Vec<Self>;
    /// The linear map from scalar field to B1 for multi-scalar multiplication and quadratic equations.
    fn scalar_linear_map(x: &E::Fr, key: &CRS<E>) -> Self;
    fn batch_scalar_linear_map(x_vec: &Vec<E::Fr>, key: &CRS<E>) -> Vec<Self>;

    fn scalar_mul(&self, other: &E::Fr) -> Self;
}

/// Provides linear maps and vector conversions for the extension of the GS commitment group.
pub trait B2<E: PairingEngine>:
    B<E>
//    + MulAssign<E::Fr>
    + From<Matrix<E::G2Affine>>
{
    fn as_col_vec(&self) -> Matrix<E::G2Affine>;
    fn as_vec(&self) -> Vec<E::G2Affine>;
    /// The linear map from G2 to B2 for pairing-product and multi-scalar multiplication equations.
    fn linear_map(y: &E::G2Affine) -> Self;
    fn batch_linear_map(y_vec: &Vec<E::G2Affine>) -> Vec<Self>;
    /// The linear map from scalar field to B2 for multi-scalar multiplication and quadratic equations.
    fn scalar_linear_map(y: &E::Fr, key: &CRS<E>) -> Self;
    fn batch_scalar_linear_map(y_vec: &Vec<E::Fr>, key: &CRS<E>) -> Vec<Self>;

    fn scalar_mul(&self, other: &E::Fr) -> Self;
}

// TODO: GROTH-SAHAI -- Implement linear map for quadratic equations if needed
// TODO: Use linear map with GSType match instead, with ad-hoc polymorphism
// TODO: Allow to multiply with base element (E::Fqk)?
/// Provides linear maps and matrix conversions for the target of the GS commitment group, as well as the equipped pairing.
pub trait BT<E: PairingEngine, C1: B1<E>, C2: B2<E>>:
    B<E>    
    + From<Matrix<E::Fqk>>
{
    fn as_matrix(&self) -> Matrix<E::Fqk>;
    
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
    fn linear_map_MSG1(z: &E::G1Affine, key: &CRS<E>) -> Self;
    /// The linear map from G2 to BT for multi-scalar multiplication equations.
    #[allow(non_snake_case)]
    fn linear_map_MSG2(z: &E::G2Affine, key: &CRS<E>) -> Self;
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

// TODO: Move to matrix trait impl (Into vec?)
/// Collapse matrix into a single vector.
pub fn col_vec_to_vec<F: Clone>(mat: &Matrix<F>) -> Vec<F> {

    // TODO: OPTIMIZATION -- rewrite this function
    if mat.len() == 1 {
        let mut res = Vec::with_capacity(mat[0].len());
        for elem in mat[0].iter() {
            res.push(elem.clone());
        }
        res
    }
    else {
        let mut res = Vec::with_capacity(mat.len());
        for i in 0..mat.len() {
            assert_eq!(mat[i].len(), 1);
            res.push(mat[i][0].clone());
        }
        res
    }
}

// TODO: Move to matrix trait impl (From vec?)
/// Expand vector into column vector (in matrix form).
pub fn vec_to_col_vec<F: Clone>(vec: &Vec<F>) -> Matrix<F> {

    let mut mat = Vec::with_capacity(vec.len());
    for elem in vec.iter() {
        mat.push(vec![elem.clone()]);
    }
    mat
}

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
        )*
    }
}
impl_base_commit_groups!(Com1, Com2);

// TODO: Figure out how to include G1Affine / G2Affine in macro as match for Com1 / Com2, respectively

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

impl<E: PairingEngine> From<Matrix<E::G1Affine>> for Com1<E> {
    fn from(mat: Matrix<E::G1Affine>) -> Self {
        assert_eq!(mat.len(), 2);
        assert_eq!(mat[0].len(), 1);
        assert_eq!(mat[1].len(), 1);
        Self (
            mat[0][0],
            mat[1][0]
        )
    }
}
impl<E: PairingEngine> From<Matrix<E::G2Affine>> for Com2<E> {
    fn from(mat: Matrix<E::G2Affine>) -> Self {
        assert_eq!(mat.len(), 2);
        assert_eq!(mat[0].len(), 1);
        assert_eq!(mat[1].len(), 1);
        Self (
            mat[0][0],
            mat[1][0]
        )
    }
}

// TODO: Parallelize batched linear maps for B1/B2

impl<E: PairingEngine> B1<E> for Com1<E> {
    fn as_col_vec(&self) -> Matrix<E::G1Affine> {
        vec![ vec![self.0], vec![self.1] ]
    }

    fn as_vec(&self) -> Vec<E::G1Affine> {
        vec![ self.0, self.1 ]
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
        ( key.u[1] + Com1::<E>::linear_map(&key.g1_gen) ).scalar_mul(&x)
    }

    #[inline]
    fn batch_scalar_linear_map(x_vec: &Vec<E::Fr>, key: &CRS<E>) -> Vec<Self> {
        x_vec
            .into_iter()
            .map( |elem| Self::scalar_linear_map(&elem, key))
            .collect::<Vec<Self>>()
    }

    fn scalar_mul(&self, rhs: &E::Fr) -> Self {

        let mut s1p = self.0.clone().into_projective();
        let mut s2p = self.1.clone().into_projective();
        s1p *= *rhs;
        s2p *= *rhs;
        Self (
            s1p.into_affine().clone(),
            s2p.into_affine().clone()
        )
    }
}

impl<E: PairingEngine> B2<E> for Com2<E> {
    fn as_col_vec(&self) -> Matrix<E::G2Affine> {
        vec![ vec![self.0], vec![self.1] ]
    }

    fn as_vec(&self) -> Vec<E::G2Affine> {
        vec![ self.0, self.1 ]
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
        ( key.v[1] + Com2::<E>::linear_map(&key.g2_gen) ).scalar_mul(&y)
    }

    #[inline]
    fn batch_scalar_linear_map(y_vec: &Vec<E::Fr>, key: &CRS<E>) -> Vec<Self> {
        y_vec
            .into_iter()
            .map( |elem| Self::scalar_linear_map(&elem, key))
            .collect::<Vec<Self>>()
    }

    fn scalar_mul(&self, rhs: &E::Fr) -> Self{

        let mut s1p = self.0.clone().into_projective();
        let mut s2p = self.1.clone().into_projective();
        s1p *= *rhs;
        s2p *= *rhs;
        Self (
            s1p.into_affine().clone(),
            s2p.into_affine().clone()
        )
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
impl<E: PairingEngine> From<Matrix<E::Fqk>> for ComT<E> {
    fn from(mat: Matrix<E::Fqk>) -> Self {
        assert_eq!(mat.len(), 2);
        assert_eq!(mat[0].len(), 2);
        assert_eq!(mat[1].len(), 2);
        Self (
            mat[0][0],
            mat[0][1],
            mat[1][0],
            mat[1][1]
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

impl<E: PairingEngine> B<E> for ComT<E> {}
impl<E: PairingEngine> BT<E, Com1<E>, Com2<E>> for ComT<E> {
    #[inline]
    fn pairing(x: Com1<E>, y: Com2<E>) -> ComT<E> {
        ComT::<E>(
            // TODO: OPTIMIZATION ? -- If either element is 0 (G1 / G2), just output 1 (Fqk)
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

    fn as_matrix(&self) -> Matrix<E::Fqk> {
        vec![
            vec![ self.0, self.1 ],
            vec![ self.2, self.3 ]
        ]
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
    fn linear_map_MSG1(z: &E::G1Affine, key: &CRS<E>) -> Self {
        Self::pairing(Com1::<E>::linear_map(z), Com2::<E>::scalar_linear_map(&E::Fr::one(), key))
    }

    #[inline]
    fn linear_map_MSG2(z: &E::G2Affine, key: &CRS<E>) -> Self {
        Self::pairing(Com1::<E>::scalar_linear_map(&E::Fr::one(), key), Com2::<E>::linear_map(z))
    }
}

// TODO: Clean up the option to specify parallel computation
// TODO: Convert all for loops to iter / par_iter

// Matrix multiplication algorithm based on source: https://boydjohnson.dev/blog/concurrency-matrix-multiplication/
// TODO: OPTIMIZATION -- Use more efficient matrix multiplication algorithm than naive

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

                /* TODO: Paralellize scalar_mul */
                fn scalar_mul(&self, other: &Self::Other) -> Self {
                    let m = self.len();
                    let n = self[0].len();
                    let mut smul: Matrix<$com<E>> = Vec::with_capacity(m);
                    for i in 0..m {
                        smul.push(Vec::with_capacity(n));
                        for j in 0..n {
                            smul[i].push(self[i][j].scalar_mul(&other));
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
                                        (j, (0..dim).map( |k| row[k].scalar_mul(&rhs[k][j])).sum())
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
                                        (0..dim).map( |k| row[k].scalar_mul(&rhs[k][j]) ).sum()
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
                                        (j, (0..dim).map( |k| self[k][j].scalar_mul(&row[k]) ).sum())
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
                                        (0..dim).map( |k| self[k][j].scalar_mul(&row[k]) ).sum()
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

// TODO: combine with commitment matrix definition (i.e. if Com1, Com2 can be represented by
// fields, with multiplication wrt its scalar field. Or match/switch for minor differences in impl)
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


#[cfg(test)]
mod tests {

    #![allow(non_snake_case)]
    mod SXDH_com_group {

        use ark_bls12_381::{Bls12_381 as F};
        use ark_ff::UniformRand;
        use ark_ec::ProjectiveCurve;
        use ark_std::test_rng;

        use crate::data_structures::*;

        type G1Affine = <F as PairingEngine>::G1Affine;
        type G1Projective = <F as PairingEngine>::G1Projective;
        type G2Affine = <F as PairingEngine>::G2Affine;
        type G2Projective = <F as PairingEngine>::G2Projective;
        type Fqk = <F as PairingEngine>::Fqk;
        type Fr = <F as PairingEngine>::Fr;

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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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


        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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
            let bres = b.scalar_mul(&scalar);
            let bexp = Com1::<F>(b0.into_affine(), b1.into_affine());

            assert_eq!(bres, bexp);
        }

        #[allow(non_snake_case)]
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
            let bres = b.scalar_mul(&scalar);
            let bexp = Com1::<F>(b0.into_affine(), b1.into_affine());

            assert_eq!(bres, bexp);
        }

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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

        #[allow(non_snake_case)]
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
        fn test_B_into_matrix() {

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
            assert_eq!(b1.as_col_vec(), vec![vec![b1.0], vec![b1.1]]);
            assert_eq!(b2.as_col_vec(), vec![vec![b2.0], vec![b2.1]]);
            // BT can be represented as a 2 x 2 matrix
            assert_eq!(bt.as_matrix(), vec![vec![bt.0, bt.1], vec![bt.2, bt.3]]);
        }

        #[test]
        fn test_B_from_matrix() {

            let mut rng = test_rng();
            let b1_vec = vec![
                vec![G1Projective::rand(&mut rng).into_affine()],
                vec![G1Projective::rand(&mut rng).into_affine()]
            ];

            let b2_vec = vec![
                vec![G2Projective::rand(&mut rng).into_affine()],
                vec![G2Projective::rand(&mut rng).into_affine()]
            ];
            let bt_vec = vec![
                vec![
                    F::pairing::<G1Affine, G2Affine>(b1_vec[0][0].clone(), b2_vec[0][0].clone()),
                    F::pairing::<G1Affine, G2Affine>(b1_vec[0][0].clone(), b2_vec[1][0].clone()),
                ],
                vec![
                    F::pairing::<G1Affine, G2Affine>(b1_vec[1][0].clone(), b2_vec[0][0].clone()),
                    F::pairing::<G1Affine, G2Affine>(b1_vec[1][0].clone(), b2_vec[1][0].clone())
                ]
            ];

            let b1 = Com1::<F>::from(b1_vec.clone());
            let b2 = Com2::<F>::from(b2_vec.clone());
            let bt = ComT::<F>::from(bt_vec.clone());

            assert_eq!(b1.0, b1_vec[0][0]);
            assert_eq!(b1.1, b1_vec[1][0]);
            assert_eq!(b2.0, b2_vec[0][0]);
            assert_eq!(b2.1, b2_vec[1][0]);
            assert_eq!(bt.0, bt_vec[0][0]);
            assert_eq!(bt.1, bt_vec[0][1]);
            assert_eq!(bt.2, bt_vec[1][0]);
            assert_eq!(bt.3, bt_vec[1][1]);
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
            let bt = ComT::<F>::linear_map_MSG1(&at, &key);

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
            let bt = ComT::<F>::linear_map_MSG2(&at, &key);

            assert_eq!(b1.0, key.u[1].0.mul(a1));
            assert_eq!(b1.1, (key.u[1].1 + key.g1_gen).mul(a1));
            assert_eq!(b2.0, G2Affine::zero());
            assert_eq!(b2.1, a2);
            assert_eq!(bt.0, Fqk::one());
            assert_eq!(bt.1, F::pairing(key.u[1].0.clone(), at.clone()));
            assert_eq!(bt.2, Fqk::one());
            assert_eq!(bt.3, F::pairing(key.u[1].1.clone() + key.g1_gen.clone(), at.clone()));
        }
    }

    mod matrix {

        use ark_bls12_381::{Bls12_381 as F};
        use ark_ff::{UniformRand, field_new};
        use ark_ec::ProjectiveCurve;
        use ark_std::test_rng;

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
        fn test_col_vec_to_vec() {
            let mat = vec![vec![field_new!(Fr, "1")], vec![field_new!(Fr, "2")], vec![field_new!(Fr, "3")]];
            let vec: Vec<Fr> = col_vec_to_vec(&mat);
            let exp = vec![field_new!(Fr, "1"), field_new!(Fr, "2"), field_new!(Fr, "3")];
            assert_eq!(vec, exp);
        }

        #[test]
        fn test_vec_to_col_vec() {
            let vec = vec![field_new!(Fr, "1"), field_new!(Fr, "2"), field_new!(Fr, "3")];
            let mat: Matrix<Fr> = vec_to_col_vec(&vec);
            let exp = vec![vec![field_new!(Fr, "1")], vec![field_new!(Fr, "2")], vec![field_new!(Fr, "3")]];
            assert_eq!(mat, exp);
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
}
