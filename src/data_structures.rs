//! Contains the abstract Groth-Sahai commitment group `(B1, B2, BT)`, its concrete representation
//! within the SXDH instantiation `(Com1, Com2, ComT)`, as well as an implementation of matrix
//! arithmetic.
//!
//! # Properties
//!
//! In a Type-III pairing setting, the Groth-Sahai instantiation requires the SXDH assumption,
//! implementing the commitment group using elements of the bilinear group over an elliptic curve.
//! [`Com1`](crate::data_structures::Com1) and [`Com2`](crate::data_structures::Com2) are represented by 2 x 1 vectors of elements
//! in the corresponding groups [`G1Affine`](ark_ec::Pairing::G1Affine) and [`G2Affine`](ark_ec::Pairing::G2Affine).
//! [`ComT`](crate::data_structures::ComT) represents a 2 x 2 matrix of elements in [`GT`](ark_ec::PairingOutput).
//!
//! All of `Com1`, `Com2`, `ComT` are expressed as follows:
//! * Addition is defined by entry-wise addition of elements in `G1Affine`, `G2Affine`:
//!     * The equality is equality of all elements
//!     * The zero element is the zero vector
//!     * The negation is the additive inverse (i.e. negation) of all elements
//!
//! The Groth-Sahai proof system uses matrices of commitment group elements in its computations as
//! well.

use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr, CurveGroup,
};
use ark_ff::{Field, One, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};
use rayon::prelude::*;

use crate::generator::CRS;

pub trait Mat<Elem: Clone>: Eq + Clone + Debug {
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
pub trait B<E: Pairing>:
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
{
}

/// Provides linear maps and vector conversions for the base of the GS commitment group.
pub trait B1<E: Pairing>:
    B<E>
//    + MulAssign<E::ScalarField>
    + From<Matrix<E::G1Affine>>
{
    fn as_col_vec(&self) -> Matrix<E::G1Affine>;
    fn as_vec(&self) -> Vec<E::G1Affine>;
    /// The linear map from G1 to B1 for pairing-product and multi-scalar multiplication equations.
    fn linear_map(x: &E::G1Affine) -> Self;
    fn batch_linear_map(x_vec: &[E::G1Affine]) -> Vec<Self>;
    /// The linear map from scalar field to B1 for multi-scalar multiplication and quadratic equations.
    fn scalar_linear_map(x: &E::ScalarField, key: &CRS<E>) -> Self;
    fn batch_scalar_linear_map(x_vec: &[E::ScalarField], key: &CRS<E>) -> Vec<Self>;

    fn scalar_mul(&self, other: &E::ScalarField) -> Self;
}

/// Provides linear maps and vector conversions for the extension of the GS commitment group.
pub trait B2<E: Pairing>:
    B<E>
//    + MulAssign<E::ScalarField>
    + From<Matrix<E::G2Affine>>
{
    fn as_col_vec(&self) -> Matrix<E::G2Affine>;
    fn as_vec(&self) -> Vec<E::G2Affine>;
    /// The linear map from G2 to B2 for pairing-product and multi-scalar multiplication equations.
    fn linear_map(y: &E::G2Affine) -> Self;
    fn batch_linear_map(y_vec: &[E::G2Affine]) -> Vec<Self>;
    /// The linear map from scalar field to B2 for multi-scalar multiplication and quadratic equations.
    fn scalar_linear_map(y: &E::ScalarField, key: &CRS<E>) -> Self;
    fn batch_scalar_linear_map(y_vec: &[E::ScalarField], key: &CRS<E>) -> Vec<Self>;

    fn scalar_mul(&self, other: &E::ScalarField) -> Self;
}

/// Provides linear maps and matrix conversions for the target of the GS commitment group, as well as the equipped pairing.
pub trait BT<E: Pairing, C1: B1<E>, C2: B2<E>>: B<E> + From<Matrix<PairingOutput<E>>> {
    fn as_matrix(&self) -> Matrix<PairingOutput<E>>;

    /// The bilinear pairing over the GS commitment group (B1, B2, BT) is the tensor product.
    /// with respect to the bilinear pairing over the bilinear group (G1, G2, GT).
    fn pairing(x: C1, y: C2) -> Self;
    /// The entry-wise sum of bilinear pairings over the GS commitment group.
    fn pairing_sum(x_vec: &[C1], y_vec: &[C2]) -> Self;

    /// The linear map from GT to BT for pairing-sum equations.
    #[allow(non_snake_case)]
    fn linear_map_PPE(z: &PairingOutput<E>) -> Self;
    /// The linear map from G1 to BT for multi-scalar multiplication equations.
    #[allow(non_snake_case)]
    fn linear_map_MSMEG1(z: &E::G1Affine, key: &CRS<E>) -> Self;
    /// The linear map from G2 to BT for multi-scalar multiplication equations.
    #[allow(non_snake_case)]
    fn linear_map_MSMEG2(z: &E::G2Affine, key: &CRS<E>) -> Self;
    /// The linear map from Fr to BT for quadratic equations.
    fn linear_map_quad(z: &E::ScalarField, key: &CRS<E>) -> Self;
}

// SXDH instantiation's bilinear group for commitments

/// Base [`B1`](crate::data_structures::B1) for the commitment group in the SXDH instantiation.
#[derive(Copy, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Com1<E: Pairing>(pub E::G1Affine, pub E::G1Affine);

/// Extension [`B2`](crate::data_structures::B2) for the commitment group in the SXDH instantiation.
#[derive(Copy, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Com2<E: Pairing>(pub E::G2Affine, pub E::G2Affine);

/// Target [`BT`](crate::data_structures::BT) for the commitment group in the SXDH instantiation.
#[derive(Copy, Clone, Debug)]
pub struct ComT<E: Pairing>(
    pub PairingOutput<E>,
    pub PairingOutput<E>,
    pub PairingOutput<E>,
    pub PairingOutput<E>,
);

/// Collapse matrix into a single vector.
pub fn col_vec_to_vec<F: Clone>(mat: &Matrix<F>) -> Vec<F> {
    if mat.len() == 1 {
        mat[0].clone()
    } else {
        mat.iter().map(|v| v[0].clone()).collect()
    }
}

/// Expand vector into column vector (in matrix form).
pub fn vec_to_col_vec<F: Clone>(vec: &[F]) -> Matrix<F> {
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
            impl<E: Pairing> PartialEq for $com<E> {

                #[inline]
                fn eq(&self, other: &Self) -> bool {
                    self.0 == other.0 && self.1 == other.1
                }
            }
            impl<E: Pairing> Eq for $com<E> {}

            // Addition for Com group
            impl<E: Pairing> Add<$com<E>> for $com<E> {
                type Output = Self;

                #[inline]
                fn add(self, other: Self) -> Self {
                    Self (
                        (self.0 + other.0).into(),
                        (self.1 + other.1).into()
                    )
                }
            }
            impl<E: Pairing> AddAssign<$com<E>> for $com<E> {

                #[inline]
                fn add_assign(&mut self, other: Self) {
                    *self = Self (
                        (self.0 + other.0).into(),
                        (self.1 + other.1).into()
                    );
                }
            }
            impl<E: Pairing> Neg for $com<E> {
                type Output = Self;

                #[inline]
                fn neg(self) -> Self::Output {
                    Self (
                        (-(self.0.into_group())).into(),
                        (-(self.1.into_group())).into()
                    )
                }
            }
            impl<E: Pairing> Sub<$com<E>> for $com<E> {
                type Output = Self;

                #[inline]
                fn sub(self, other: Self) -> Self {
                    self + -other
                }
            }
            impl<E: Pairing> SubAssign<$com<E>> for $com<E> {

                #[inline]
                fn sub_assign(&mut self, other: Self) {
                    *self += -other;
                }
            }
            /*
            // Entry-wise scalar point-multiplication
            impl <E: Pairing> MulAssign<E::ScalarField> for $com<E> {
                fn mul_assign(&mut self, rhs: E::ScalarField) {

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
            impl<E: Pairing> Sum for $com<E> {
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

impl<E: Pairing> Zero for Com1<E> {
    #[inline]
    fn zero() -> Self {
        Self(E::G1Affine::zero(), E::G1Affine::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}
impl<E: Pairing> Zero for Com2<E> {
    #[inline]
    fn zero() -> Self {
        Self(E::G2Affine::zero(), E::G2Affine::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<E: Pairing> B<E> for Com1<E> {}
impl<E: Pairing> B<E> for Com2<E> {}

impl<E: Pairing> From<Matrix<E::G1Affine>> for Com1<E> {
    fn from(mat: Matrix<E::G1Affine>) -> Self {
        assert_eq!(mat.len(), 2);
        assert_eq!(mat[0].len(), 1);
        assert_eq!(mat[1].len(), 1);
        Self(mat[0][0], mat[1][0])
    }
}
impl<E: Pairing> From<Matrix<E::G2Affine>> for Com2<E> {
    fn from(mat: Matrix<E::G2Affine>) -> Self {
        assert_eq!(mat.len(), 2);
        assert_eq!(mat[0].len(), 1);
        assert_eq!(mat[1].len(), 1);
        Self(mat[0][0], mat[1][0])
    }
}

impl<E: Pairing> B1<E> for Com1<E> {
    fn as_col_vec(&self) -> Matrix<E::G1Affine> {
        vec![vec![self.0], vec![self.1]]
    }

    fn as_vec(&self) -> Vec<E::G1Affine> {
        vec![self.0, self.1]
    }

    #[inline]
    fn linear_map(x: &E::G1Affine) -> Self {
        Self(E::G1Affine::zero(), *x)
    }

    #[inline]
    fn batch_linear_map(x_vec: &[E::G1Affine]) -> Vec<Self> {
        x_vec
            .iter()
            .map(|elem| Self::linear_map(elem))
            .collect::<Vec<Self>>()
    }

    #[inline]
    fn scalar_linear_map(x: &E::ScalarField, key: &CRS<E>) -> Self {
        // = xu, where u = u_2 + (O, P) is a commitment group element
        (key.u[1] + Com1::<E>::linear_map(&key.g1_gen)).scalar_mul(x)
    }

    #[inline]
    fn batch_scalar_linear_map(x_vec: &[E::ScalarField], key: &CRS<E>) -> Vec<Self> {
        x_vec
            .iter()
            .map(|elem| Self::scalar_linear_map(elem, key))
            .collect::<Vec<Self>>()
    }

    fn scalar_mul(&self, rhs: &E::ScalarField) -> Self {
        let mut s1p = self.0.into_group();
        let mut s2p = self.1.into_group();
        s1p *= *rhs;
        s2p *= *rhs;
        Self(s1p.into_affine(), s2p.into_affine())
    }
}

impl<E: Pairing> B2<E> for Com2<E> {
    fn as_col_vec(&self) -> Matrix<E::G2Affine> {
        vec![vec![self.0], vec![self.1]]
    }

    fn as_vec(&self) -> Vec<E::G2Affine> {
        vec![self.0, self.1]
    }

    #[inline]
    fn linear_map(y: &E::G2Affine) -> Self {
        Self(E::G2Affine::zero(), *y)
    }

    #[inline]
    fn batch_linear_map(y_vec: &[E::G2Affine]) -> Vec<Self> {
        y_vec
            .iter()
            .map(|elem| Self::linear_map(elem))
            .collect::<Vec<Self>>()
    }

    #[inline]
    fn scalar_linear_map(y: &E::ScalarField, key: &CRS<E>) -> Self {
        // = yv, where v = v_2 + (O, P) is a commitment group element
        (key.v[1] + Com2::<E>::linear_map(&key.g2_gen)).scalar_mul(y)
    }

    #[inline]
    fn batch_scalar_linear_map(y_vec: &[E::ScalarField], key: &CRS<E>) -> Vec<Self> {
        y_vec
            .iter()
            .map(|elem| Self::scalar_linear_map(elem, key))
            .collect::<Vec<Self>>()
    }

    fn scalar_mul(&self, rhs: &E::ScalarField) -> Self {
        let mut s1p = self.0.into_group();
        let mut s2p = self.1.into_group();
        s1p *= *rhs;
        s2p *= *rhs;
        Self(s1p.into_affine(), s2p.into_affine())
    }
}

// ComT<Com1, Com2> is an instantiation of BT<B1, B2>
impl<E: Pairing> PartialEq for ComT<E> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1 && self.2 == other.2 && self.3 == other.3
    }
}
impl<E: Pairing> Eq for ComT<E> {}

impl<E: Pairing> Add<ComT<E>> for ComT<E> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self(
            self.0 + other.0,
            self.1 + other.1,
            self.2 + other.2,
            self.3 + other.3,
        )
    }
}
impl<E: Pairing> Zero for ComT<E> {
    #[inline]
    fn zero() -> Self {
        Self(
            PairingOutput::zero(),
            PairingOutput::zero(),
            PairingOutput::zero(),
            PairingOutput::zero(),
        )
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}
impl<E: Pairing> AddAssign<ComT<E>> for ComT<E> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
        self.1 += other.1;
        self.2 += other.2;
        self.3 += other.3;
    }
}
impl<E: Pairing> Neg for ComT<E> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1, -self.2, -self.3)
    }
}
impl<E: Pairing> Sub<ComT<E>> for ComT<E> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(
            self.0 - other.0,
            self.1 - other.1,
            self.2 - other.2,
            self.3 - other.3,
        )
    }
}
impl<E: Pairing> SubAssign<ComT<E>> for ComT<E> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
        self.1 -= other.1;
        self.2 -= other.2;
        self.3 -= other.3;
    }
}
impl<E: Pairing> From<Matrix<PairingOutput<E>>> for ComT<E> {
    fn from(mat: Matrix<PairingOutput<E>>) -> Self {
        assert_eq!(mat.len(), 2);
        assert_eq!(mat[0].len(), 2);
        assert_eq!(mat[1].len(), 2);
        Self(mat[0][0], mat[0][1], mat[1][0], mat[1][1])
    }
}
impl<E: Pairing> Sum for ComT<E> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl<E: Pairing> B<E> for ComT<E> {}
impl<E: Pairing> BT<E, Com1<E>, Com2<E>> for ComT<E> {
    #[inline]
    fn pairing(x: Com1<E>, y: Com2<E>) -> ComT<E> {
        ComT::<E>(
            E::pairing(x.0, y.0),
            E::pairing(x.0, y.1),
            E::pairing(x.1, y.0),
            E::pairing(x.1, y.1),
        )
    }

    #[inline]
    fn pairing_sum(x_vec: &[Com1<E>], y_vec: &[Com2<E>]) -> Self {
        assert_eq!(x_vec.len(), y_vec.len());
        Self(
            E::multi_pairing(x_vec.iter().map(|x| x.0), y_vec.iter().map(|y| y.0)),
            E::multi_pairing(x_vec.iter().map(|x| x.0), y_vec.iter().map(|y| y.1)),
            E::multi_pairing(x_vec.iter().map(|x| x.1), y_vec.iter().map(|y| y.0)),
            E::multi_pairing(x_vec.iter().map(|x| x.1), y_vec.iter().map(|y| y.1)),
        )
    }

    fn as_matrix(&self) -> Matrix<PairingOutput<E>> {
        vec![vec![self.0, self.1], vec![self.2, self.3]]
    }

    #[inline]
    fn linear_map_PPE(z: &PairingOutput<E>) -> Self {
        Self(
            PairingOutput::zero(),
            PairingOutput::zero(),
            PairingOutput::zero(),
            *z,
        )
    }

    #[inline]
    fn linear_map_MSMEG1(z: &E::G1Affine, key: &CRS<E>) -> Self {
        Self::pairing(
            Com1::<E>::linear_map(z),
            Com2::<E>::scalar_linear_map(&E::ScalarField::one(), key),
        )
    }

    #[inline]
    fn linear_map_MSMEG2(z: &E::G2Affine, key: &CRS<E>) -> Self {
        Self::pairing(
            Com1::<E>::scalar_linear_map(&E::ScalarField::one(), key),
            Com2::<E>::linear_map(z),
        )
    }

    #[inline]
    fn linear_map_quad(z: &E::ScalarField, key: &CRS<E>) -> Self {
        Self::pairing(
            Com1::<E>::scalar_linear_map(&E::ScalarField::one(), key),
            Com2::<E>::scalar_linear_map(&E::ScalarField::one(), key).scalar_mul(z),
        )
    }
}

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
            impl<E: Pairing> MulAssign<E::ScalarField> for Matrix<$com<E>> {
                fn mul_assign(&mut self, other: E::ScalarField) {
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
            impl<E: Pairing> Neg<Output = Self> for Matrix<$com<E>> {

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
            impl<E: Pairing> Mat<$com<E>> for Matrix<$com<E>> {
                type Other = E::ScalarField;

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
                    if self.is_empty() || self[0].is_empty() {
                        return vec![];
                    }
                    if rhs.is_empty() || rhs[0].is_empty() {
                        return vec![];
                    }

                    // Check that every row in a and column in b has the same length
                    assert_eq!(self[0].len(), rhs.len());
                    let row_dim = self.len();

                    if is_parallel {
                        let rows: Vec<_> = (0..row_dim)
                            .into_par_iter()
                            .map( |i| {
                                let row = &self[i];
                                let dim = rhs.len();

                                // Perform multiplication for single row
                                // Assuming every column in b has the same length
                                let cols: Vec<_> = (0..rhs[0].len())
                                    .into_par_iter()
                                    .map( |j| {
                                        (0..dim).map( |k| row[k].scalar_mul(&rhs[k][j])).sum()
                                    })
                                    .collect();

                                cols
                            })
                            .collect();

                        rows
                    } else {
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
                    if lhs.is_empty() || lhs[0].is_empty() {
                        return vec![];
                    }
                    if self.is_empty() || self[0].is_empty() {
                        return vec![];
                    }

                    // Check that every row in a and column in b has the same length
                    assert_eq!(lhs[0].len(), self.len());
                    let row_dim = lhs.len();

                    if is_parallel {
                        let rows: Vec<_> = (0..row_dim)
                            .into_par_iter()
                            .map( |i| {
                                let row = &lhs[i];
                                let dim = self.len();

                                // Perform matrix multiplication for single row
                                let cols: Vec<_> = (0..self[0].len())
                                    .into_par_iter()
                                    .map(|j| {
                                        (0..dim).map( |k| self[k][j].scalar_mul(&row[k])).sum()
                                    })
                                    .collect();

                                cols
                            })
                            .collect();

                        rows
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
                add[i].push(self[i][j] + other[i][j]);
            }
        }
        add
    }

    #[inline]
    fn neg(&self) -> Self {
        (0..self.len())
            .map(|i| {
                let row = &self[i];
                (0..row.len()).map(|j| -row[j]).collect::<Vec<F>>()
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
                trans[i].push(row[i]);
            }
        }
        trans
    }

    fn right_mul(&self, rhs: &Matrix<Self::Other>, is_parallel: bool) -> Self {
        if self.is_empty() || self[0].is_empty() {
            return vec![];
        }
        if rhs.is_empty() || rhs[0].is_empty() {
            return vec![];
        }

        // Check that every row in a and column in b has the same length
        assert_eq!(self[0].len(), rhs.len());
        let row_dim = self.len();

        if is_parallel {
            let rows: Vec<_> = (0..row_dim)
                .into_par_iter()
                .map(|i| {
                    let row = &self[i];
                    let dim = rhs.len();

                    // Perform multiplication for single row
                    // Assuming every column in b has the same length
                    let cols: Vec<_> = (0..rhs[0].len())
                        .into_par_iter()
                        .map(|j| (0..dim).map(|k| row[k] * rhs[k][j]).sum())
                        .collect();

                    cols
                })
                .collect();

            rows
        } else {
            (0..row_dim)
                .map(|i| {
                    let row = &self[i];
                    let dim = rhs.len();

                    // Perform matrix multiplication for single row
                    // Assuming every column in b has the same length
                    (0..rhs[0].len())
                        .map(|j| (0..dim).map(|k| row[k] * rhs[k][j]).sum())
                        .collect::<Vec<F>>()
                })
                .collect::<Vec<Vec<F>>>()
        }
    }

    fn left_mul(&self, lhs: &Matrix<Self::Other>, is_parallel: bool) -> Self {
        if lhs.is_empty() || lhs[0].is_empty() {
            return vec![];
        }
        if self.is_empty() || self[0].is_empty() {
            return vec![];
        }

        // Check that every row in a and column in b has the same length
        assert_eq!(lhs[0].len(), self.len());
        let row_dim = lhs.len();

        if is_parallel {
            let rows: Vec<_> = (0..row_dim)
                .into_par_iter()
                .map(|i| {
                    let row = &lhs[i];
                    let dim = self.len();

                    // Perform matrix multiplication for single row
                    let cols: Vec<_> = (0..self[0].len())
                        .into_par_iter()
                        .map(|j| (0..dim).map(|k| self[k][j] * row[k]).sum())
                        .collect();

                    cols
                })
                .collect();

            rows
        } else {
            (0..row_dim)
                .map(|i| {
                    let row = &lhs[i];
                    let dim = self.len();
                    (0..self[0].len())
                        .map(|j| (0..dim).map(|k| self[k][j] * row[k]).sum())
                        .collect::<Vec<F>>()
                })
                .collect::<Vec<Vec<F>>>()
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;

    mod SXDH_com_group {

        use ark_bls12_381::Bls12_381 as F;
        use ark_ec::{
            pairing::{Pairing, PairingOutput},
            AffineRepr, CurveGroup,
        };
        use ark_ff::UniformRand;
        use ark_std::ops::Mul;
        use ark_std::test_rng;

        use crate::AbstractCrs;

        use super::*;

        type G1Affine = <F as Pairing>::G1Affine;
        type G1Projective = <F as Pairing>::G1;
        type G2Affine = <F as Pairing>::G2Affine;
        type G2Projective = <F as Pairing>::G2;
        type GT = PairingOutput<F>;
        type Fr = <F as Pairing>::ScalarField;

        #[test]
        fn test_B1_add_zero() {
            let mut rng = test_rng();
            let a = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
            );
            let zero = Com1::<F>(G1Affine::zero(), G1Affine::zero());
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
                G2Projective::rand(&mut rng).into_affine(),
            );
            let zero = Com2::<F>(G2Affine::zero(), G2Affine::zero());
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
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
            );
            let zero = ComT::<F>(GT::zero(), GT::zero(), GT::zero(), GT::zero());
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
                G1Projective::rand(&mut rng).into_affine(),
            );
            let b = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
            );
            let ab = a + b;
            let ba = b + a;

            assert_eq!(ab, Com1::<F>((a.0 + b.0).into(), (a.1 + b.1).into()));
            assert_eq!(ab, ba);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B2_add() {
            let mut rng = test_rng();
            let a = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            );
            let b = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            );
            let ab = a + b;
            let ba = b + a;

            assert_eq!(ab, Com2::<F>((a.0 + b.0).into(), (a.1 + b.1).into()));
            assert_eq!(ab, ba);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_BT_add() {
            let mut rng = test_rng();
            let a = ComT::<F>(
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
            );
            let b = ComT::<F>(
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
            );
            let ab = a + b;
            let ba = b + a;

            assert_eq!(ab, ComT::<F>(a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3));
            assert_eq!(ab, ba);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B1_sum() {
            let mut rng = test_rng();
            let a = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
            );
            let b = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
            );
            let c = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
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
                G2Projective::rand(&mut rng).into_affine(),
            );
            let b = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            );
            let c = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
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
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
            );
            let b = ComT::<F>(
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
            );
            let c = ComT::<F>(
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
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
                G1Projective::rand(&mut rng).into_affine(),
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
                G2Projective::rand(&mut rng).into_affine(),
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
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
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
                G1Projective::rand(&mut rng).into_affine(),
            );
            let b = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
            );
            let ab = a - b;
            let ba = b - a;

            assert_eq!(ab, Com1::<F>((a.0 + -b.0).into(), (a.1 + -b.1).into()));
            assert_eq!(ab, -ba);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B2_sub() {
            let mut rng = test_rng();
            let a = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            );
            let b = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            );
            let ab = a - b;
            let ba = b - a;

            assert_eq!(ab, Com2::<F>((a.0 + -b.0).into(), (a.1 + -b.1).into()));
            assert_eq!(ab, -ba);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_BT_sub() {
            let mut rng = test_rng();
            let a = ComT::<F>(
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
            );
            let b = ComT::<F>(
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
                GT::rand(&mut rng),
            );
            let ab = a - b;
            let ba = b - a;

            assert_eq!(ab, ComT::<F>(a.0 - b.0, a.1 - b.1, a.2 - b.2, a.3 - b.3));
            assert_eq!(ab, -ba);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B1_scalar_mul() {
            let mut rng = test_rng();
            let b = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
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
            let b = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            );
            let scalar = Fr::rand(&mut rng);
            let b0 = b.0.mul(scalar);
            let b1 = b.1.mul(scalar);
            let bres = b.scalar_mul(&scalar);
            let bexp = Com2::<F>(b0.into_affine(), b1.into_affine());

            assert_eq!(bres, bexp);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B1_serde() {
            let mut rng = test_rng();
            let a = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
            );

            // Serialize and deserialize Com1.

            let mut c_bytes = Vec::new();
            a.serialize_compressed(&mut c_bytes).unwrap();
            let a_de = Com1::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
            assert_eq!(a, a_de);

            let mut u_bytes = Vec::new();
            a.serialize_uncompressed(&mut u_bytes).unwrap();
            let a_de = Com1::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
            assert_eq!(a, a_de);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B2_serde() {
            let mut rng = test_rng();
            let a = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            );

            // Serialize and deserialize Com2.

            let mut c_bytes = Vec::new();
            a.serialize_compressed(&mut c_bytes).unwrap();
            let a_de = Com2::<F>::deserialize_compressed(&c_bytes[..]).unwrap();
            assert_eq!(a, a_de);

            let mut u_bytes = Vec::new();
            a.serialize_uncompressed(&mut u_bytes).unwrap();
            let a_de = Com2::<F>::deserialize_uncompressed(&u_bytes[..]).unwrap();
            assert_eq!(a, a_de);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B_pairing_zero_G1() {
            let mut rng = test_rng();
            let b1 = Com1::<F>(G1Affine::zero(), G1Affine::zero());
            let b2 = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            );
            let bt = ComT::pairing(b1, b2);

            assert_eq!(bt.0, GT::zero());
            assert_eq!(bt.1, GT::zero());
            assert_eq!(bt.2, GT::zero());
            assert_eq!(bt.3, GT::zero());
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B_pairing_zero_G2() {
            let mut rng = test_rng();
            let b1 = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
            );
            let b2 = Com2::<F>(G2Affine::zero(), G2Affine::zero());
            let bt = ComT::pairing(b1, b2);

            assert_eq!(bt.0, GT::zero());
            assert_eq!(bt.1, GT::zero());
            assert_eq!(bt.2, GT::zero());
            assert_eq!(bt.3, GT::zero());
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B_pairing_commit() {
            let mut rng = test_rng();
            let b1 = Com1::<F>(G1Affine::zero(), G1Projective::rand(&mut rng).into_affine());
            let b2 = Com2::<F>(G2Affine::zero(), G2Projective::rand(&mut rng).into_affine());
            let bt = ComT::pairing(b1, b2);

            assert_eq!(bt.0, GT::zero());
            assert_eq!(bt.1, GT::zero());
            assert_eq!(bt.2, GT::zero());
            assert_eq!(bt.3, F::pairing(b1.1, b2.1));
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B_pairing_rand() {
            let mut rng = test_rng();
            let b1 = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
            );
            let b2 = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            );
            let bt = ComT::pairing(b1, b2);

            assert_eq!(bt.0, F::pairing(b1.0, b2.0));
            assert_eq!(bt.1, F::pairing(b1.0, b2.1));
            assert_eq!(bt.2, F::pairing(b1.1, b2.0));
            assert_eq!(bt.3, F::pairing(b1.1, b2.1));
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
            let exp: ComT<F> = vec![ComT::<F>::pairing(x1, y1), ComT::<F>::pairing(x2, y2)]
                .into_iter()
                .sum();
            let res: ComT<F> = ComT::<F>::pairing_sum(&x, &y);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_B_into_matrix() {
            let mut rng = test_rng();
            let b1 = Com1::<F>(
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
            );
            let b2 = Com2::<F>(
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            );
            let bt = ComT::pairing(b1, b2);

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
                vec![G1Projective::rand(&mut rng).into_affine()],
            ];

            let b2_vec = vec![
                vec![G2Projective::rand(&mut rng).into_affine()],
                vec![G2Projective::rand(&mut rng).into_affine()],
            ];
            let bt_vec = vec![
                vec![
                    F::pairing(b1_vec[0][0], b2_vec[0][0]),
                    F::pairing(b1_vec[0][0], b2_vec[1][0]),
                ],
                vec![
                    F::pairing(b1_vec[1][0], b2_vec[0][0]),
                    F::pairing(b1_vec[1][0], b2_vec[1][0]),
                ],
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
            let vec_g1 = vec![
                G1Projective::rand(&mut rng).into_affine(),
                G1Projective::rand(&mut rng).into_affine(),
            ];
            let vec_g2 = vec![
                G2Projective::rand(&mut rng).into_affine(),
                G2Projective::rand(&mut rng).into_affine(),
            ];
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

            assert_eq!(
                vec_b1[0],
                Com1::<F>::scalar_linear_map(&vec_scalar[0], &key)
            );
            assert_eq!(
                vec_b1[1],
                Com1::<F>::scalar_linear_map(&vec_scalar[1], &key)
            );
            assert_eq!(
                vec_b2[0],
                Com2::<F>::scalar_linear_map(&vec_scalar[0], &key)
            );
            assert_eq!(
                vec_b2[1],
                Com2::<F>::scalar_linear_map(&vec_scalar[1], &key)
            );
        }

        #[test]
        fn test_PPE_linear_maps() {
            let mut rng = test_rng();
            let a1 = G1Projective::rand(&mut rng).into_affine();
            let a2 = G2Projective::rand(&mut rng).into_affine();
            let at = F::pairing(a1, a2);
            let b1 = Com1::<F>::linear_map(&a1);
            let b2 = Com2::<F>::linear_map(&a2);
            let bt = ComT::<F>::linear_map_PPE(&at);

            assert_eq!(b1.0, G1Affine::zero());
            assert_eq!(b1.1, a1);
            assert_eq!(b2.0, G2Affine::zero());
            assert_eq!(b2.1, a2);
            assert_eq!(bt.0, GT::zero());
            assert_eq!(bt.1, GT::zero());
            assert_eq!(bt.2, GT::zero());
            assert_eq!(bt.3, F::pairing(a1, a2));
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
            assert_eq!(bt.0, GT::zero());
            assert_eq!(bt.1, GT::zero());
            assert_eq!(bt.2, F::pairing(at, key.v[1].0));
            assert_eq!(bt.3, F::pairing(at, key.v[1].1 + key.g2_gen));
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
            assert_eq!(bt.0, GT::zero());
            assert_eq!(bt.1, F::pairing(key.u[1].0, at));
            assert_eq!(bt.2, GT::zero());
            assert_eq!(bt.3, F::pairing(key.u[1].1 + key.g1_gen, at));
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
            let W1 = Com1::<F>(key.u[1].0, (key.u[1].1 + key.g1_gen).into());
            let W2 = Com2::<F>(key.v[1].0, (key.v[1].1 + key.g2_gen).into());
            assert_eq!(b1.0, W1.0.mul(a1));
            assert_eq!(b1.1, W1.1.mul(a1));
            assert_eq!(b2.0, W2.0.mul(a2));
            assert_eq!(b2.1, W2.1.mul(a2));
            assert_eq!(
                bt,
                ComT::<F>::pairing(W1.scalar_mul(&a1), W2.scalar_mul(&a2))
            );
            assert_eq!(bt, ComT::<F>::pairing(W1, W2.scalar_mul(&at)));
        }
    }

    mod matrix {

        use ark_bls12_381::Bls12_381 as F;
        use ark_ec::pairing::Pairing;
        use ark_ff::UniformRand;
        use ark_std::ops::Mul;
        use ark_std::str::FromStr;
        use ark_std::test_rng;

        use super::*;

        type G1Affine = <F as Pairing>::G1Affine;
        type G1Projective = <F as Pairing>::G1;
        type G2Affine = <F as Pairing>::G2Affine;
        type G2Projective = <F as Pairing>::G2;
        type Fr = <F as Pairing>::ScalarField;

        // Uses an affine group generator to produce an affine group element represented by the numeric string.
        #[allow(unused_macros)]
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

        macro_rules! assert_matrix_dimensions {
            ($matrix:ident, $rows:expr, $cols:expr) => {
                assert_eq!($matrix.len(), $rows);
                $matrix.iter().for_each(|r| assert_eq!(r.len(), $cols));
            };
        }

        #[test]
        fn test_col_vec_to_vec() {
            let mat = vec![
                vec![Fr::from_str("1").unwrap()],
                vec![Fr::from_str("2").unwrap()],
                vec![Fr::from_str("3").unwrap()],
            ];
            let vec: Vec<Fr> = col_vec_to_vec(&mat);
            let exp = vec![
                Fr::from_str("1").unwrap(),
                Fr::from_str("2").unwrap(),
                Fr::from_str("3").unwrap(),
            ];
            assert_eq!(vec, exp);
        }

        #[test]
        fn test_vec_to_col_vec() {
            let vec = vec![
                Fr::from_str("1").unwrap(),
                Fr::from_str("2").unwrap(),
                Fr::from_str("3").unwrap(),
            ];
            let mat: Matrix<Fr> = vec_to_col_vec(&vec);
            let exp = vec![
                vec![Fr::from_str("1").unwrap()],
                vec![Fr::from_str("2").unwrap()],
                vec![Fr::from_str("3").unwrap()],
            ];
            assert_eq!(mat, exp);
        }

        #[test]
        fn test_field_matrix_left_mul_entry() {
            // 1 x 3 (row) vector
            let one = Fr::one();
            let lhs: Matrix<Fr> = vec![vec![
                one,
                Fr::from_str("2").unwrap(),
                Fr::from_str("3").unwrap(),
            ]];
            // 3 x 1 (column) vector
            let rhs: Matrix<Fr> = vec![
                vec![Fr::from_str("4").unwrap()],
                vec![Fr::from_str("5").unwrap()],
                vec![Fr::from_str("6").unwrap()],
            ];
            let exp: Matrix<Fr> = vec![vec![Fr::from_str("32").unwrap()]];
            let res: Matrix<Fr> = rhs.left_mul(&lhs, false);

            // 1 x 1 resulting matrix
            assert_matrix_dimensions!(res, 1, 1);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_right_mul_entry() {
            // 1 x 3 (row) vector
            let one = Fr::one();
            let lhs: Matrix<Fr> = vec![vec![
                one,
                Fr::from_str("2").unwrap(),
                Fr::from_str("3").unwrap(),
            ]];
            // 3 x 1 (column) vector
            let rhs: Matrix<Fr> = vec![
                vec![Fr::from_str("4").unwrap()],
                vec![Fr::from_str("5").unwrap()],
                vec![Fr::from_str("6").unwrap()],
            ];
            let exp: Matrix<Fr> = vec![vec![Fr::from_str("32").unwrap()]];
            let res: Matrix<Fr> = lhs.right_mul(&rhs, false);

            // 1 x 1 resulting matrix
            assert_matrix_dimensions!(res, 1, 1);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_left_mul() {
            // 2 x 3 matrix
            let one = Fr::one();
            let lhs: Matrix<Fr> = vec![
                vec![one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
                vec![
                    Fr::from_str("4").unwrap(),
                    Fr::from_str("5").unwrap(),
                    Fr::from_str("6").unwrap(),
                ],
            ];
            // 3 x 4 matrix
            let rhs: Matrix<Fr> = vec![
                vec![
                    Fr::from_str("7").unwrap(),
                    Fr::from_str("8").unwrap(),
                    Fr::from_str("9").unwrap(),
                    Fr::from_str("10").unwrap(),
                ],
                vec![
                    Fr::from_str("11").unwrap(),
                    Fr::from_str("12").unwrap(),
                    Fr::from_str("13").unwrap(),
                    Fr::from_str("14").unwrap(),
                ],
                vec![
                    Fr::from_str("15").unwrap(),
                    Fr::from_str("16").unwrap(),
                    Fr::from_str("17").unwrap(),
                    Fr::from_str("18").unwrap(),
                ],
            ];
            let exp: Matrix<Fr> = vec![
                vec![
                    Fr::from_str("74").unwrap(),
                    Fr::from_str("80").unwrap(),
                    Fr::from_str("86").unwrap(),
                    Fr::from_str("92").unwrap(),
                ],
                vec![
                    Fr::from_str("173").unwrap(),
                    Fr::from_str("188").unwrap(),
                    Fr::from_str("203").unwrap(),
                    Fr::from_str("218").unwrap(),
                ],
            ];
            let res: Matrix<Fr> = rhs.left_mul(&lhs, false);

            // 2 x 4 resulting matrix
            assert_matrix_dimensions!(res, 2, 4);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_right_mul() {
            // 2 x 3 matrix
            let one = Fr::one();
            let lhs: Matrix<Fr> = vec![
                vec![one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
                vec![
                    Fr::from_str("4").unwrap(),
                    Fr::from_str("5").unwrap(),
                    Fr::from_str("6").unwrap(),
                ],
            ];
            // 3 x 4 matrix
            let rhs: Matrix<Fr> = vec![
                vec![
                    Fr::from_str("7").unwrap(),
                    Fr::from_str("8").unwrap(),
                    Fr::from_str("9").unwrap(),
                    Fr::from_str("10").unwrap(),
                ],
                vec![
                    Fr::from_str("11").unwrap(),
                    Fr::from_str("12").unwrap(),
                    Fr::from_str("13").unwrap(),
                    Fr::from_str("14").unwrap(),
                ],
                vec![
                    Fr::from_str("15").unwrap(),
                    Fr::from_str("16").unwrap(),
                    Fr::from_str("17").unwrap(),
                    Fr::from_str("18").unwrap(),
                ],
            ];
            let exp: Matrix<Fr> = vec![
                vec![
                    Fr::from_str("74").unwrap(),
                    Fr::from_str("80").unwrap(),
                    Fr::from_str("86").unwrap(),
                    Fr::from_str("92").unwrap(),
                ],
                vec![
                    Fr::from_str("173").unwrap(),
                    Fr::from_str("188").unwrap(),
                    Fr::from_str("203").unwrap(),
                    Fr::from_str("218").unwrap(),
                ],
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
                vec![one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
                vec![
                    Fr::from_str("4").unwrap(),
                    Fr::from_str("5").unwrap(),
                    Fr::from_str("6").unwrap(),
                ],
            ];
            // 3 x 4 matrix
            let rhs: Matrix<Fr> = vec![
                vec![
                    Fr::from_str("7").unwrap(),
                    Fr::from_str("8").unwrap(),
                    Fr::from_str("9").unwrap(),
                    Fr::from_str("10").unwrap(),
                ],
                vec![
                    Fr::from_str("11").unwrap(),
                    Fr::from_str("12").unwrap(),
                    Fr::from_str("13").unwrap(),
                    Fr::from_str("14").unwrap(),
                ],
                vec![
                    Fr::from_str("15").unwrap(),
                    Fr::from_str("16").unwrap(),
                    Fr::from_str("17").unwrap(),
                    Fr::from_str("18").unwrap(),
                ],
            ];
            let exp: Matrix<Fr> = vec![
                vec![
                    Fr::from_str("74").unwrap(),
                    Fr::from_str("80").unwrap(),
                    Fr::from_str("86").unwrap(),
                    Fr::from_str("92").unwrap(),
                ],
                vec![
                    Fr::from_str("173").unwrap(),
                    Fr::from_str("188").unwrap(),
                    Fr::from_str("203").unwrap(),
                    Fr::from_str("218").unwrap(),
                ],
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
                vec![one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
                vec![
                    Fr::from_str("4").unwrap(),
                    Fr::from_str("5").unwrap(),
                    Fr::from_str("6").unwrap(),
                ],
            ];
            // 3 x 4 matrix
            let rhs: Matrix<Fr> = vec![
                vec![
                    Fr::from_str("7").unwrap(),
                    Fr::from_str("8").unwrap(),
                    Fr::from_str("9").unwrap(),
                    Fr::from_str("10").unwrap(),
                ],
                vec![
                    Fr::from_str("11").unwrap(),
                    Fr::from_str("12").unwrap(),
                    Fr::from_str("13").unwrap(),
                    Fr::from_str("14").unwrap(),
                ],
                vec![
                    Fr::from_str("15").unwrap(),
                    Fr::from_str("16").unwrap(),
                    Fr::from_str("17").unwrap(),
                    Fr::from_str("18").unwrap(),
                ],
            ];
            let exp: Matrix<Fr> = vec![
                vec![
                    Fr::from_str("74").unwrap(),
                    Fr::from_str("80").unwrap(),
                    Fr::from_str("86").unwrap(),
                    Fr::from_str("92").unwrap(),
                ],
                vec![
                    Fr::from_str("173").unwrap(),
                    Fr::from_str("188").unwrap(),
                    Fr::from_str("203").unwrap(),
                    Fr::from_str("218").unwrap(),
                ],
            ];
            let res: Matrix<Fr> = lhs.right_mul(&rhs, true);

            // 2 x 4 resulting matrix
            assert_matrix_dimensions!(res, 2, 4);

            assert_eq!(exp, res);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B1_matrix_left_mul_entry() {
            // 1 x 3 (row) vector
            let one = Fr::one();
            let mut rng = test_rng();
            let g1gen = G1Projective::rand(&mut rng).into_affine();

            let lhs: Matrix<Fr> = vec![vec![
                one,
                Fr::from_str("2").unwrap(),
                Fr::from_str("3").unwrap(),
            ]];
            // 3 x 1 (column) vector
            let rhs: Matrix<Com1<F>> = vec![
                vec![Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "4"))],
                vec![Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "5"))],
                vec![Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "6"))],
            ];
            let exp: Matrix<Com1<F>> = vec![vec![Com1::<F>(
                G1Affine::zero(),
                affine_group_new!(g1gen, "32"),
            )]];
            let res: Matrix<Com1<F>> = rhs.left_mul(&lhs, false);

            // 1 x 1 resulting matrix
            assert_matrix_dimensions!(res, 1, 1);

            assert_eq!(exp, res);
        }

        #[allow(non_snake_case)]
        #[test]
        fn test_B1_matrix_right_mul_entry() {
            // 1 x 3 (row) vector
            let mut rng = test_rng();
            let g1gen = G1Projective::rand(&mut rng).into_affine();
            let lhs: Matrix<Com1<F>> = vec![vec![
                Com1::<F>(G1Affine::zero(), g1gen),
                Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "2")),
                Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "3")),
            ]];
            // 3 x 1 (column) vector
            let rhs: Matrix<Fr> = vec![
                vec![Fr::from_str("4").unwrap()],
                vec![Fr::from_str("5").unwrap()],
                vec![Fr::from_str("6").unwrap()],
            ];
            let exp: Matrix<Com1<F>> = vec![vec![Com1::<F>(
                G1Affine::zero(),
                affine_group_new!(g1gen, "32"),
            )]];
            let res: Matrix<Com1<F>> = lhs.right_mul(&rhs, false);

            assert_matrix_dimensions!(res, 1, 1);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_scalar_mul() {
            // 3 x 3 matrices
            let one = Fr::one();
            let scalar: Fr = Fr::from_str("3").unwrap();
            let mat: Matrix<Fr> = vec![
                vec![one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
                vec![
                    Fr::from_str("4").unwrap(),
                    Fr::from_str("5").unwrap(),
                    Fr::from_str("6").unwrap(),
                ],
                vec![
                    Fr::from_str("7").unwrap(),
                    Fr::from_str("8").unwrap(),
                    Fr::from_str("9").unwrap(),
                ],
            ];

            let exp: Matrix<Fr> = vec![
                vec![
                    Fr::from_str("3").unwrap(),
                    Fr::from_str("6").unwrap(),
                    Fr::from_str("9").unwrap(),
                ],
                vec![
                    Fr::from_str("12").unwrap(),
                    Fr::from_str("15").unwrap(),
                    Fr::from_str("18").unwrap(),
                ],
                vec![
                    Fr::from_str("21").unwrap(),
                    Fr::from_str("24").unwrap(),
                    Fr::from_str("27").unwrap(),
                ],
            ];
            let res: Matrix<Fr> = mat.scalar_mul(&scalar);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_B1_matrix_scalar_mul() {
            let scalar: Fr = Fr::from_str("3").unwrap();

            // 3 x 3 matrix of Com1 elements (0, 3)
            let mut rng = test_rng();
            let g1gen = G1Projective::rand(&mut rng).into_affine();
            let mut mat: Matrix<Com1<F>> = Vec::with_capacity(3);

            for i in 0..3 {
                mat.push(Vec::with_capacity(3));
                for _ in 0..3 {
                    mat[i].push(Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "1")));
                }
            }

            let mut exp: Matrix<Com1<F>> = Vec::with_capacity(3);
            for i in 0..3 {
                exp.push(Vec::with_capacity(3));
                for _ in 0..3 {
                    exp[i].push(Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "3")));
                }
            }

            let res: Matrix<Com1<F>> = mat.scalar_mul(&scalar);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_B2_matrix_scalar_mul() {
            let scalar: Fr = Fr::from_str("3").unwrap();

            // 3 x 3 matrix of Com1 elements (0, 3)
            let mut rng = test_rng();
            let g2gen = G2Projective::rand(&mut rng).into_affine();
            let mut mat: Matrix<Com2<F>> = Vec::with_capacity(3);

            for i in 0..3 {
                mat.push(Vec::with_capacity(3));
                for _ in 0..3 {
                    mat[i].push(Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "1")));
                }
            }

            let mut exp: Matrix<Com2<F>> = Vec::with_capacity(3);
            for i in 0..3 {
                exp.push(Vec::with_capacity(3));
                for _ in 0..3 {
                    exp[i].push(Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "3")));
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
                Com1::<F>(G1Affine::zero(), g1gen),
                Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "2")),
                Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "3")),
            ]];
            // 3 x 1 transpose (column) vector
            let exp: Matrix<Com1<F>> = vec![
                vec![Com1::<F>(G1Affine::zero(), g1gen)],
                vec![Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "2"))],
                vec![Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "3"))],
            ];
            let res: Matrix<Com1<F>> = mat.transpose();

            assert_matrix_dimensions!(res, 3, 1);
            assert_eq!(exp, res);
        }

        #[test]
        fn test_B1_matrix_transpose() {
            // 3 x 3 matrix
            let mut rng = test_rng();
            let g1gen = G1Projective::rand(&mut rng).into_affine();
            let mat: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>(G1Affine::zero(), g1gen),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "2")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "3")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "4")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "5")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "6")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "7")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "8")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "9")),
                ],
            ];
            // 3 x 3 transpose matrix
            let exp: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>(G1Affine::zero(), g1gen),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "4")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "7")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "2")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "5")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "8")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "3")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "6")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "9")),
                ],
            ];
            let res: Matrix<Com1<F>> = mat.transpose();

            assert_matrix_dimensions!(res, 3, 3);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_B2_transpose_vec() {
            let mut rng = test_rng();
            let g2gen = G2Projective::rand(&mut rng).into_affine();
            // 1 x 3 (row) vector
            let mat: Matrix<Com2<F>> = vec![vec![
                Com2::<F>(G2Affine::zero(), g2gen),
                Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "2")),
                Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "3")),
            ]];
            // 3 x 1 transpose (column) vector
            let exp: Matrix<Com2<F>> = vec![
                vec![Com2::<F>(G2Affine::zero(), g2gen)],
                vec![Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "2"))],
                vec![Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "3"))],
            ];
            let res: Matrix<Com2<F>> = mat.transpose();

            assert_matrix_dimensions!(res, 3, 1);
            assert_eq!(exp, res);
        }

        #[test]
        fn test_B2_matrix_transpose() {
            // 3 x 3 matrix
            let mut rng = test_rng();
            let g2gen = G2Projective::rand(&mut rng).into_affine();
            let mat: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>(G2Affine::zero(), g2gen),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "2")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "3")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "4")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "5")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "6")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "7")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "8")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "9")),
                ],
            ];
            // 3 x 3 transpose matrix
            let exp: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>(G2Affine::zero(), g2gen),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "4")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "7")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "2")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "5")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "8")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "3")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "6")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "9")),
                ],
            ];
            let res: Matrix<Com2<F>> = mat.transpose();

            assert_matrix_dimensions!(res, 3, 3);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_transpose_vec() {
            // 1 x 3 (row) vector
            let one = Fr::one();
            let mat: Matrix<Fr> = vec![vec![
                one,
                Fr::from_str("2").unwrap(),
                Fr::from_str("3").unwrap(),
            ]];

            // 3 x 1 transpose (column) vector
            let exp: Matrix<Fr> = vec![
                vec![one],
                vec![Fr::from_str("2").unwrap()],
                vec![Fr::from_str("3").unwrap()],
            ];
            let res: Matrix<Fr> = mat.transpose();

            assert_matrix_dimensions!(res, 3, 1);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_transpose() {
            // 3 x 3 matrix
            let one = Fr::one();
            let mat: Matrix<Fr> = vec![
                vec![one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
                vec![
                    Fr::from_str("4").unwrap(),
                    Fr::from_str("5").unwrap(),
                    Fr::from_str("6").unwrap(),
                ],
                vec![
                    Fr::from_str("7").unwrap(),
                    Fr::from_str("8").unwrap(),
                    Fr::from_str("9").unwrap(),
                ],
            ];
            // 3 x 3 transpose matrix
            let exp: Matrix<Fr> = vec![
                vec![one, Fr::from_str("4").unwrap(), Fr::from_str("7").unwrap()],
                vec![
                    Fr::from_str("2").unwrap(),
                    Fr::from_str("5").unwrap(),
                    Fr::from_str("8").unwrap(),
                ],
                vec![
                    Fr::from_str("3").unwrap(),
                    Fr::from_str("6").unwrap(),
                    Fr::from_str("9").unwrap(),
                ],
            ];
            let res: Matrix<Fr> = mat.transpose();

            assert_matrix_dimensions!(res, 3, 3);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_field_matrix_neg() {
            // 3 x 3 matrix
            let one = Fr::one();
            let mat: Matrix<Fr> = vec![
                vec![one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
                vec![
                    Fr::from_str("4").unwrap(),
                    Fr::from_str("5").unwrap(),
                    Fr::from_str("6").unwrap(),
                ],
                vec![
                    Fr::from_str("7").unwrap(),
                    Fr::from_str("8").unwrap(),
                    Fr::from_str("9").unwrap(),
                ],
            ];
            // 3 x 3 transpose matrix
            let exp: Matrix<Fr> = vec![
                vec![
                    -one,
                    -Fr::from_str("2").unwrap(),
                    -Fr::from_str("3").unwrap(),
                ],
                vec![
                    -Fr::from_str("4").unwrap(),
                    -Fr::from_str("5").unwrap(),
                    -Fr::from_str("6").unwrap(),
                ],
                vec![
                    -Fr::from_str("7").unwrap(),
                    -Fr::from_str("8").unwrap(),
                    -Fr::from_str("9").unwrap(),
                ],
            ];
            let res: Matrix<Fr> = mat.neg();

            assert_matrix_dimensions!(res, 3, 3);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_B1_matrix_neg() {
            // 3 x 3 matrix
            let mut rng = test_rng();
            let g1gen = G1Projective::rand(&mut rng).into_affine();
            let mat: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>(G1Affine::zero(), g1gen),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "2")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "3")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "4")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "5")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "6")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "7")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "8")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "9")),
                ],
            ];
            // 3 x 3 transpose matrix
            let exp: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>(G1Affine::zero(), -g1gen),
                    Com1::<F>(G1Affine::zero(), -affine_group_new!(g1gen, "2")),
                    Com1::<F>(G1Affine::zero(), -affine_group_new!(g1gen, "3")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), -affine_group_new!(g1gen, "4")),
                    Com1::<F>(G1Affine::zero(), -affine_group_new!(g1gen, "5")),
                    Com1::<F>(G1Affine::zero(), -affine_group_new!(g1gen, "6")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), -affine_group_new!(g1gen, "7")),
                    Com1::<F>(G1Affine::zero(), -affine_group_new!(g1gen, "8")),
                    Com1::<F>(G1Affine::zero(), -affine_group_new!(g1gen, "9")),
                ],
            ];
            let res: Matrix<Com1<F>> = mat.neg();

            assert_matrix_dimensions!(res, 3, 3);

            assert_eq!(exp, res);
        }

        #[test]
        fn test_B2_matrix_neg() {
            // 3 x 3 matrix
            let mut rng = test_rng();
            let g2gen = G2Projective::rand(&mut rng).into_affine();
            let mat: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>(G2Affine::zero(), g2gen),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "2")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "3")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "4")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "5")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "6")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "7")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "8")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "9")),
                ],
            ];
            // 3 x 3 transpose matrix
            let exp: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>(G2Affine::zero(), -g2gen),
                    Com2::<F>(G2Affine::zero(), -affine_group_new!(g2gen, "2")),
                    Com2::<F>(G2Affine::zero(), -affine_group_new!(g2gen, "3")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), -affine_group_new!(g2gen, "4")),
                    Com2::<F>(G2Affine::zero(), -affine_group_new!(g2gen, "5")),
                    Com2::<F>(G2Affine::zero(), -affine_group_new!(g2gen, "6")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), -affine_group_new!(g2gen, "7")),
                    Com2::<F>(G2Affine::zero(), -affine_group_new!(g2gen, "8")),
                    Com2::<F>(G2Affine::zero(), -affine_group_new!(g2gen, "9")),
                ],
            ];
            let res: Matrix<Com2<F>> = mat.neg();

            assert_matrix_dimensions!(res, 3, 3);

            assert_eq!(exp, res);
        }
        #[test]
        fn test_field_matrix_add() {
            // 3 x 3 matrices
            let one = Fr::one();
            let lhs: Matrix<Fr> = vec![
                vec![one, Fr::from_str("2").unwrap(), Fr::from_str("3").unwrap()],
                vec![
                    Fr::from_str("4").unwrap(),
                    Fr::from_str("5").unwrap(),
                    Fr::from_str("6").unwrap(),
                ],
                vec![
                    Fr::from_str("7").unwrap(),
                    Fr::from_str("8").unwrap(),
                    Fr::from_str("9").unwrap(),
                ],
            ];
            let rhs: Matrix<Fr> = vec![
                vec![
                    Fr::from_str("10").unwrap(),
                    Fr::from_str("11").unwrap(),
                    Fr::from_str("12").unwrap(),
                ],
                vec![
                    Fr::from_str("13").unwrap(),
                    Fr::from_str("14").unwrap(),
                    Fr::from_str("15").unwrap(),
                ],
                vec![
                    Fr::from_str("16").unwrap(),
                    Fr::from_str("17").unwrap(),
                    Fr::from_str("18").unwrap(),
                ],
            ];

            let exp: Matrix<Fr> = vec![
                vec![
                    Fr::from_str("11").unwrap(),
                    Fr::from_str("13").unwrap(),
                    Fr::from_str("15").unwrap(),
                ],
                vec![
                    Fr::from_str("17").unwrap(),
                    Fr::from_str("19").unwrap(),
                    Fr::from_str("21").unwrap(),
                ],
                vec![
                    Fr::from_str("23").unwrap(),
                    Fr::from_str("25").unwrap(),
                    Fr::from_str("27").unwrap(),
                ],
            ];
            let lr: Matrix<Fr> = lhs.add(&rhs);
            let rl: Matrix<Fr> = rhs.add(&lhs);

            assert_matrix_dimensions!(lr, 3, 3);

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
                    Com1::<F>(G1Affine::zero(), g1gen),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "2")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "3")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "4")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "5")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "6")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "7")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "8")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "9")),
                ],
            ];
            let rhs: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "10")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "11")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "12")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "13")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "14")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "15")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "16")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "17")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "18")),
                ],
            ];

            let exp: Matrix<Com1<F>> = vec![
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "11")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "13")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "15")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "17")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "19")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "21")),
                ],
                vec![
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "23")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "25")),
                    Com1::<F>(G1Affine::zero(), affine_group_new!(g1gen, "27")),
                ],
            ];
            let lr: Matrix<Com1<F>> = lhs.add(&rhs);
            let rl: Matrix<Com1<F>> = rhs.add(&lhs);

            assert_matrix_dimensions!(lr, 3, 3);

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
                    Com2::<F>(G2Affine::zero(), g2gen),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "2")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "3")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "4")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "5")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "6")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "7")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "8")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "9")),
                ],
            ];
            let rhs: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "10")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "11")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "12")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "13")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "14")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "15")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "16")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "17")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "18")),
                ],
            ];

            let exp: Matrix<Com2<F>> = vec![
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "11")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "13")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "15")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "17")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "19")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "21")),
                ],
                vec![
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "23")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "25")),
                    Com2::<F>(G2Affine::zero(), affine_group_new!(g2gen, "27")),
                ],
            ];
            let lr: Matrix<Com2<F>> = lhs.add(&rhs);
            let rl: Matrix<Com2<F>> = rhs.add(&lhs);

            assert_matrix_dimensions!(lr, 3, 3);

            assert_eq!(exp, lr);
            assert_eq!(lr, rl);
        }
    }
}
