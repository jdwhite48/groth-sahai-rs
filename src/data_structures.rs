use ark_ec::{PairingEngine, AffineCurve, ProjectiveCurve};
use ark_ff::{Zero, One, Field};
use ark_std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign}
};
use rayon::prelude::*;

use crate::generator::CRS;

/// B1,B2,BT forms a bilinear group for GS commitments

// TODO: Parallelize batched linear maps for B1/B2
pub trait B<E: PairingEngine>:
    Eq
    + Clone
    + Debug
    + Zero
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Neg<Output = Self>
{}

pub trait B1<E: PairingEngine>:
    B<E>
    + From<Matrix<E::G1Affine>>
{
    fn as_col_vec(&self) -> Matrix<E::G1Affine>;
    fn as_vec(&self) -> Vec<E::G1Affine>;
    fn linear_map(x: &E::G1Affine) -> Self;
    fn batch_linear_map(x_vec: &Vec<E::G1Affine>) -> Vec<Self>;
    fn scalar_linear_map(x: &E::Fr, key: &CRS<E>) -> Self;
    fn batch_scalar_linear_map(x_vec: &Vec<E::Fr>, key: &CRS<E>) -> Vec<Self>;
}
pub trait B2<E: PairingEngine>:
    B<E>
    + From<Matrix<E::G2Affine>>
{
    fn as_col_vec(&self) -> Matrix<E::G2Affine>;
    fn as_vec(&self) -> Vec<E::G2Affine>;
    fn linear_map(y: &E::G2Affine) -> Self;
    fn batch_linear_map(y_vec: &Vec<E::G2Affine>) -> Vec<Self>;
    fn scalar_linear_map(y: &E::Fr, key: &CRS<E>) -> Self;
    fn batch_scalar_linear_map(y_vec: &Vec<E::Fr>, key: &CRS<E>) -> Vec<Self>;
}

// TODO: Implement linear map for quadratic equations if we really need it
pub trait BT<E: PairingEngine, C1: B1<E>, C2: B2<E>>:
    Eq
    + Clone
    + Debug
    + Zero
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Neg<Output = Self>
    + From<Matrix<E::Fqk>>
{
    fn as_matrix(&self) -> Matrix<E::Fqk>;
    fn pairing(x: C1, y: C2) -> Self;

    #[allow(non_snake_case)]
    fn linear_map_PPE(z: &E::Fqk) -> Self;
    #[allow(non_snake_case)]
    fn linear_map_MSG1(z: &E::G1Affine, key: &CRS<E>) -> Self;
    #[allow(non_snake_case)]
    fn linear_map_MSG2(z: &E::G2Affine, key: &CRS<E>) -> Self;
}

// SXDH instantiation's bilinear group for commitments

// TODO: Expose randomness? (see example data_structures in Arkworks)
#[derive(Clone, Debug)]
pub struct Com1<E: PairingEngine>(pub E::G1Affine, pub E::G1Affine);
#[derive(Clone, Debug)]
pub struct Com2<E: PairingEngine>(pub E::G2Affine, pub E::G2Affine);
#[derive(Clone, Debug)]
pub struct ComT<E: PairingEngine>(pub E::Fqk, pub E::Fqk, pub E::Fqk, pub E::Fqk);

// TODO: Refactor matrix code to use Matrix trait (cleaner)
// Would have to implement Matrix as a struct instead of pub type ... Vec<...> because "impl X for
// Vec<...> doesn't work; Vec defined outside of crate
/*
pub trait Matrix<Other = Self>:
    Eq
    + Debug
    + Zero
    + One
    + Add<Other, Output = Self>
    + Mul<Other, Output = Self>
    + AddAssign<Other>
    + MulAssign<Other>
{
    fn transpose(&mut self);
}
*/

/// Sparse representation of matrices
pub type Matrix<F> = Vec<Vec<F>>;

/// Collapse matrix into a single vector
pub fn col_vec_to_vec<F: Clone>(mat: &Matrix<F>) -> Vec<F> {

    if mat.len() == 1 {
        // TODO: OPTIMIZATION -- find mor efficient way of copying over vector
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

/// Expand vector into column vector (in matrix form)
pub fn vec_to_col_vec<F: Clone>(vec: &Vec<F>) -> Matrix<F> {

    let mut mat = Vec::with_capacity(vec.len());
    for elem in vec.iter() {
        mat.push(vec![elem.clone()]);
    }
    mat
}

// TODO: Combine this into a macro for Com1<E>: B1, Com2<E>: B2, ComT<E>: BT<B1,B2> (cleaner?)
/*
macro_rules! impl_Com {
    (for $($t:ty),+) => {
        $(impl<E: PairingEngine> PartialEq for Com$t<E> {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0 && self.1 == other.1
            }
        })*
        $(impl<E: PairingEngine> Eq for Com$t<E> {})*
        $(impl<E: PairingEngine> B$t for Com$t<E> {})*
    }
}
impl_Com!(for 1, 2);
*/

// Com1 implements B1
impl<E: PairingEngine> PartialEq for Com1<E> {

    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl<E: PairingEngine> Eq for Com1<E> {}

/// Addition for B1 is entry-wise addition of elements in G1
impl<E: PairingEngine> Add<Com1<E>> for Com1<E> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self (
            self.0 + other.0,
            self.1 + other.1
        )
    }
}
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
impl<E: PairingEngine> AddAssign<Com1<E>> for Com1<E> {

    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = Self (
            self.0 + other.0,
            self.1 + other.1
        );
    }
}
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
impl<E: PairingEngine> Neg for Com1<E> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self (
            -self.0,
            -self.1
        )
    }
}
impl<E: PairingEngine> Sub<Com1<E>> for Com1<E> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self (
            self.0 + -other.0,
            self.1 + -other.1
        )
    }
}
impl<E: PairingEngine> SubAssign<Com1<E>> for Com1<E> {

    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = Self (
            self.0 + -other.0,
            self.1 + -other.1
        );
    }
}

impl<E: PairingEngine> B<E> for Com1<E> {}
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
        Com1::<E>::from(group_matrix_scalar_mul::<E, E::G1Affine>(&x, &matrix_add(&key.u.1.as_col_vec(), &Com1::<E>::linear_map(&key.g1_gen).as_col_vec()) ))
    }

    #[inline]
    fn batch_scalar_linear_map(x_vec: &Vec<E::Fr>, key: &CRS<E>) -> Vec<Self> {
        x_vec
            .into_iter()
            .map( |elem| Self::scalar_linear_map(&elem, key))
            .collect::<Vec<Self>>()
    }
}


// Com2 implements B2
impl<E: PairingEngine> PartialEq for Com2<E> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl<E: PairingEngine> Eq for Com2<E> {}

/// Addition for B2 is entry-wise addition of elements in G2
impl<E: PairingEngine> Add<Com2<E>> for Com2<E> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self (
            self.0 + other.0,
            self.1 + other.1
        )
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
impl<E: PairingEngine> AddAssign<Com2<E>> for Com2<E> {

    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = Self (
            self.0 + other.0,
            self.1 + other.1
        );
    }
}
impl<E: PairingEngine> Neg for Com2<E> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self (
            -self.0,
            -self.1
        )
    }
}
impl<E: PairingEngine> Sub<Com2<E>> for Com2<E> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self (
            self.0 + -other.0,
            self.1 + -other.1
        )
    }
}
impl<E: PairingEngine> SubAssign<Com2<E>> for Com2<E> {

    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = Self (
            self.0 + -other.0,
            self.1 + -other.1
        );
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

impl<E: PairingEngine> B<E> for Com2<E> {}
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
        Com2::<E>::from(group_matrix_scalar_mul::<E, E::G2Affine>(&y, &matrix_add(&key.v.1.as_col_vec(), &Com2::<E>::linear_map(&key.g2_gen).as_col_vec()) ))
    }

    #[inline]
    fn batch_scalar_linear_map(y_vec: &Vec<E::Fr>, key: &CRS<E>) -> Vec<Self> {
        y_vec
            .into_iter()
            .map( |elem| Self::scalar_linear_map(&elem, key))
            .collect::<Vec<Self>>()
    }
}

// ComT implements BT<B1, B2>
impl<E: PairingEngine> PartialEq for ComT<E> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1 && self.2 == other.2 && self.3 == other.3
    }
}
impl<E: PairingEngine> Eq for ComT<E> {}

/// Addition for BT is entry-wise multiplication of elements in GT
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
            self.0 * (E::Fqk::one() / other.0),
            self.1 * (E::Fqk::one() / other.1),
            self.2 * (E::Fqk::one() / other.2),
            self.3 * (E::Fqk::one() / other.3)
        )
    }
}
impl<E: PairingEngine> SubAssign<ComT<E>> for ComT<E> {

    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = Self (
            self.0 * (E::Fqk::one() / other.0),
            self.1 * (E::Fqk::one() / other.1),
            self.2 * (E::Fqk::one() / other.2),
            self.3 * (E::Fqk::one() / other.3)
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

impl<E: PairingEngine> B<E> for ComT<E> {}
impl<E: PairingEngine> BT<E, Com1<E>, Com2<E>> for ComT<E> {
    #[inline]
    /// Commitment bilinear group pairing computes entry-wise pairing products
    fn pairing(x: Com1<E>, y: Com2<E>) -> ComT<E> {
        ComT::<E>(
            // TODO: OPTIMIZATION -- If either element is 0 (G1 / G2), just output 1 (Fqk)
            E::pairing::<E::G1Affine, E::G2Affine>(x.0.clone(), y.0.clone()),
            E::pairing::<E::G1Affine, E::G2Affine>(x.0.clone(), y.1.clone()),
            E::pairing::<E::G1Affine, E::G2Affine>(x.1.clone(), y.0.clone()),
            E::pairing::<E::G1Affine, E::G2Affine>(x.1.clone(), y.1.clone()),
        )
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
// TODO: Refactor to reuse code for both scalars and group elements, if possible

// Matrix multiplication algorithm based on source: https://boydjohnson.dev/blog/concurrency-matrix-multiplication/

/// Computes row of matrix corresponding to multiplication of field matrices
fn field_matrix_mul_row<F: Field>(row: &[F], rhs: &Matrix<F>, dim: usize, is_parallel: bool) -> Vec<F> {
    
    // Assuming every column in b has the same length
    let rhs_col_dim = rhs[0].len();
    
    if is_parallel {
        let mut cols = (0..rhs_col_dim)
            .into_par_iter()
            .map( |j| {
                (j, (0..dim).map( |k| row[k] * rhs[k][j] ).sum())
            })
            .collect::<Vec<(usize, F)>>();

        // After computing concurrently, sort by index
        cols.par_sort_by(|left, right| left.0.cmp(&right.0));

        // Strip off index and return Vec<F>
        cols.into_iter()
            .map( |(_, elem)| elem)
            .collect()
    }
    else {
        (0..rhs_col_dim)
            .map( |j| {
                (0..dim).map( |k| row[k] * rhs[k][j] ).sum()
            })
            .collect::<Vec<F>>()
    }
}

/// Computes row of matrix corresponding to multiplication of group matrix (G1 or G2) with scalar
/// matrix
fn group_left_matrix_mul_row<E: PairingEngine, G: AffineCurve>(row: &[G], rhs: &Matrix<G::ScalarField>, dim: usize, is_parallel: bool) -> Vec<G> {

    // Assuming every column in b has the same length
    let rhs_col_dim = rhs[0].len();

    if is_parallel {
        let mut cols = (0..rhs_col_dim)
            .into_par_iter()
            .map( |j| {
                (j, (0..dim).map( |k| row[k].mul(rhs[k][j]).into_affine() ).sum())
            })
            .collect::<Vec<(usize, G)>>();

        // After computing concurrently, sort by index
        cols.par_sort_by(|left, right| left.0.cmp(&right.0));

        // Strip off index and return Vec<F>
        cols.into_iter()
            .map( |(_, elem)| elem)
            .collect()
    }
    else {
        (0..rhs_col_dim)
            .map( |j| {
                (0..dim).map( |k| row[k].mul(rhs[k][j]).into_affine() ).sum()
            })
            .collect::<Vec<G>>()
    }
}

/// Computes row of matrix corresponding to multiplication of scalar matrix with group matrix (G1 or
/// G2)
fn group_right_matrix_mul_row<E: PairingEngine, G: AffineCurve>(row: &[G::ScalarField], rhs: &Matrix<G>, dim: usize, is_parallel: bool) -> Vec<G> {

    // Assuming every column in b has the same length
    let rhs_col_dim = rhs[0].len();

    if is_parallel {
        let mut cols = (0..rhs_col_dim)
            .into_par_iter()
            .map( |j| {
                (j, (0..dim).map( |k| rhs[k][j].mul(row[k]).into_affine() ).sum())
            })
            .collect::<Vec<(usize, G)>>();

        // After computing concurrently, sort by index
        cols.par_sort_by(|left, right| left.0.cmp(&right.0));

        // Strip off index and return Vec<F>
        cols.into_iter()
            .map( |(_, elem)| elem)
            .collect()
    }
    else {
        (0..rhs_col_dim)
            .map( |j| {
                (0..dim).map( |k| rhs[k][j].mul(row[k]).into_affine() ).sum()
            })
            .collect::<Vec<G>>()
    }
}

// TODO: Change all pub matrix functions to pub(crate)? (or else move benches inward)

// TODO: Move all the group matrix functions into B1/B2/BT (proof system operates on these, not group)
/// Matrix multiplication of field matrices (scalar/Fr or GT/Fqk)
pub fn field_matrix_mul<F: Field>(lhs: &Matrix<F>, rhs: &Matrix<F>, is_parallel: bool) -> Matrix<F> {
    if lhs.len() == 0 || lhs[0].len() == 0 {
        return vec![];
    }
    if rhs.len() == 0 || rhs[0].len() == 0 {
        return vec![];
    }

    // Assuming every row in a and column in b has the same length
    assert_eq!(lhs[0].len(), rhs.len());
    let row_dim = lhs.len();

    if is_parallel {
        let mut rows = (0..row_dim)
            .into_par_iter()
            .map( |i| {
                let row = &lhs[i];
                let dim = rhs.len();
                (i, field_matrix_mul_row::<F>(row, rhs, dim, is_parallel))
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
                let dim = rhs.len();
                field_matrix_mul_row::<F>(row, rhs, dim, is_parallel)
            })
            .collect::<Vec<Vec<F>>>()
    }
}

/// Computes multiplication of group matrix (G1 or G2) with scalar matrix
pub fn group_left_matrix_mul<E: PairingEngine, G: AffineCurve>(lhs: &Matrix<G>, rhs: &Matrix<G::ScalarField>, is_parallel: bool) -> Matrix<G> {
    if lhs.len() == 0 || lhs[0].len() == 0 {
        return vec![];
    }
    if rhs.len() == 0 || rhs[0].len() == 0 {
        return vec![];
    }

    // Assuming every row in a and column in b has the same length
    assert_eq!(lhs[0].len(), rhs.len());
    let row_dim = lhs.len();

    if is_parallel { 
        let mut rows = (0..row_dim)
            .into_par_iter()
            .map( |i| {
                let row = &lhs[i];
                let dim = rhs.len();
                (i, group_left_matrix_mul_row::<E,G>(row, rhs, dim, is_parallel))
            })
            .collect::<Vec<(usize, Vec<G>)>>();

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
                let dim = rhs.len();
                group_left_matrix_mul_row::<E,G>(row, rhs, dim, is_parallel)
            })
            .collect::<Vec<Vec<G>>>()
    }
}


/// Computes multiplication of scalar matrix with group matrix (G1 or G2)
pub fn group_right_matrix_mul<E: PairingEngine, G: AffineCurve>(lhs: &Matrix<G::ScalarField>, rhs: &Matrix<G>, is_parallel: bool) -> Matrix<G> {
    if lhs.len() == 0 || lhs[0].len() == 0 {
        return vec![];
    }
    if rhs.len() == 0 || rhs[0].len() == 0 {
        return vec![];
    }

    // Assuming every row in a and column in b has the same length
    assert_eq!(lhs[0].len(), rhs.len());
    let row_dim = lhs.len();

    if is_parallel { 
        let mut rows = (0..row_dim)
            .into_par_iter()
            .map( |i| {
                let row = &lhs[i];
                let dim = rhs.len();
                (i, group_right_matrix_mul_row::<E,G>(row, rhs, dim, is_parallel))
            })
            .collect::<Vec<(usize, Vec<G>)>>();

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
                let dim = rhs.len();
                group_right_matrix_mul_row::<E,G>(row, rhs, dim, is_parallel)
            })
            .collect::<Vec<Vec<G>>>()
    }
}

/// Computes multiplication of scalar with a field matrix (scalar/Fr or GT/Fqk)
pub fn field_matrix_scalar_mul<F: Field>(scalar: &F, mat: &Matrix<F>) -> Matrix<F> {
    let m = mat.len();
    let n = mat[0].len();
    let mut smul = Vec::with_capacity(m);
    for i in 0..m {
        smul.push(Vec::with_capacity(n));
        for j in 0..n {
            smul[i].push( *scalar * mat[i][j] );
        }
    }
    smul
}

/// Computes multiplication of scalar with a commitment group matrix (B1 or B2)
pub fn group_matrix_scalar_mul<E: PairingEngine, G: AffineCurve>(scalar: &G::ScalarField, mat: &Matrix<G>) -> Matrix<G> {
    let m = mat.len();
    let n = mat[0].len();
    let mut smul = Vec::with_capacity(m);
    for i in 0..m {
        smul.push(Vec::with_capacity(n));
        for j in 0..n {
            smul[i].push( mat[i][j].mul(*scalar).into_affine() );
        }
    }
    smul
}

/// Computes out-of-place transpose of a matrix
pub fn matrix_transpose<F: Clone>(mat: &Matrix<F>) -> Matrix<F> {
    let mut trans = Vec::with_capacity(mat[0].len());
    for _ in 0..mat[0].len() {
        trans.push(Vec::with_capacity(mat.len()));
    }

    for row in mat {
        for i in 0..row.len() {
            // Push rows onto columns
            trans[i].push(row[i].clone());
        }
    }
    trans
}

/// Computes matrix addition
pub fn matrix_add<F: Add<Output = F> + Clone>(lhs: &Matrix<F>, rhs: &Matrix<F>) -> Matrix<F> {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs[0].len(), rhs[0].len());
    let m = lhs.len();
    let n = lhs[0].len();
    let mut add = Vec::with_capacity(m);
    for i in 0..m {
        add.push(Vec::with_capacity(n));
        for j in 0..n {
            add[i].push(lhs[i][j].clone() + rhs[i][j].clone());
        }
    }
    add
}

#[cfg(test)]
mod tests {

    use ark_bls12_381::{Bls12_381 as F};
    use ark_ff::{UniformRand, field_new};
    use ark_ec::ProjectiveCurve;
    use ark_std::test_rng;

    use crate::data_structures::*;

    type G1Affine = <F as PairingEngine>::G1Affine;
    type G1Projective = <F as PairingEngine>::G1Projective;
    type G2Affine = <F as PairingEngine>::G2Affine;
    type G2Projective = <F as PairingEngine>::G2Projective;
    type GT = <F as PairingEngine>::Fqk;
    type Fqk = <F as PairingEngine>::Fqk;
    type Fr = <F as PairingEngine>::Fr;

    
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

        assert_eq!(bt.0, GT::one());
        assert_eq!(bt.1, GT::one());
        assert_eq!(bt.2, GT::one());
        assert_eq!(bt.3, GT::one());
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

        assert_eq!(bt.0, GT::one());
        assert_eq!(bt.1, GT::one());
        assert_eq!(bt.2, GT::one());
        assert_eq!(bt.3, GT::one());
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

        assert_eq!(bt.0, GT::one());
        assert_eq!(bt.1, GT::one());
        assert_eq!(bt.2, GT::one());
        assert_eq!(bt.3, F::pairing::<G1Affine, G2Affine>(b1.1.clone(), b2.1.clone()));
    }


    #[allow(non_snake_case)]
    #[test]
    fn test_B1_neg() {
        let mut rng = test_rng();
        let b = Com1::<F>(
            G1Projective::rand(&mut rng).into_affine(),
            G1Projective::rand(&mut rng).into_affine()
        );
        let bneg = -b.clone();
        let zero = b.clone() + bneg.clone();

        assert_eq!(zero, Com1::<F>::zero());
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
        let bneg = -b.clone();
        let zero = b.clone() + bneg.clone();

        assert_eq!(zero, Com2::<F>::zero());
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
        let bneg = -b.clone();
        let zero = b.clone() + bneg.clone();

        assert_eq!(zero, ComT::<F>::zero());
        assert!(zero.is_zero());
    }

    #[test]
    fn test_scalar_matrix_mul_row() {

        // 1 x 3 (row) vector
        let one = Fr::one();
        let lhs: Vec<Fr> = vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")];
        // 3 x 1 (column) vector
        let rhs: Matrix<Fr> = vec![
            vec![field_new!(Fr, "4")],
            vec![field_new!(Fr, "5")],
            vec![field_new!(Fr, "6")]
        ];
        let exp: Vec<Fr> = vec![field_new!(Fr, "32")];
        let res: Vec<Fr> = field_matrix_mul_row::<Fr>(&lhs, &rhs, 3, false);

        // 1 x 1 resulting matrix
        assert_eq!(res.len(), 1);
   
        assert_eq!(exp, res);
    }

    #[test]
    fn test_scalar_matrix_mul_entry() {

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
        let res: Matrix<Fr> = field_matrix_mul::<Fr>(&lhs, &rhs, false);

        // 1 x 1 resulting matrix
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].len(), 1);
   
        assert_eq!(exp, res);
    }


    #[test]
    fn test_scalar_matrix_mul() {

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
        let res: Matrix<Fr> = field_matrix_mul::<Fr>(&lhs, &rhs, false);

        // 2 x 4 resulting matrix
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].len(), 4);
        assert_eq!(res[1].len(), 4);

        assert_eq!(exp, res);
    }

    #[test]
    fn test_scalar_matrix_mul_rayon() {

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
        let res: Matrix<Fr> = field_matrix_mul::<Fr>(&lhs, &rhs, true);

        // 2 x 4 resulting matrix
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].len(), 4);
        assert_eq!(res[1].len(), 4);

        assert_eq!(exp, res);
    }


    #[test]
    fn test_group_left_matrix_mul_row() {

        // 1 x 3 (row) vector
        let mut rng = test_rng();
        let g1gen = G1Projective::rand(&mut rng).into_affine();
        let lhs: Vec<G1Affine> = vec![
            g1gen,
            g1gen.mul(field_new!(Fr, "2")).into_affine(),
            g1gen.mul(field_new!(Fr, "3")).into_affine()
        ];
        // 3 x 1 (column) vector
        let rhs: Matrix<Fr> = vec![
            vec![field_new!(Fr, "4")],
            vec![field_new!(Fr, "5")],
            vec![field_new!(Fr, "6")]
        ];
        let exp: Vec<G1Affine> = vec![g1gen.mul(field_new!(Fr, "32")).into_affine()];
        let res: Vec<G1Affine> = group_left_matrix_mul_row::<F,G1Affine>(&lhs, &rhs, 3, false);

        // 1 x 1 resulting matrix
        assert_eq!(res.len(), 1);
   
        assert_eq!(exp, res);
    }

    #[test]
    fn test_group_left_matrix_mul_entry() {

        // 1 x 3 (row) vector
        let mut rng = test_rng();
        let g1gen = G1Projective::rand(&mut rng).into_affine();
        let lhs: Matrix<G1Affine> = vec![vec![
            g1gen,
            g1gen.mul(field_new!(Fr, "2")).into_affine(),
            g1gen.mul(field_new!(Fr, "3")).into_affine()
        ]];
        // 3 x 1 (column) vector
        let rhs: Matrix<Fr> = vec![
            vec![field_new!(Fr, "4")],
            vec![field_new!(Fr, "5")],
            vec![field_new!(Fr, "6")]
        ];
        let exp: Matrix<G1Affine> = vec![vec![g1gen.mul(field_new!(Fr, "32")).into_affine()]];
        let res: Matrix<G1Affine> = group_left_matrix_mul::<F,G1Affine>(&lhs, &rhs, false);
        
        // 1 x 1 resulting matrix
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].len(), 1);
   
        assert_eq!(exp, res);
    }

    #[test]
    fn test_group_right_matrix_mul_row() {

        // 1 x 3 (row) vector
        let one = Fr::one();
        let mut rng = test_rng();
        let g1gen = G1Projective::rand(&mut rng).into_affine();
        let lhs: Vec<Fr> = vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")];
        // 3 x 1 (column) vector
        let rhs: Matrix<G1Affine> = vec![
            vec![g1gen.mul(field_new!(Fr, "4")).into_affine()],
            vec![g1gen.mul(field_new!(Fr, "5")).into_affine()],
            vec![g1gen.mul(field_new!(Fr, "6")).into_affine()]
        ];
        let exp: Vec<G1Affine> = vec![g1gen.mul(field_new!(Fr, "32")).into_affine()];
        let res: Vec<G1Affine> = group_right_matrix_mul_row::<F,G1Affine>(&lhs, &rhs, 3, false);

        // 1 x 1 resulting matrix
        assert_eq!(res.len(), 1);
   
        assert_eq!(exp, res);
    }

    #[test]
    fn test_group_right_matrix_mul_entry() {

        // 1 x 3 (row) vector
        let one = Fr::one();
        let mut rng = test_rng();
        let g1gen = G1Projective::rand(&mut rng).into_affine();
        
        let lhs: Matrix<Fr> = vec![vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")]];
        // 3 x 1 (column) vector
        let rhs: Matrix<G1Affine> = vec![
            vec![g1gen.mul(field_new!(Fr, "4")).into_affine()],
            vec![g1gen.mul(field_new!(Fr, "5")).into_affine()],
            vec![g1gen.mul(field_new!(Fr, "6")).into_affine()]
        ];
        let exp: Matrix<G1Affine> = vec![vec![g1gen.mul(field_new!(Fr, "32")).into_affine()]];
        let res: Matrix<G1Affine> = group_right_matrix_mul::<F,G1Affine>(&lhs, &rhs, false);
        
        // 1 x 1 resulting matrix
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].len(), 1);
   
        assert_eq!(exp, res);
    }

    #[test]
    fn test_matrix_transpose_vec() {

        // 1 x 3 (row) vector
        let one = Fr::one();
        let mat: Matrix<Fr> = vec![vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")]];

        // 3 x 1 transpose (column) vector
        let exp: Matrix<Fr> = vec![
            vec![one],
            vec![field_new!(Fr, "2")],
            vec![field_new!(Fr, "3")]
        ];
        let res: Matrix<Fr> = matrix_transpose::<Fr>(&mat);

        assert_eq!(res.len(), 3);
        for i in 0..res.len() {
            assert_eq!(res[i].len(), 1);
        }

        assert_eq!(exp, res);
    }

    #[test]
    fn test_matrix_transpose() {

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
        let res: Matrix<Fr> = matrix_transpose::<Fr>(&mat);

        assert_eq!(res.len(), 3);
        for i in 0..res.len() {
            assert_eq!(res[i].len(), 3);
        }

        assert_eq!(exp, res);
    }

    #[test]
    fn test_scalar_matrix_scalar_mul() {

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
        let res: Matrix<Fr> = field_matrix_scalar_mul::<Fr>(&scalar, &mat);

        assert_eq!(exp, res);
    }

    #[test]
    fn test_group_matrix_scalar_mul() {

        let scalar: Fr = field_new!(Fr, "3");

        // 3 x 3 matrix {{1,2,3}, {4,5,6}, {7,8,9}}
        let mut rng = test_rng();
        let g1gen = G1Projective::rand(&mut rng).into_affine();
        let mut mat: Matrix<G1Affine> = Vec::with_capacity(3);
        for i in 0..3 {
            mat.push(Vec::with_capacity(3));
            for _ in 0..3 {

                mat[i].push(g1gen.mul(field_new!(Fr,"3")).into_affine());
            }
        }

        let mut exp: Matrix<G1Affine> = Vec::with_capacity(3);
        for i in 0..3 {
            exp.push(Vec::with_capacity(3));
            for _ in 0..3 {
                exp[i].push(g1gen.mul(field_new!(Fr,"9")).into_affine());
            }
        }
        let res: Matrix<G1Affine> = group_matrix_scalar_mul::<F,G1Affine>(&scalar, &mat);

        assert_eq!(exp, res);
    }

    #[test]
    fn test_matrix_add() {

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
        let res: Matrix<Fr> = matrix_add::<Fr>(&lhs, &rhs);

        assert_eq!(res.len(), 3);
        for i in 0..res.len() {
            assert_eq!(res[i].len(), 3);
        }

        assert_eq!(exp, res);
    }


    #[test]
    fn test_matrix_add_commutativity() {

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
        let lhres: Matrix<Fr> = matrix_add::<Fr>(&lhs, &rhs);
        let hlres: Matrix<Fr> = matrix_add::<Fr>(&rhs, &lhs);

        assert_eq!(lhres, hlres);
    }

    #[test]
    fn test_into_vec_and_matrix() {

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
    fn test_from_vec_and_matrix() {

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

    #[allow(non_snake_case)]
    #[test]
    fn test_PPE_linear_maps() {

        let mut rng = test_rng();
        let g1 = G1Projective::rand(&mut rng).into_affine();
        let g2 = G2Projective::rand(&mut rng).into_affine();
        let b1 = Com1::<F>::linear_map(&g1);
        let b2 = Com2::<F>::linear_map(&g2);

        assert_eq!(b1.0, G1Affine::zero());
        assert_eq!(b1.1, g1);
        assert_eq!(b2.0, G2Affine::zero());
        assert_eq!(b2.1, g2);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_PPE_linear_bilinear_map_commutativity() {

        let mut rng = test_rng();
        let g1 = G1Projective::rand(&mut rng).into_affine();
        let g2 = G2Projective::rand(&mut rng).into_affine();
        let gt = F::pairing::<G1Affine, G2Affine>(g1.clone(), g2.clone());
        let b1 = Com1::<F>::linear_map(&g1);
        let b2 = Com2::<F>::linear_map(&g2);

        let bt_lin_bilin = ComT::<F>::pairing(b1.clone(), b2.clone());
        let bt_bilin_lin = ComT::<F>::linear_map_PPE(&gt);

        assert_eq!(bt_lin_bilin.0, Fqk::one());
        assert_eq!(bt_lin_bilin.1, Fqk::one());
        assert_eq!(bt_lin_bilin.2, Fqk::one());
        assert_eq!(bt_bilin_lin.0, Fqk::one());
        assert_eq!(bt_bilin_lin.1, Fqk::one());
        assert_eq!(bt_bilin_lin.2, Fqk::one());
        assert_eq!(bt_lin_bilin.3, bt_bilin_lin.3);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_MSG1_linear_maps() {

        let mut rng = test_rng();
        let key = CRS::<F>::generate_crs(&mut rng);

        let g1 = G1Projective::rand(&mut rng).into_affine();
        let g2 = Fr::rand(&mut rng);
        let b1 = Com1::<F>::linear_map(&g1);
        let b2 = Com2::<F>::scalar_linear_map(&g2, &key);

        assert_eq!(b1.0, G1Affine::zero());
        assert_eq!(b1.1, g1);
        assert_eq!(b2.0, key.v.1.0.mul(g2));
        assert_eq!(b2.1, (key.v.1.1 + key.g2_gen).mul(g2));
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_MSG1_linear_bilinear_map_commutativity() {

        let mut rng = test_rng();
        let key = CRS::<F>::generate_crs(&mut rng);

        let g1 = G1Projective::rand(&mut rng).into_affine();
        let g2 = Fr::rand(&mut rng);
        let gt = g1.mul(g2).into_affine();
        let b1 = Com1::<F>::linear_map(&g1);
        let b2 = Com2::<F>::scalar_linear_map(&g2, &key);

        let bt_lin_bilin = ComT::<F>::pairing(b1.clone(), b2.clone());
        let bt_bilin_lin = ComT::<F>::linear_map_MSG1(&gt, &key);

        assert_eq!(bt_lin_bilin.0, Fqk::one());
        assert_eq!(bt_lin_bilin.1, Fqk::one());
        assert_eq!(bt_lin_bilin.2, F::pairing(gt.clone(), key.v.1.0.clone()));
        assert_eq!(bt_lin_bilin.3, F::pairing(gt.clone(), key.v.1.1.clone() + key.g2_gen.clone()));
        assert_eq!(bt_lin_bilin, bt_bilin_lin);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_MSG2_linear_maps() {

        let mut rng = test_rng();
        let key = CRS::<F>::generate_crs(&mut rng);

        let g1 = Fr::rand(&mut rng);
        let g2 = G2Projective::rand(&mut rng).into_affine();
        let b1 = Com1::<F>::scalar_linear_map(&g1, &key);
        let b2 = Com2::<F>::linear_map(&g2);

        assert_eq!(b1.0, key.u.1.0.mul(g1));
        assert_eq!(b1.1, (key.u.1.1 + key.g1_gen).mul(g1));
        assert_eq!(b2.0, G2Affine::zero());
        assert_eq!(b2.1, g2);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_MSG2_linear_bilinear_map_commutativity() {

        let mut rng = test_rng();
        let key = CRS::<F>::generate_crs(&mut rng);

        let g1 = Fr::rand(&mut rng);
        let g2 = G2Projective::rand(&mut rng).into_affine();
        let b1 = Com1::<F>::scalar_linear_map(&g1, &key);
        let b2 = Com2::<F>::linear_map(&g2);
        let gt = g2.mul(g1).into_affine();

        let bt_lin_bilin = ComT::<F>::pairing(b1.clone(), b2.clone());
        let bt_bilin_lin = ComT::<F>::linear_map_MSG2(&gt, &key);

        assert_eq!(bt_lin_bilin.0, Fqk::one());
        assert_eq!(bt_lin_bilin.1, F::pairing(key.u.1.0.clone(), gt.clone()));
        assert_eq!(bt_lin_bilin.2, Fqk::one());
        assert_eq!(bt_lin_bilin.3, F::pairing(key.u.1.1.clone() + key.g1_gen.clone(), gt.clone()));
        assert_eq!(bt_lin_bilin, bt_bilin_lin);
    }
}
