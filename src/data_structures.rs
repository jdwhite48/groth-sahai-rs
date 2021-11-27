use ark_ec::{PairingEngine};
use ark_ff::{Zero, One, Field};
use ark_std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign}
};


/// B1,B2,BT forms a bilinear group for GS commitments

// TODO: implement AddAssign/MulAssign
// TODO: Implement as_col_vec for B1, B2 and as_matrix for BT
// TODO: Implement linear maps for each of B1, B2, BT
pub trait B1: Eq
    + Clone
    + Debug
    + Zero
    + Add<Self, Output = Self>
//    + AddAssign<Self>
{

}
pub trait B2: Eq
    + Clone
    + Debug
    + Zero
    + Add<Self, Output = Self>
//    + AddAssign<Self>
{

}
pub trait BT<C1: B1, C2: B2>: 
    Eq
    + Clone
    + Debug
// TODO: What's multiplication for commitment group BT?
//    + One
//    + Mul<Com1<E>, Com2<E>>
//    + MulAssign<Self>
{
    fn pairing(x: C1, y: C2) -> Self;
}

// SXDH instantiation's bilinear group for commitments

// TODO: Expose randomness? (see example data_structures in Arkworks)
#[derive(Clone, Debug)]
pub struct Com1<E: PairingEngine>(pub E::G1Affine, pub E::G1Affine);
#[derive(Clone, Debug)]
pub struct Com2<E: PairingEngine>(pub E::G2Affine, pub E::G2Affine);
#[derive(Clone, Debug)]
pub struct ComT<E: PairingEngine>(pub E::Fqk, pub E::Fqk, pub E::Fqk, pub E::Fqk);


// TODO: Refactor code to use Matrix trait (cleaner?)
// Would have to implement Matrix as a struct instead of pub type ... Vec<...> because "impl X for
// Vec<...> doesn't work
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

/// Sparse representation of matrices (with entries being scalar or GT)
pub type Matrix<F> = Vec<Vec<F>>;


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
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl<E: PairingEngine> Eq for Com1<E> {}
impl<E: PairingEngine> Add<Com1<E>> for Com1<E> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self (
            self.0 + other.0,
            self.1 + other.1
        )
    }
}
impl<E: PairingEngine> Zero for Com1<E> {
    fn zero() -> Com1<E> {
        Com1::<E> (
            E::G1Affine::zero(),
            E::G1Affine::zero()
        )
    }

    fn is_zero(&self) -> bool {
        *self == Com1::<E>::zero()
    }
}
impl<E: PairingEngine> B1 for Com1<E> {}


// Com2 implements B2
impl<E: PairingEngine> PartialEq for Com2<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl<E: PairingEngine> Eq for Com2<E> {}
impl<E: PairingEngine> Add<Com2<E>> for Com2<E> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self (
            self.0 + other.0,
            self.1 + other.1
        )
    }
}
impl<E: PairingEngine> Zero for Com2<E> {
    fn zero() -> Com2<E> {
        Com2::<E> (
            E::G2Affine::zero(),
            E::G2Affine::zero()
        )
    }

    fn is_zero(&self) -> bool {
        *self == Com2::<E>::zero()
    }
}
impl<E: PairingEngine> B2 for Com2<E> {}

// ComT implements BT<B1, B2>
impl<E: PairingEngine> PartialEq for ComT<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1 && self.2 == other.2 && self.3 == other.3
    }
}
impl<E: PairingEngine> Eq for ComT<E> {}
/*
impl<E: PairingEngine> One for ComT<E> {
    fn one() -> ComT<E> {
        ComT::<E> (
            E::Fqk::one(),
            E::Fqk::one(),
            E::Fqk::one(),
            E::Fqk::one()
        )
    }
}
*/
impl<E: PairingEngine> BT<Com1<E>, Com2<E>> for ComT<E> {
    #[inline]
    /// B_pairing computes entry-wise pairing products
    fn pairing(x: Com1<E>, y: Com2<E>) -> ComT<E> {
        ComT::<E>(
            // TODO: OPTIMIZATION -- If either element is 0 (G1 / G2), just output 1 (Fqk)
            E::pairing::<E::G1Affine, E::G2Affine>(x.0.clone(), y.0.clone()),
            E::pairing::<E::G1Affine, E::G2Affine>(x.0.clone(), y.1.clone()),
            E::pairing::<E::G1Affine, E::G2Affine>(x.1.clone(), y.0.clone()),
            E::pairing::<E::G1Affine, E::G2Affine>(x.1.clone(), y.1.clone()),
        )
    }
}


/// Compute row of matrix corresponding to multiplication of scalar matrices
// TODO: OPTIMIZATION -- paralellize with Rayon
fn matrix_mul_row<F: Field>(row: &[F], rhs: &Matrix<F>, dim: usize) -> Vec<F> {
    
    // Assuming every column in b has the same length
    let rhs_col_dim = rhs[0].len();
    (0..rhs_col_dim)
        .map( |j| {
            (0..dim) 
                .map( |k| row[k] * rhs[k][j] ).sum()
        })
        .collect::<Vec<F>>()
}

/// Matrix multiplication of field elements (scalar/Fr or GT/Fqk)
// TODO: OPTIMIZATION -- parallelize with Rayon
pub(crate) fn matrix_mul<F: Field>(lhs: &Matrix<F>, rhs: &Matrix<F>) -> Matrix<F> {
    if lhs.len() == 0 || lhs[0].len() == 0 {
        return vec![];
    }
    if rhs.len() == 0 || rhs[0].len() == 0 {
        return vec![];
    }

    // Assuming every row in a and column in b has the same length
    assert_eq!(lhs[0].len(), rhs.len());
    let row_dim = lhs.len();

    (0..row_dim)
        .map( |i| {
            let row = &lhs[i];
            let dim = rhs.len();
            matrix_mul_row::<F>(row, rhs, dim)
        })
        .collect::<Matrix<F>>()
}


#[cfg(test)]
mod tests {

    use ark_bls12_381::{Bls12_381 as F};
    use ark_ff::{UniformRand, field_new};
    use ark_ec::ProjectiveCurve;
    use ark_std::test_rng;

    use crate::data_structures::*;

    type G1Projective = <F as PairingEngine>::G1Projective;
    type G1Affine = <F as PairingEngine>::G1Affine;
    type G2Projective = <F as PairingEngine>::G2Projective;
    type G2Affine = <F as PairingEngine>::G2Affine;
    type GT = <F as PairingEngine>::Fqk;

    
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

    #[test]
    fn test_scalar_matrix_mul_row() {

        type Fr = <F as PairingEngine>::Fr;

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
        let res: Vec<Fr> = matrix_mul_row::<Fr>(&lhs, &rhs, 3);

        // 1 x 1 resulting matrix
        assert_eq!(res.len(), 1);
   
        assert_eq!(exp, res);
    }


    #[test]
    fn test_scalar_matrix_mul_entry() {
        
        type Fr = <F as PairingEngine>::Fr;
        
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
        let res: Matrix<Fr> = matrix_mul::<Fr>(&lhs, &rhs);

        // 1 x 1 resulting matrix
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].len(), 1);
   
        assert_eq!(exp, res);
    }


    #[test]
    fn test_scalar_matrix_mul() {
        
        type Fr = <F as PairingEngine>::Fr;
        
        // 2 x 3 (row) vector
        let one = Fr::one();
        let lhs: Matrix<Fr> = vec![
            vec![one, field_new!(Fr, "2"), field_new!(Fr, "3")],
            vec![field_new!(Fr, "4"), field_new!(Fr, "5"), field_new!(Fr, "6")]
        ];
        // 3 x 4 (column) vector
        let rhs: Matrix<Fr> = vec![
            vec![field_new!(Fr, "7"), field_new!(Fr, "8"), field_new!(Fr, "9"), field_new!(Fr, "10")],
            vec![field_new!(Fr, "11"), field_new!(Fr, "12"), field_new!(Fr, "13"), field_new!(Fr, "14")],
            vec![field_new!(Fr, "15"), field_new!(Fr, "16"), field_new!(Fr, "17"), field_new!(Fr, "18")]
        ];
        let exp: Matrix<Fr> = vec![
            vec![field_new!(Fr, "74"), field_new!(Fr, "80"), field_new!(Fr, "86"), field_new!(Fr, "92")],
            vec![field_new!(Fr, "173"), field_new!(Fr, "188"), field_new!(Fr, "203"), field_new!(Fr, "218")]
        ];
        let res: Matrix<Fr> = matrix_mul::<Fr>(&lhs, &rhs);

        // 2 x 4 resulting matrix
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].len(), 4);
        assert_eq!(res[1].len(), 4);

        assert_eq!(exp, res);
    }
}
