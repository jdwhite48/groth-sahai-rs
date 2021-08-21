use ark_ec::PairingEngine;

/// bilinear group for commitments
pub struct B1<E: PairingEngine>(pub E::G1Affine, pub E::G1Affine);
pub struct B2<E: PairingEngine>(pub E::G2Affine, pub E::G2Affine);
pub struct BT<E: PairingEngine>(pub E::Fqk, pub E::Fqk, pub E::Fqk, pub E::Fqk);

#[allow(non_snake_case)]
#[inline]
/// Entry-wise pairing products
pub(crate) fn B_pairing<E>(x: B1<E>, y: B2<E>) -> BT<E>
where
    E: PairingEngine,
{
    BT::<E>(
        E::pairing::<E::G1Affine, E::G2Affine>(x.0.clone(), y.0.clone()),
        E::pairing::<E::G1Affine, E::G2Affine>(x.0.clone(), y.1.clone()),
        E::pairing::<E::G1Affine, E::G2Affine>(x.1.clone(), y.0.clone()),
        E::pairing::<E::G1Affine, E::G2Affine>(x.1.clone(), y.1.clone()),
    )
}
