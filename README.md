<h1 align="center">Groth-Sahai</h1>
<p align="center">
    <a href="https://github.com/jdwhite88/groth-sahai-rs/blob/main/LICENSE-APACHE"><img src="https://img.shields.io/badge/license-APACHE-blue.svg"></a>
    <a href="https://github.com/jdwhite88/groth-sahai-rs/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
    <a href="https://deps.rs/repo/github/jdwhite88/groth-sahai-rs"><img src="https://deps.rs/repo/github/jdwhite88/groth-sahai-rs/status.svg"></a>
</p>

A Rust library for constructing Groth-Sahai non-interactive witness-indistinguishable and zero-knowledge proofs about the satisfiability of equations over bilinear groups [[1]](https://eprint.iacr.org/eprint-bin/getfile.pl?entry=2007/155&version=20160411:065033&file=155.pdf) [[2]](https://www.iacr.org/archive/pkc2010/60560179/60560179.pdf). Inspired by the [Java implementation](https://github.com/gijsvl/groth-sahai) of the Groth-Sahai protocol written by Gijs Van Laer.

This library is distributed under the MIT License and the Apache v2 License (see [License](#license)).

### Dependencies
* **[Arkworks](https://github.com/arkworks-rs/)** - A Rust ecosystem for developing and programming with zkSNARKs as well as finite field and elliptic curve arithmetic.
* **[Rayon](https://docs.rs/rayon/1.5.1/rayon/)** - A data parallelism library for Rust.

### ⚠ Disclaimer ⚠

* This library is a (currently incomplete) academic proof-of-concept only, and has not undergone thorough testing. **Do NOT use this implementation in production code.**

* **Your choice of bilinear group (G1, G2, GT, e) must be secure under the SXDH assumption.**

## Getting Started

### Installation

First, install the latest version of Rust using `rustup`:
```bash
rustup install stable
```
After that, use `cargo`, the standard Rust build tool, to build the library:
```bash
git clone https://github.com/jdwhite88/groth-sahai-rs.git
cd groth-sahai-rs
cargo build
```

### Test

To run the unit tests (i.e. private API; within library) and integration tests (i.e. public API; in `tests`):
```bash
cargo test
```
To run the benchmark tests (public API; in `benches`):
```bash
cargo bench
```

## Contributing

If you would like to contribute, and have not been invited as a direct collaborator to the project, follow the procedure below:

1. Fork the project
2. Create your feature branch (`git checkout -b feature-branch main`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request to merge with this repo

## References

**[1]** Jens Groth and Amit Sahai. **Efficient Non-interactive Proof Systems for Bilinear Groups**, *Advances in Cryptology -- EUROCRYPT 2008: 27th Annual International Conference on the Theory and Applications of Cryptographic Techniques*, Istanbul, Turkey. Springer Berlin Heidelberg, vol 4965: 415–432, 2008.

**[2]** Essam Ghadafi, Nigel P. Smart, and Bogdan Warinschi. **Groth-Sahai proofs revisited**. In Phong Q. Nguyen and David Pointcheval, editors, *PKC 2010, volume 6056 of LNCS*, pages 177–192. Springer, Heidelberg, May 2010.

## License

 This library is distributed under either of the following licenses:
 
 * Apache License v2.0 ([LICENSE-APACHE](LICENSE-APACHE))
 * MIT License ([LICENSE-MIT](LICENSE-MIT))
 
Unless explicitly stated otherwise, any contribution submitted for inclusion in this library shall be dual-licensed as above (as defined in the Apache v2 License), without any additional terms or conditions.

## Authors

* Jacob White - white570@purdue.edu
