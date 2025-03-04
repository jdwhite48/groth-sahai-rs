#! /bin/bash

# USAGE:
# sh generate_benches.sh
# Run with <groth-sahai-rs>/benches as current / working directory

# TODO: Check that cargo is installed? Check that logs directory is created?

# TODO: Move this to Docker setup
echo "Installing necessary Python packages for 3.10.12 ..."
pip3 install -r requirements.txt --quiet --no-input #--python-version "3.10.12"

echo "Removing old benchmark files ..."
cargo clean --profile bench --quiet
rm -rf "./logs/"

echo "Compiling new benchmark target ..."
cargo bench --all-features --quiet --no-run

echo "Generating new benchmarks (this may take a few minutes) ..."
cargo bench --all-features --quiet
# Benchmark groups:
# cargo bench "BLS12-381/Microbenchmarks"
# cargo bench "BN254/Microbenchmarks"
# cargo bench "BLS12-381/Groth16" # (requires crate feature "groth16")
# cargo bench "BN254/Groth16" # (requires crate feature "groth16")

echo "Extracting benchmark data from target/ reports ..."
mkdir logs
python3 extract_benches.py
