#! /bin/bash

# USAGE:
# sh generate_benches.sh
# Run with <groth-sahai-rs>/benches as current / working directory


helpFunction() {
	echo ""
	echo "Usage: $0 [-hxv]"
	echo -e "\t-h\t--help\t\t\tDisplay help and usage information."
	echo -e "\t-x\t--exclude-benching\tDo not compute new benchmarks before generating graphs."
	echo -e "\t-m\t--include-microbenching\tInclude microbenchmarks when computing new benchmarks."
	echo -e "\t-v\t--verbose\t\tRun script in verbose mode."
	exit 1
}

options=$(getopt -o "hmxv" -l "help,include-microbenching,exclude-benching,verbose" -- "$@")
bench="true"
verbose="false"
microbench="false"
eval set -- "$options"
while true; do
	case "$1" in
		-x|--exclude-benching) bench="false" ;;
		-m|--include-microbenching) microbench="true" ;;
		-v|--verbose) verbose="true" ;;
		-h|--help) helpFunction ;;
		--)
			shift
			break
			;;
	esac
	shift
done

# TODO: Check that cargo is installed? Check that logs directory is created?
# TODO: Move this to Docker setup
echo "Installing necessary Python packages for 3.10.12 ..."
[ "$verbose" == "true" ] \
	&& pip3 install -r requirements.txt --no-input \
	|| pip3 install -r requirements.txt --no-input --quiet

if [ "$bench" == "true" ]; then
	echo "Removing old benchmark files ..."

	[ "$verbose" == "true" ] \
		&& cargo clean --profile bench \
		|| cargo clean --profile bench --quiet
	rm -rf "./logs/"

	echo "Compiling new benchmark target ..."
	[ "$verbose" == "true" ] \
		&& cargo bench --all-features --no-run \
		|| cargo bench --all-features --no-run --quiet

	echo "Generating new benchmarks (this may take a few minutes) ..."
	mkdir logs
	echo "description,size (B),compressed size (B)" >> ./logs/sizes.csv
	[ "$verbose" == "true" ] \
		&& ( [ "$microbench" == "true" ] \
			&& cargo bench --all-features --verbose \
			|| ( cargo bench "BLS12-381/Groth16" --all-features --verbose && cargo bench "BN254/Groth16" --all-features --verbose ) \
		) \
		|| ( [ "$microbench" == "true" ] \
			&& cargo bench --all-features --quiet \
			|| ( cargo bench "BLS12-381/Groth16" --all-features --quiet && cargo bench "BN254/Groth16" --all-features --quiet ) \
		) \
	# Benchmark groups:
	# cargo bench "BLS12-381/Microbenchmarks"
	# cargo bench "BN254/Microbenchmarks"
	# cargo bench "BLS12-381/Groth16" # (requires crate feature "groth16")
	# cargo bench "BN254/Groth16" # (requires crate feature "groth16")
fi

echo "Extracting benchmark data from target/ reports ..."
[ "$verbose" == "true" ] \
	&& python3 extract_benches.py verbose \
	|| python3 extract_benches.py

echo "Generating graphs from extracted benchmark data ..."
[ "$verbose" == "true" ] \
	&& python3 report_benches.py verbose \
	|| python3 report_benches.py
