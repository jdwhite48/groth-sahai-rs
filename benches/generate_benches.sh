#! /bin/bash
set -e

# USAGE:
# sh generate_benches.sh
# Run with <groth-sahai-rs>/benches as current / working directory


helpFunction() {
	echo ""
	echo "Usage: $0 [-hnsv]"
	echo -e "\t-h\t--help\t\tDisplay help and usage information."
	echo -e "\t-n\t--no-benching\tDo not compute any new benchmarks."
	echo -e "\t-s\t--short\t\tExclude memory usage and microbenchmarks if computing new benchmarks."
	echo -e "\t-v\t--verbose\tRun script in verbose mode."
	exit 1
}

memName=''
memTest=''
benchMem() {
	# See Valgrind's Massif profiler section of the manual https://valgrind.org/docs/manual/ms-manual.html#ms-manual.not-measured for more details
	# Entire pages allocated for memory: stack, heap, code, BSS, data (all in mem_heap_B)
	valgrind --vgdb=no --tool=massif --pages-as-heap=yes --massif-out-file=./logs/mem_${memTest}_full.log ../target/release/examples/${memTest} 2> /dev/null
	# Only heap (mem_heap_B and mem_heap_extra_B) and stack (mem_stacks_B) memory segments
	valgrind --vgdb=no --tool=massif --heap=yes --stacks=yes --massif-out-file=./logs/mem_${memTest}.log ../target/release/examples/${memTest} 2> /dev/null
	pageSize="$(cat ./logs/mem_${memTest}_full.log | perl -ne 'm/mem_heap_B=(\d+)/ && print "$1\n"' | tail -n1)"
	heapSize="$(cat ./logs/mem_${memTest}.log | perl -ne 'm/mem_heap_B=(\d+)/ && print "$1\n"' | tail -n1)"
	xHeapSize="$(cat ./logs/mem_${memTest}.log | perl -ne 'm/mem_heap_extra_B=(\d+)/ && print "$1\n"' | tail -n1)"
	stackSize="$(cat ./logs/mem_${memTest}.log | perl -ne 'm/mem_stacks_B=(\d+)/ && print "$1\n"' | tail -n1)"
	# Average (%K) and max (%M) RSS for process, measured by wait3()/wait4() --> getrusage()
	#echo "description,Avg RSS (KB),Max RSS (KB),Heap Size (B),Extra Heap Size (B),Stack Size (B),Heap + Stack Size (B),Page Size (B)" > ./logs/memory.csv
	/usr/bin/time --format "$memName,%K,%M,$heapSize,$xHeapSize,$stackSize,$(echo "$(($heapSize + $xHeapSize + $stackSize))"),$pageSize" --append --output ./logs/memory.csv ../target/release/examples/${memTest}
}

options=$(getopt -o "hnsv" -l "help,no-benching,short,verbose" -- "$@")
bench="true"
verbose="false"
microbench="true"
eval set -- "$options"
while true; do
	case "$1" in
		-n|--no-benching) bench="false" ;;
		-s|--short) microbench="false" ;;
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
echo -e "\e[0;36mInstalling necessary Python packages for 3.10.12 ...\e[0m"
[ "$verbose" == "true" ] \
	&& pip3 install -r requirements.txt --no-input \
	|| pip3 install -r requirements.txt --no-input --quiet

# Check that wait3(2) or wait4(2) --> getrusage(2) syscall is used for RSS
echo -e "\e[0;36mChecking support for accurate memory usage sampling ...\e[0m"
strace /usr/bin/time --format "%K,%M" echo "Hello, world!" > /dev/null 2> trace.txt
grep -e "^wait3(" -e "^wait4" ./trace.txt > /dev/null || echo -e "\e[0;31mWARNING: /usr/bin/time uses times(2) instead of wait3(2)/wait4(2) system calls; memory usage benchmarks may be inaccurate.\e[0m"
rm -f ./trace.txt

if [ "$bench" == "true" ]; then
	echo -e "\e[0;36mRemoving old benchmark files ...\e[0m"

	[ "$verbose" == "true" ] \
		&& cargo clean \
		|| cargo clean --quiet
	rm -rf "./logs/"

	echo -e "\e[0;36mCompiling new benchmark targets (this may take a few minutes) ...\e[0m"
	[ "$verbose" == "true" ] \
		&& cargo bench --no-run --all-features \
		|| cargo bench --no-run --all-features --quiet

	([ "$verbose" == "true" ] && [ "$microbench" = "true" ]) \
		&& cargo build --release --examples --all-features \
		|| cargo build --release --examples --all-features --quiet

	mkdir logs
	echo -e "\e[0;36mRunning memory benchmarks ...\e[0m"

	echo "description,Avg RSS (KB),Max RSS (KB),Heap Size (B),Extra Heap Size (B),Stack Size (B),Heap + Stack Size (B),Page Size (B)" > ./logs/memory.csv

	# NOTE: Ordering here is important, since ("full" aside) prior examples serialize data for use in later examples

	#/usr/bin/time --format "CRS,%K,%M" --append --output ./logs/memory.csv ../target/release/examples/crs && sed '$ s/.$//' ./logs/memory.csv
	#valgrind --vgdb=no --tool=massif --pages-as-heap=yes --massif-out-file=./logs/crs ../target/release/examples/crs
	#cat ./logs/crs | perl -ne 'm/mem_heap_B=(\d+)/ && print "$1\n"' | tail -n1 >> ./logs/memory.csv
	memName="GS CRS"; memTest="crs"; benchMem
	memName="GS Commit scalar to G1"; memTest="commit_scalar_to_G1"; benchMem
	memName="GS Commit scalar to G2"; memTest="commit_scalar_to_G2"; benchMem
	memName="GS Commit G1"; memTest="commit_G1"; benchMem
	memName="GS Commit G2"; memTest="commit_G2"; benchMem

	# TODO: Real example of a circuit (e.g. Merkle trees)
	memName="Tiny Groth16 keygen/setup"; memTest="groth16_keygen"; benchMem
	memName="Tiny Groth16 prove"; memTest="groth16_prove"; benchMem

	memName="GS-over-Groth16 commit"; memTest="groth16_gscommit"; benchMem
	memName="GS-over-Groth16 prove"; memTest="groth16_gsprove"; benchMem
	memName="GS-over-Groth16 verify"; memTest="groth16_gsverify"; benchMem
	memName="GS-over-Groth16 full"; memTest="groth16_gs"; benchMem

	echo -e "\e[0;36mRunning time and size benchmarks (this may take up to an hour) ...\e[0m"
	echo "description,size (B),compressed size (B)" > ./logs/sizes.csv
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

echo -e "\e[0;36mExtracting benchmark data from target/ reports ...\e[0m"
[ "$verbose" == "true" ] \
	&& python3 extract_benches.py verbose \
	|| python3 extract_benches.py

echo -e "\e[0;36mGenerating graphs from extracted benchmark data ...\e[0m"
[ "$verbose" == "true" ] \
	&& python3 report_benches.py verbose \
	|| python3 report_benches.py
