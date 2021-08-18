#[macro_use]
extern crate bencher;

use bencher::Bencher;

fn bench_test(bench: &mut Bencher) {
    bench.iter(|| {
        (0..1000).fold(0, |x, y| x + y)
    })
}

benchmark_group!(benches, bench_test);
benchmark_main!(benches);
