use criterion::{criterion_group, criterion_main, Criterion};

pub fn benchmarks_work(c: &mut Criterion) {    
    c.bench_function(
        &format!("testing benchmark functionality"),
        |bench| {
            bench.iter(|| {
                (1..1000).fold(1, |sum, x| sum + x)
            });
        }
    );
}

criterion_group!(benches, benchmarks_work);
criterion_main!(benches);
