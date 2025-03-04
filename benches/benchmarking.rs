#![feature(test)]
extern crate test;
use test::Bencher;

#[allow(unused_imports)]
use rust_simd_benchmarking::*;

// define the parameters for the B-spline we'll use in each benchmark
fn get_test_parameters() -> (usize, Vec<f64>, Vec<f64>, Vec<f64>) {
    let spline_size = 100;
    let input_size = 100;
    let degree = 4;
    let control_points = vec![1.0; spline_size];
    let knots = (0..spline_size + degree + 1)
        .map(|x| x as f64 / (spline_size + degree + 1) as f64)
        .collect::<Vec<_>>();
    let inputs = (0..input_size)
        .map(|x| x as f64 / input_size as f64)
        .collect::<Vec<_>>();
    (degree, control_points, knots, inputs)
}

#[bench]
// benchmark evaluating a degree-3 B-spline with 20 knots and 16 basis functions, over 100 different input values
fn bench_recursive_method(b: &mut Bencher) {
    let (degree, control_points, knots, inputs) = get_test_parameters();
    b.iter(|| {
        for x in inputs.iter() {
            let _ = b_spline(*x, &control_points, &knots, degree);
        }
    });
}

#[bench]
fn bench_simple_loop_method(b: &mut Bencher) {
    let (degree, control_points, knots, inputs) = get_test_parameters();
    b.iter(|| {
        let _ = b_spline_loop_over_basis(&inputs, &control_points, &knots, degree);
    });
}

#[bench]
fn bench_portable_simd_method(b: &mut Bencher) {
    let (degree, control_points, knots, inputs) = get_test_parameters();
    b.iter(|| {
        let _ = b_spline_portable_simd(&inputs, &control_points, &knots, degree);
    });
}

// #[bench]
// fn bench_portable_simd_transposed_method(b: &mut Bencher) {
//     let (degree, control_points, knots, inputs) = get_test_parameters();
//     b.iter(|| {
//         let _ = b_spline_portable_simd_transpose(&inputs, &control_points, &knots, degree);
//     });
// }

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
#[bench]
fn bench_intrinsic_simd_method(b: &mut Bencher) {
    let (degree, control_points, knots, inputs) = get_test_parameters();
    b.iter(|| {
        let _ = b_spline_x86_intrinsics(&inputs, &control_points, &knots, degree);
    });
}
