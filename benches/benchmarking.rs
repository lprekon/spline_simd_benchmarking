#![feature(test)]
extern crate test;
use test::Bencher;

use rust_simd_becnhmarking::{b_spline, b_spline_loop_over_basis};

// define the parameters for the B-spline we'll use in each benchmark
const DEGREE: usize = 4;
const CONTROL_POINTS: [f64; 16] = [1.0; 16];
const KNOTS: [f64; 21] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 20.0,
];

const INPUT_SIZE: usize = 100;

#[bench]
/// benchmark evaluating a degree-3 B-spline with 20 knots and 16 basis functions, over 100 different input values
fn bench_recursive_method(b: &mut Bencher) {
    let input_values: Vec<f64> = (0..INPUT_SIZE).map(|x| x as f64 / 10.0).collect(); // 100 input values, ranging from 0.0 to 9.9
    b.iter(|| {
        for x in input_values.iter() {
            let _ = b_spline(*x, &CONTROL_POINTS, &KNOTS, DEGREE);
        }
    });
}

#[bench]
fn bench_simple_loop_method(b: &mut Bencher) {
    let input_values: Vec<f64> = (0..INPUT_SIZE).map(|x| x as f64 / 10.0).collect(); // 100 input values, ranging from 0.0 to 9.9
    b.iter(|| {
        // measure how long it takes to evaluate the B-spline for each input value
        for x in input_values.iter() {
            let _ = b_spline_loop_over_basis(*x, &CONTROL_POINTS, &KNOTS, DEGREE);
        }
    });
}
