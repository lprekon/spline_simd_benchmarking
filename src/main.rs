pub fn main() {
    let spline_size = 2000;
    let input_size = 2000;
    let degree = 4;
    let control_points = vec![1.0; spline_size];
    let knots = (0..spline_size + degree + 1)
        .map(|x| x as f64 / (spline_size + degree + 1) as f64)
        .collect::<Vec<_>>();
    let inputs = (0..input_size)
        .map(|x| x as f64 / input_size as f64)
        .collect::<Vec<_>>();
    let _ =
        rust_simd_benchmarking::b_spline_portable_simd(&inputs, &control_points, &knots, degree);
}
