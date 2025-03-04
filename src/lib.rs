#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]
use std::vec;

/// recursivly compute the b-spline basis function for the given index `i`, degree `k`, and knot vector, at the given parameter `x`
pub fn basis_activation(i: usize, k: usize, x: f64, knots: &[f64]) -> f64 {
    // If degree is 0, the basis function is 1 if the parameter is within the range of the knot, and 0 otherwise
    if k == 0 {
        if knots[i] <= x && x < knots[i + 1] {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    // Otherwise, we compute compute basis functions one degree lower, and use them to compute the current basis function
    let left_recursion = basis_activation(i, k - 1, x, knots);
    let right_recursion = basis_activation(i + 1, k - 1, x, knots);

    // Compute the weights for the left and right "child" basis functions
    let left_coefficient = (x - knots[i]) / (knots[i + k] - knots[i]);
    let right_coefficient = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]);

    // Combine the left and right basis functions with the computed weights to produce the value of the current basis function
    let result = left_coefficient * left_recursion + right_coefficient * right_recursion;
    return result;
}

/// Calculate the value of the B-spline at the given parameter `x`
pub fn b_spline(x: f64, control_points: &[f64], knots: &[f64], degree: usize) -> f64 {
    let mut result = 0.0;
    for i in 0..control_points.len() {
        // compute the value of each top-level basis function, multiplying it by the corresponding control point, and sum the results
        result += control_points[i] * basis_activation(i, degree, x, knots);
    }
    return result;
}

/// Calculate the value of the B-spline at the given parameter `x` by looping over the basis functions
pub fn b_spline_loop_over_basis(
    inputs: &[f64],
    control_points: &[f64],
    knots: &[f64],
    degree: usize,
) -> Vec<f64> {
    // setup our data structures to hold the intermediate and final results
    let mut outputs = Vec::with_capacity(inputs.len());
    let mut basis_activations = vec![0.0; knots.len() - 1];

    for x in inputs {
        let x = *x;

        // For the current value of `x`, fill the basis activations vec with the value of the degree-0 basis functions
        for i in 0..knots.len() - 1 {
            if knots[i] <= x && x < knots[i + 1] {
                basis_activations[i] = 1.0;
            } else {
                basis_activations[i] = 0.0;
            }
        }
        /* For each degree k, compute the higher degree basis functions, 
         using the value of the "child" basis functions that were stored in the previous iteration,
         overwriting the child values once they're no longer needed */
        for k in 1..=degree {
            for i in 0..knots.len() - k - 1 {
                let left_coefficient = (x - knots[i]) / (knots[i + k] - knots[i]);
                let left_recursion = basis_activations[i];

                let right_coefficient = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]);
                let right_recursion = basis_activations[i + 1];

                basis_activations[i] =
                    left_coefficient * left_recursion + right_coefficient * right_recursion;
            }
        }

        // Finally, compute the ultimate value of this B-Spline at `x` by multiplying each basis function by the corresponding control point, and summing the results
        let mut result = 0.0;
        for i in 0..control_points.len() {
            result += control_points[i] * basis_activations[i];
        }
        outputs.push(result);
    }
    return outputs;
}

const SIMD_WIDTH: usize = 8; // We use this value to denote how many elements we intend to process in a single SIMD operation

pub fn b_spline_portable_simd(
    inputs: &[f64],
    control_points: &[f64],
    knots: &[f64],
    degree: usize,
) -> Vec<f64> {
    use std::simd::prelude::*;
    let mut outputs = Vec::with_capacity(inputs.len());
    let num_k0_activations = knots.len() - 1;
    let mut basis_activations = vec![0.0; num_k0_activations];

    for x in inputs {
        let x_splat: Simd<f64, SIMD_WIDTH> = Simd::splat(*x);
        // fill the basis activations vec with the value of the degree-0 basis functions
        let mut i = 0;
        // make sure we have enough elements to do a SIMD_WIDTH chunk
        // if we have e.g 8 k=0 activations, then starting with i=0 we'll pull write to basis_activations[0..7]
        while i + SIMD_WIDTH <= num_k0_activations {
            let knots_i_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&knots[i..]);
            let knots_i_plus_1_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&knots[i + 1..]);

            let left_mask: Mask<i64, SIMD_WIDTH> = knots_i_vec.simd_le(x_splat); // create a bitvector representing whether knots[i] <= x
            let right_mask: Mask<i64, SIMD_WIDTH> = x_splat.simd_lt(knots_i_plus_1_vec); // create a bitvector representing whether x < knots[i + 1]
            let full_mask: Mask<i64, SIMD_WIDTH> = left_mask & right_mask; // combine the two masks
            let activation_vec: Simd<f64, SIMD_WIDTH> =
                full_mask.select(Simd::splat(1.0), Simd::splat(0.0)); // create a vector with 1 in each position j where knots[i + j] <= x < knots[i + j + 1] and zeros elsewhere
            activation_vec.copy_to_slice(&mut basis_activations[i..]); // write the activations back to our basis_activations vector

            i += SIMD_WIDTH; // increment i by SIMD_WIDTH, to advance to the next chunk
        }
        // since knots.len() - 1 is not guaranteed to be a multiple of SIMD_WIDTH, we need to handle the remaining elements one by one
        while i < num_k0_activations {
            if knots[i] <= *x && *x < knots[i + 1] {
                basis_activations[i] = 1.0;
            } else {
                basis_activations[i] = 0.0;
            }
            i += 1;
        }

        // now to compute the higher degree basis functions
        for k in 1..=degree {
            let mut i = 0;
            while i + SIMD_WIDTH <= num_k0_activations - k {
                let knots_i_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&knots[i..]);
                let knots_i_plus_k_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&knots[i + k..]);
                let knots_i_plus_1_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&knots[i + 1..]);
                let knots_i_plus_k_plus_1_vec: Simd<f64, SIMD_WIDTH> =
                    Simd::from_slice(&knots[i + k + 1..]);

                // grab the value for and calculate the coefficient for the left term of the recursion, doing a SIMD_WIDTH chunk at a time
                let left_coefficient_vec =
                    (x_splat - knots_i_vec) / (knots_i_plus_k_vec - knots_i_vec);
                let left_recursion_vec: Simd<f64, SIMD_WIDTH> =
                    Simd::from_slice(&basis_activations[i..]);

                // grab the value for and calculate the coefficient for the right term of the recursion, doing a SIMD_WIDTH chunk at a time
                let right_coefficient = (knots_i_plus_k_plus_1_vec - x_splat)
                    / (knots_i_plus_k_plus_1_vec - knots_i_plus_1_vec);
                let right_recursion_vec: Simd<f64, SIMD_WIDTH> =
                    Simd::from_slice(&basis_activations[i + 1..]);

                let new_basis_activations_vec = left_coefficient_vec * left_recursion_vec
                    + right_coefficient * right_recursion_vec;
                new_basis_activations_vec.copy_to_slice(&mut basis_activations[i..]);

                i += SIMD_WIDTH;
            }
            // again, since knots.len() - k - 1 is not guaranteed to be a multiple of SIMD_WIDTH, we need to handle the remaining elements one by one
            while i < num_k0_activations - k {
                let left_coefficient = (x - knots[i]) / (knots[i + k] - knots[i]);
                let left_recursion = basis_activations[i];

                let right_coefficient = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]);
                let right_recursion = basis_activations[i + 1];

                basis_activations[i] =
                    left_coefficient * left_recursion + right_coefficient * right_recursion;
                i += 1;
            }
        }

        // now to compute the final result, in chunks of SIMD_WIDTH
        let mut i = 0;
        let mut result = 0.0;
        while i + SIMD_WIDTH <= control_points.len() {
            let control_points_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&control_points[i..]);
            let basis_activations_vec: Simd<f64, SIMD_WIDTH> =
                Simd::from_slice(&basis_activations[i..]);
            result += (control_points_vec * basis_activations_vec).reduce_sum();
            i += SIMD_WIDTH;
        }
        // handle the remaining elements one by one
        while i < control_points.len() {
            result += control_points[i] * basis_activations[i];
            i += 1;
        }
        outputs.push(result);
    }

    return outputs;
}

const FINAL_GATHER_IDX_ADJUSTMENT: [usize; SIMD_WIDTH] = [0, 1, 2, 3, 4, 5, 6, 7];

pub fn b_spline_portable_simd_transpose(
    inputs: &[f64],
    control_points: &[f64],
    knots: &[f64],
    degree: usize,
) -> Vec<f64> {
    use std::simd::prelude::*;
    let mut outputs = Vec::with_capacity(inputs.len());
    // the first `num_inputs` entries are the 0th basis functions for each input. the next `num_inputs` entries are the 1st basis functions for each input, and so on
    let mut basis_activations = vec![0.0; (knots.len() - 1) * inputs.len()];
    // start with k=0
    for i in 0..knots.len() - 1 {
        let knot_i_splat: Simd<f64, SIMD_WIDTH> = Simd::splat(knots[i]);
        let knot_i_plus_1_splat: Simd<f64, SIMD_WIDTH> = Simd::splat(knots[i + 1]);

        let mut x_idx = 0;
        // SIMD step
        while x_idx + SIMD_WIDTH < inputs.len() {
            let x_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&inputs[x_idx..]);

            let left_mask: Mask<i64, SIMD_WIDTH> = knot_i_splat.simd_le(x_vec); // create a bitvector representing whether knots[i] <= x
            let right_mask: Mask<i64, SIMD_WIDTH> = x_vec.simd_lt(knot_i_plus_1_splat); // create a bitvector representing whether x < knots[i + 1]
            let full_mask: Mask<i64, SIMD_WIDTH> = left_mask & right_mask; // combine the two masks
            let activation_vec: Simd<f64, SIMD_WIDTH> =
                full_mask.select(Simd::splat(1.0), Simd::splat(0.0)); // create a vector with 1 in each position j where knots[i + j] <= x < knots[i + j + 1] and zeros elsewhere

            let basis_index = i * inputs.len() + x_idx;
            activation_vec.copy_to_slice(&mut basis_activations[basis_index..]); // write the activations back to our basis_activations vector

            x_idx += SIMD_WIDTH;
        }
        // scalar step
        while x_idx < inputs.len() {
            let basis_index = i * inputs.len() + x_idx;
            if knots[i] <= inputs[x_idx] && inputs[x_idx] < knots[i + 1] {
                basis_activations[basis_index] = 1.0;
            } else {
                basis_activations[basis_index] = 0.0;
            }
            x_idx += 1;
        }
    }
    // now to compute the higher degree basis functions
    for k in 1..=degree {
        for i in 0..knots.len() - k - 1 {
            let knot_i_splat: Simd<f64, SIMD_WIDTH> = Simd::splat(knots[i]);
            let knot_i_plus_k_splat: Simd<f64, SIMD_WIDTH> = Simd::splat(knots[i + k]);
            let knot_i_plus_1_splat: Simd<f64, SIMD_WIDTH> = Simd::splat(knots[i + 1]);
            let knots_i_plus_k_plus_1_splat: Simd<f64, SIMD_WIDTH> = Simd::splat(knots[i + k + 1]);

            let mut x_idx = 0;
            // SIMD step
            while x_idx + SIMD_WIDTH < inputs.len() {
                let basis_index = i * inputs.len() + x_idx;
                let x_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&inputs[x_idx..]);

                // grab the value for and calculate the coefficient for the left term of the recursion, doing a SIMD_WIDTH chunk at a time
                let left_coefficient_vec =
                    (x_vec - knot_i_splat) / (knot_i_plus_k_splat - knot_i_splat);
                let left_recursion_vec: Simd<f64, SIMD_WIDTH> =
                    Simd::from_slice(&basis_activations[basis_index..]);

                // grab the value for and calculate the coefficient for the right term of the recursion, doing a SIMD_WIDTH chunk at a time
                let right_coefficient = (knot_i_plus_k_splat - x_vec)
                    / (knots_i_plus_k_plus_1_splat - knot_i_plus_1_splat);
                let right_recursion_vec: Simd<f64, SIMD_WIDTH> =
                    Simd::from_slice(&basis_activations[basis_index + inputs.len()..]);

                let new_basis_activations_vec = left_coefficient_vec * left_recursion_vec
                    + right_coefficient * right_recursion_vec;

                new_basis_activations_vec.copy_to_slice(&mut basis_activations[basis_index..]);

                x_idx += SIMD_WIDTH;
            }
            // again, since knots.len() - k - 1 is not guaranteed to be a multiple of SIMD_WIDTH, we need to handle the remaining elements one by one
            while x_idx < inputs.len() {
                let basis_index = i * inputs.len() + x_idx;
                let left_coefficient = (inputs[x_idx] - knots[i]) / (knots[i + k] - knots[i]);
                let left_recursion = basis_activations[basis_index];
                let right_coefficient =
                    (knots[i + k + 1] - inputs[x_idx]) / (knots[i + k + 1] - knots[i + 1]);
                let right_recursion = basis_activations[basis_index + inputs.len()];
                basis_activations[basis_index] =
                    left_coefficient * left_recursion + right_coefficient * right_recursion;
                x_idx += 1;
            }
        }
    }

    // now to comput the final results
    let input_num_splat = Simd::splat(inputs.len());
    for x_idx in 0..inputs.len() {
        let x_idx_splat = Simd::splat(x_idx);
        let mut result = 0.0;
        let mut i = 0;
        while i + SIMD_WIDTH < control_points.len() {
            let control_points_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&control_points[i..]);
            let gather_indexes = (Simd::splat(i) + Simd::from_array(FINAL_GATHER_IDX_ADJUSTMENT))
                * input_num_splat
                + x_idx_splat; // (i.. i+SIMD_WIDTH).map(|j| j * inputs.len() + x_idx)
            let basis_activations_vec: Simd<f64, SIMD_WIDTH> =
                Simd::gather_or_default(&basis_activations, gather_indexes);
            result += (control_points_vec * basis_activations_vec).reduce_sum();
            i += SIMD_WIDTH;
        }
        // handle the remaining elements one by one
        while i < control_points.len() {
            let basis_index = i * inputs.len() + x_idx;
            result += control_points[i] * basis_activations[basis_index];
            i += 1;
        }
        outputs.push(result);
    }

    return outputs;
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
pub fn b_spline_x86_intrinsics(
    inputs: &[f64],
    control_points: &[f64],
    knots: &[f64],
    degree: usize,
) -> Vec<f64> {
    use std::arch::x86_64::*;
    let mut outputs = Vec::with_capacity(inputs.len());
    let num_k0_activations = knots.len() - 1;
    let mut basis_activations = vec![0.0; num_k0_activations];
    for x in inputs {
        let x_splat = unsafe { _mm512_set1_pd(*x) };

        let mut i = 0;
        // SIMD step for the degree-0 basis functions
        while i + SIMD_WIDTH <= num_k0_activations {
            unsafe {
                let knots_i_vec = _mm512_loadu_pd(&knots[i]);
                let knots_i_plus_1_vec = _mm512_loadu_pd(&knots[i + 1]);
                let left_mask = _mm512_cmp_pd_mask(knots_i_vec, x_splat, _CMP_LE_OQ);
                let right_mask = _mm512_cmp_pd_mask(x_splat, knots_i_plus_1_vec, _CMP_LT_OQ);
                let full_mask = left_mask & right_mask;
                let activation_vec =
                    _mm512_mask_blend_pd(full_mask, _mm512_set1_pd(0.0), _mm512_set1_pd(1.0));

                _mm512_storeu_pd(&mut basis_activations[i], activation_vec);
            }
            i += SIMD_WIDTH;
        }
        // scalar step for the degree-0 basis functions
        while i < num_k0_activations {
            if knots[i] <= *x && *x < knots[i + 1] {
                basis_activations[i] = 1.0;
            } else {
                basis_activations[i] = 0.0;
            }
            i += 1;
        }
        for k in 1..=degree {
            let mut i = 0;
            // SIMD step for the higher degree basis functions
            while i + SIMD_WIDTH <= num_k0_activations - k {
                unsafe {
                    let knots_i_vec = _mm512_loadu_pd(&knots[i]);
                    let knots_i_plus_1_vec = _mm512_loadu_pd(&knots[i + 1]);
                    let knots_i_plus_k_vec = _mm512_loadu_pd(&knots[i + k]);
                    let knots_i_plus_k_plus_1_vec = _mm512_loadu_pd(&knots[i + k + 1]);

                    let left_coefficient_numerator_vec = _mm512_sub_pd(x_splat, knots_i_vec);
                    let left_coefficient_denominator_vec =
                        _mm512_sub_pd(knots_i_plus_k_vec, knots_i_vec);
                    let left_coefficient_vec = _mm512_div_pd(
                        left_coefficient_numerator_vec,
                        left_coefficient_denominator_vec,
                    );
                    let left_recursion_vec = _mm512_loadu_pd(&basis_activations[i]);

                    let right_coefficient_numerator_vec =
                        _mm512_sub_pd(knots_i_plus_k_plus_1_vec, x_splat);
                    let right_coefficient_denominator_vec =
                        _mm512_sub_pd(knots_i_plus_k_plus_1_vec, knots_i_plus_1_vec);
                    let right_coefficient_vec = _mm512_div_pd(
                        right_coefficient_numerator_vec,
                        right_coefficient_denominator_vec,
                    );
                    let right_recursion_vec = _mm512_loadu_pd(&basis_activations[i + 1]);

                    let left_val_vec = _mm512_mul_pd(left_coefficient_vec, left_recursion_vec);
                    let right_val_vec = _mm512_mul_pd(right_coefficient_vec, right_recursion_vec);

                    let new_basis_activations_vec = _mm512_add_pd(left_val_vec, right_val_vec);
                    _mm512_storeu_pd(&mut basis_activations[i], new_basis_activations_vec);
                }
                i += SIMD_WIDTH;
            }
            // scalar step for the higher degree basis functions
            while i < num_k0_activations - k {
                let left_coefficient = (x - knots[i]) / (knots[i + k] - knots[i]);
                let left_recursion = basis_activations[i];

                let right_coefficient = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]);
                let right_recursion = basis_activations[i + 1];

                basis_activations[i] =
                    left_coefficient * left_recursion + right_coefficient * right_recursion;
                i += 1;
            }
        }

        // SIMD step for the final result
        let mut i = 0;
        let mut result = 0.0;
        while i + SIMD_WIDTH <= control_points.len() {
            unsafe {
                let control_points_vec = _mm512_loadu_pd(&control_points[i]);
                let basis_activations_vec = _mm512_loadu_pd(&basis_activations[i]);
                let result_vec = _mm512_mul_pd(control_points_vec, basis_activations_vec);
                result += _mm512_reduce_add_pd(result_vec);
            }
            i += SIMD_WIDTH;
        }
        while i < control_points.len() {
            result += control_points[i] * basis_activations[i];
            i += 1;
        }
        outputs.push(result);
    }

    return outputs;
}

mod tests {

    #![allow(unused_imports)]
    use super::*;
    #[test]
    fn test_recursive() {
        let knots = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let control_points = vec![1.0; 9];
        let t_vals = [1.25, 6.2, 10.05];
        let expected_results = [0.31771, 1.0, 0.80712];
        for run in 0..t_vals.len() {
            let t = t_vals[run];
            let expected_result = expected_results[run];
            let result = b_spline(t, &control_points, &knots, 3);
            let rounded_result = (result * 100000.0).round() / 100000.0;
            assert_eq!(
                rounded_result, expected_result,
                "run {run}: actual != expected"
            );
        }
    }

    #[test]
    fn test_simple_loop() {
        let knots = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let control_points = vec![1.0; 9];
        let t_vals = [1.25, 6.2, 10.05];
        let expected_results = [0.31771, 1.0, 0.80712];
        let results = b_spline_loop_over_basis(&t_vals, &control_points, &knots, 3);
        let rounded_result: Vec<f64> = results
            .iter()
            .map(|x| (x * 100000.0).round() / 100000.0)
            .collect();
        assert_eq!(
            rounded_result,
            expected_results.to_vec(),
            "actual != expected"
        );
    }

    #[test]
    fn test_portable() {
        let knots = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let control_points = vec![1.0; 9];
        let t_vals = [1.25, 6.2, 10.05];
        let expected_results = [0.31771, 1.0, 0.80712];
        let results = b_spline_portable_simd(&t_vals, &control_points, &knots, 3);
        let rounded_result: Vec<f64> = results
            .iter()
            .map(|x| (x * 100000.0).round() / 100000.0)
            .collect();
        assert_eq!(
            rounded_result,
            expected_results.to_vec(),
            "actual != expected"
        );
    }

    #[test]
    fn test_portable_transpose() {
        let knots = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let control_points = vec![1.0; 9];
        let t_vals = [1.25, 6.2, 10.05];
        let expected_results = [0.31771, 1.0, 0.80712];
        let results = b_spline_portable_simd_transpose(&t_vals, &control_points, &knots, 3);
        let rounded_result: Vec<f64> = results
            .iter()
            .map(|x| (x * 100000.0).round() / 100000.0)
            .collect();
        assert_eq!(
            rounded_result,
            expected_results.to_vec(),
            "actual != expected"
        );
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        target_feature = "avx512f",
    ))]
    #[test]
    fn test_intrinsic() {
        let knots = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let control_points = vec![1.0; 9];
        let t_vals = [1.25, 6.2, 10.05];
        let expected_results = [0.31771, 1.0, 0.80712];
        let results = b_spline_x86_intrinsics(&t_vals, &control_points, &knots, 3);
        let rounded_result: Vec<f64> = results
            .iter()
            .map(|x| (x * 100000.0).round() / 100000.0)
            .collect();
        assert_eq!(
            rounded_result,
            expected_results.to_vec(),
            "actual != expected"
        );
    }
}
