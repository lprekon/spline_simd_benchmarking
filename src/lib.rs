#![feature(portable_simd)]

/// recursivly compute the b-spline basis function for the given index `i`, degree `k`, and knot vector, at the given parameter `x`
pub fn basis_activation(i: usize, k: usize, x: f64, knots: &[f64]) -> f64 {
    if k == 0 {
        if knots[i] <= x && x < knots[i + 1] {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    let left_coefficient = (x - knots[i]) / (knots[i + k] - knots[i]);
    let left_recursion = basis_activation(i, k - 1, x, knots);

    let right_coefficient = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]);
    let right_recursion = basis_activation(i + 1, k - 1, x, knots);

    let result = left_coefficient * left_recursion + right_coefficient * right_recursion;
    return result;
}

/// Calculate the value of the B-spline at the given parameter `x`
pub fn b_spline(x: f64, control_points: &[f64], knots: &[f64], degree: usize) -> f64 {
    let mut result = 0.0;
    for i in 0..control_points.len() {
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
    let mut outputs = Vec::with_capacity(inputs.len());
    let mut basis_activations = vec![0.0; knots.len() - 1];
    for x in inputs {
        let x = *x;
        // fill the basis activations vec with the value of the degree-0 basis functions
        for i in 0..knots.len() - 1 {
            if knots[i] <= x && x < knots[i + 1] {
                basis_activations[i] = 1.0;
            } else {
                basis_activations[i] = 0.0;
            }
        }

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

        let mut result = 0.0;
        for i in 0..control_points.len() {
            result += control_points[i] * basis_activations[i];
        }
        outputs.push(result);
    }
    return outputs;
}

const SIMD_WIDTH: usize = 4;

pub fn b_spline_portable_simd(
    inputs: &[f64],
    control_points: &[f64],
    knots: &[f64],
    degree: usize,
) -> Vec<f64> {
    use std::simd::prelude::*;
    let mut outputs = Vec::with_capacity(inputs.len());
    let mut basis_activations = vec![0.0; knots.len() - 1];

    for x in inputs {
        let x_splat: Simd<f64, SIMD_WIDTH> = f64x4::splat(*x);
        // fill the basis activations vec with the value of the degree-0 basis functions
        let mut i = 0;
        while i + SIMD_WIDTH < knots.len() - 1 {
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
        while i < knots.len() - 1 {
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
            while i + SIMD_WIDTH < knots.len() - k - 1 {
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
            while i < knots.len() - k - 1 {
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
        while i + SIMD_WIDTH < control_points.len() {
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
