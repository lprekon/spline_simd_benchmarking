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
    // fill the basis activations vec with the valued of the degree-0 basis functions
    for x in inputs{
        let x = *x; 
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
