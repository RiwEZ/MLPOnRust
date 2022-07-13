pub fn sigmoid(input: f64) -> f64 {
    1.0/(1.0 + (-input).exp())
}

pub fn sigmoid_vec(input: &Vec<f64>) -> Vec<f64> {
    let mut res: Vec<f64> = vec![];
    for x in input {
        res.push(sigmoid(*x))
    }
    res
}

pub fn sigmoid_der(input: f64) -> f64 {
    sigmoid(input) * (1.0 - sigmoid(input))
}