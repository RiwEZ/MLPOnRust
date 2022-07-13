#[derive(Debug)]
pub struct ActivationContainer {
    pub func: fn(f64) -> f64,
    pub der: fn(f64) -> f64
}

pub fn sigmoid_f(input: f64) -> f64 {
    1.0/(1.0 + (-input).exp())
}

pub fn sigmoid_der(input: f64) -> f64 {
    sigmoid_f(input) * (1.0 - sigmoid_f(input))
}

pub fn sigmoid() -> ActivationContainer {
    ActivationContainer { func: sigmoid_f, der: sigmoid_der }
}

pub fn sigmoid_vec(input: &Vec<f64>) -> Vec<f64> {
    let mut res: Vec<f64> = vec![];
    for x in input {
        res.push(sigmoid_f(*x))
    }
    res
}