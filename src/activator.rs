#[derive(Debug)]
pub struct ActivationContainer {
    pub func: fn(f64) -> f64,
    pub der: fn(f64) -> f64,
    pub name: String
}

fn sigmoid_f(input: f64) -> f64 {
    1.0/(1.0 + (-input).exp())
}

fn sigmoid_der(input: f64) -> f64 {
    sigmoid_f(input) * (1.0 - sigmoid_f(input))
}

pub fn sigmoid() -> ActivationContainer {
    ActivationContainer { func: sigmoid_f, der: sigmoid_der, name: "sigmoid".to_string() }
}

pub fn sigmoid_vec(input: &Vec<f64>) -> Vec<f64> {
    let mut res: Vec<f64> = vec![];
    for x in input {
        res.push(sigmoid_f(*x))
    }
    res
}

pub fn linear() -> ActivationContainer {
    fn l(input: f64) -> f64 {
        input
    }
    fn l_der(_input: f64) -> f64 {
        0.0   
    }
    ActivationContainer { func: l, der: l_der, name: "linear".to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_f() {
        assert_eq!(sigmoid_f(1.0), 0.7310585786300048792512);
        assert_eq!(sigmoid_f(-1.0), 0.2689414213699951207488);
        assert_eq!(sigmoid_f(0.0), 0.5);
    }

    #[test]
    fn test_sigmoid_der() {
        assert_eq!(sigmoid_der(1.0), 0.1966119332414818525374);
        assert_eq!(sigmoid_der(-1.0), 0.1966119332414818525374);
        assert_eq!(sigmoid_der(0.0), 0.25);
    }

    #[test]
    fn test_sidmoid_container() {
        let act = sigmoid();

        assert_eq!((act.func)(1.0), 0.7310585786300048792512);
        assert_eq!((act.func)(-1.0), 0.2689414213699951207488);
        assert_eq!((act.func)(0.0), 0.5);
        assert_eq!((act.der)(1.0), 0.1966119332414818525374);
        assert_eq!((act.der)(-1.0), 0.1966119332414818525374);
        assert_eq!((act.der)(0.0), 0.25);
    }
}