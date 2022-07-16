use crate::activator;

#[derive(Debug)]
pub struct Linear {
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>, // need to save this for backward pass
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub grads: Vec<Vec<f64>>,
    pub local_grads: Vec<f64>,
    pub act: activator::ActivationContainer
}

impl Linear {
    pub fn new(input_features: u64, output_features: u64, bias: f64, act: activator::ActivationContainer) -> Linear {
        // initialize weights matrix
        let mut weights: Vec<Vec<f64>> = vec![];
        let mut inputs: Vec<f64> = vec![];
        let mut outputs: Vec<f64> = vec![];
        let mut grads: Vec<Vec<f64>> = vec![];
        let mut b_grads: Vec<f64> = vec![];
        let mut b: Vec<f64> = vec![];

        for _ in 0..output_features {
            outputs.push(0.0);
            b_grads.push(0.0);
            b.push(bias);           

            let mut w: Vec<f64> = vec![];
            let mut g: Vec<f64> = vec![];
            for _ in 0..input_features {
                if (inputs.len() as u64) < input_features {
                    inputs.push(0.0);
                }
                g.push(0.0);
                // random both positive and negative weight
                w.push(2f64 * rand::random::<f64>() - 1f64);
            }
            weights.push(w);
            grads.push(g);
        }
        Linear { inputs, outputs, w: weights, b, grads, local_grads: b_grads, act }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.inputs.len() {
            panic!("forward: input size is wrong");
        }
        let mut result: Vec<f64> = vec![];
        for j in 0..self.outputs.len() {
            let mut sum: f64 = 0.0;
            // loop through input and add w*x + b to sum
            for i in 0..inputs.len() {
                sum += (self.w[j][i] * inputs[i]) + self.b[j]
            }
            self.outputs[j] = sum;
            result.push((self.act.func)(sum));
        }
        self.inputs = inputs;
        result
    } 

    pub fn update(&mut self, lr: f64) {
        for j in 0..self.w.len() {
            self.b[j] -= lr * self.local_grads[j]; // update each neuron bias
            for i in 0..self.w[j].len() {
                self.w[j][i] -= lr * self.grads[j][i]; // update each weights
            }
        }
    }

    pub fn zero_grad(&mut self) {
        for j in 0..self.outputs.len() {
            self.local_grads[j] = 0.0;
            for i in 0..self.grads[j].len() {
                self.grads[j][i] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_new() {
        let linear = Linear::new(2, 3, 1.0, activator::linear());
        assert_eq!(linear.outputs.len(), 3);
        assert_eq!(linear.inputs.len(), 2);   

        assert_eq!(linear.w.len(), 3);
        assert_eq!(linear.w[0].len(), 2);
        assert_eq!(linear.b.len(), 3);

        assert_eq!(linear.grads.len(), 3);
        assert_eq!(linear.grads[0].len(), 2);
        assert_eq!(linear.local_grads.len(), 3);
    }

    #[test]
    fn test_linear_forward1() {
        let mut linear = Linear::new(2, 1, 1.0, activator::sigmoid());
        
        for j in 0..linear.w.len() {
            for i in 0..linear.w[j].len() {
                linear.w[j][i] = 1.0;
            }
        }
        
        assert_eq!(linear.forward(vec![1.0, 1.0])[0], 0.982013790037908442);
        assert_eq!(linear.outputs[0], 4.0);
    }

    #[test]
    fn test_linear_forward2() {
        let mut linear = Linear::new(2, 2, 1.0, activator::sigmoid());
        
        for j in 0..linear.w.len() {
            for i in 0..linear.w[j].len() {
                linear.w[j][i] =  (j as f64) + 1.0;
            }
        }
        let result = linear.forward(vec![0.0, 1.0]);
        assert_eq!(result[0], 0.9525741268224334);
        assert_eq!(result[1], 0.9820137900379084);
        assert_eq!(linear.outputs[0], 3.0);
        assert_eq!(linear.outputs[1], 4.0);
    }
}
