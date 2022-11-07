use crate::activator;

#[derive(Debug)]
pub struct Layer {
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>, // need to save this for backward pass
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub grads: Vec<Vec<f64>>,
    pub w_prev_changes: Vec<Vec<f64>>,
    pub local_grads: Vec<f64>,
    pub b_prev_changes: Vec<f64>,
    pub act: activator::ActivationContainer,
}

impl Layer {
    pub fn new(
        input_features: u64,
        output_features: u64,
        bias: f64,
        act: activator::ActivationContainer,
    ) -> Layer {
        // initialize weights matrix
        let mut weights: Vec<Vec<f64>> = vec![];
        let mut inputs: Vec<f64> = vec![];
        let mut outputs: Vec<f64> = vec![];
        let mut grads: Vec<Vec<f64>> = vec![];
        let mut local_grads: Vec<f64> = vec![];
        let mut w_prev_changes: Vec<Vec<f64>> = vec![];
        let mut b_prev_changes: Vec<f64> = vec![];
        let mut b: Vec<f64> = vec![];

        for _ in 0..output_features {
            outputs.push(0.0);
            local_grads.push(0.0);
            b_prev_changes.push(0.0);
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
            grads.push(g.clone());
            w_prev_changes.push(g);
        }
        Layer {
            inputs,
            outputs,
            w: weights,
            b,
            grads,
            w_prev_changes,
            local_grads,
            b_prev_changes,
            act,
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.inputs.len() {
            panic!("forward: input size is wrong");
        }

        let result: Vec<f64> = self
            .w
            .iter()
            .zip(self.b.iter())
            .zip(self.outputs.iter_mut())
            .map(|((w_j, b_j), o_j)| {
                let sum = inputs
                    .iter()
                    .zip(w_j.iter())
                    .fold(0.0, |s, (v, w_ji)| s + w_ji * v)
                    + b_j;
                *o_j = sum;
                (self.act.func)(sum)
            })
            .collect();

        self.inputs = inputs.clone();
        result
    }

    pub fn update(&mut self, lr: f64, momentum: f64) {
        for j in 0..self.w.len() {
            let delta_b = lr * self.local_grads[j] + momentum * self.b_prev_changes[j];
            self.b[j] -= delta_b; // update each neuron bias
            self.b_prev_changes[j] = delta_b;
            for i in 0..self.w[j].len() {
                // update each weights
                let delta_w = lr * self.grads[j][i] + momentum * self.w_prev_changes[j][i];
                self.w[j][i] -= delta_w;
                self.w_prev_changes[j][i] = delta_w;
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

#[derive(Debug)]
pub struct Net {
    pub layers: Vec<Layer>,
    pub parameters: u64,
}

impl Net {
    pub fn from_layers(layers: Vec<Layer>) -> Net {
        let mut parameters: u64 = 0;
        for l in &layers {
            parameters += (l.w.len() * l.w[0].len()) as u64;
            parameters += l.b.len() as u64;
        }

        Net { layers, parameters }
    }

    pub fn new(architecture: Vec<u64>) -> Net {
        let mut layers: Vec<Layer> = vec![];
        for i in 1..architecture.len() {
            layers.push(Layer::new(
                architecture[i - 1],
                architecture[i],
                1f64,
                activator::sigmoid(),
            ))
        }
        Net::from_layers(layers)
    }

    /// Set this network parameters from flattened parameters.
    pub fn set_params(&mut self, params: &Vec<f64>) {
        if self.parameters != params.len() as u64 {
            panic!["The neural network parameters size is not equal to individual size"];
        }
        let mut idx: usize = 0;

        for l in self.layers.iter_mut() {
            l.w.iter_mut().for_each(|w_j| {
                w_j.iter_mut().for_each(|w_ji| {
                    *w_ji = params[idx];
                    idx += 1;
                })
            });

            l.b.iter_mut().for_each(|b_i| {
                *b_i = params[idx];
                idx += 1;
            });
        }
    }

    pub fn zero_grad(&mut self) {
        for l in 0..self.layers.len() {
            self.layers[l].zero_grad();
        }
    }

    pub fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut result = self.layers[0].forward(input);
        for l in 1..self.layers.len() {
            result = self.layers[l].forward(&result);
        }
        result
    }

    pub fn update(&mut self, lr: f64, momentum: f64) {
        for l in 0..self.layers.len() {
            self.layers[l].update(lr, momentum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_new() {
        let linear = Layer::new(2, 3, 1.0, activator::linear());
        assert_eq!(linear.outputs.len(), 3);
        assert_eq!(linear.inputs.len(), 2);

        assert_eq!(linear.w.len(), 3);
        assert_eq!(linear.w[0].len(), 2);
        assert_eq!(linear.b.len(), 3);

        assert_eq!(linear.grads.len(), 3);
        assert_eq!(linear.w_prev_changes.len(), 3);
        assert_eq!(linear.grads[0].len(), 2);
        assert_eq!(linear.w_prev_changes[0].len(), 2);
        assert_eq!(linear.local_grads.len(), 3);
        assert_eq!(linear.b_prev_changes.len(), 3);
    }

    #[test]
    fn test_linear_forward1() {
        let mut linear = Layer::new(2, 1, 1.0, activator::sigmoid());

        for j in 0..linear.w.len() {
            for i in 0..linear.w[j].len() {
                linear.w[j][i] = 1.0;
            }
        }

        assert_eq!(linear.forward(&vec![1.0, 1.0])[0], 0.9525741268224334);
        assert_eq!(linear.outputs[0], 3.0);
    }

    #[test]
    fn test_linear_forward2() {
        let mut linear = Layer::new(2, 2, 1.0, activator::sigmoid());

        for j in 0..linear.w.len() {
            for i in 0..linear.w[j].len() {
                linear.w[j][i] = (j as f64) + 1.0;
            }
        }
        let result = linear.forward(&vec![0.0, 1.0]);
        assert_eq!(linear.outputs[0], 2.0);
        assert_eq!(linear.outputs[1], 3.0);
        assert_eq!(result[0], 0.8807970779778823);
        assert_eq!(result[1], 0.9525741268224334);
    }

    #[test]
    fn test_set_params() {
        let mut layers: Vec<Layer> = vec![];
        layers.push(Layer::new(2, 2, 1.0, activator::relu()));
        layers.push(Layer::new(2, 1, 1.0, activator::linear()));
        let mut net = Net::from_layers(layers);
        net.set_params(&vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0]);

        assert_eq!(net.layers[0].w[0], vec![1.0, 1.0]);
        assert_eq!(net.layers[0].w[1], vec![1.0, 1.0]);
        assert_eq!(net.layers[0].b, vec![2.0, 2.0]);
    }
}
