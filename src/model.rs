use crate::activator;

#[derive(Debug)]
pub struct Layer {
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>, // need to save this for backward pass
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub grads: Vec<Vec<f64>>,
    pub prev_grads: Vec<Vec<f64>>,
    pub local_grads: Vec<f64>,
    pub prev_local_grads: Vec<f64>,
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
        let mut prev_grads: Vec<Vec<f64>> = vec![];
        let mut prev_local_grads: Vec<f64> = vec![];
        let mut b: Vec<f64> = vec![];

        for _ in 0..output_features {
            outputs.push(0.0);
            local_grads.push(0.0);
            prev_local_grads.push(0.0);
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
            prev_grads.push(g);
        }
        Layer {
            inputs,
            outputs,
            w: weights,
            b,
            grads,
            prev_grads,
            local_grads,
            prev_local_grads,
            act,
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
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
        self.inputs = inputs.clone();
        result
    }

    pub fn update(&mut self, lr: f64, momentum: f64) {
        for j in 0..self.w.len() {
            self.b[j] -= momentum * self.prev_local_grads[j] + lr * self.local_grads[j]; // update each neuron bias
            for i in 0..self.w[j].len() {
                self.w[j][i] -= momentum * self.prev_grads[j][i] + lr * self.grads[j][i];
                // update each weights
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
}

impl Net {
    /// NeuralNetwork
    /// # Examples
    /// ```
    /// fn main() -> Result<(), Box<dyn Error>> {
    ///    let mut net = model::Net::new(vec![2, 2, 1]);
    ///    let lr = 0.01;
    ///
    ///    let dataset = utills::xor_dataset();
    ///
    ///    let mut loss = loss::MSELoss::new();
    ///    let mut loss_vec: Vec<f64> = vec![];
    ///
    ///    for _ in 0..5000 {
    ///        let mut running_loss = 0.0;
    ///
    ///        for data in dataset.get_shuffled() {
    ///            let result = net.forward(data.inputs.clone());
    ///            loss.criterion(result, data.labels.clone());
    ///            loss.backward(&mut net.layers);
    ///            
    ///            net.update(lr, 1.0);
    ///
    ///            running_loss += loss.item();
    ///        }
    ///        loss_vec.push(running_loss);
    ///    }
    ///    println!("epoch: {}, loss: {}", loss_vec.len(), loss_vec[loss_vec.len() - 1]);
    ///    
    ///    println!("\n{}", (net.forward(vec![0.0, 0.0])[0] > 0.5) );
    ///    println!("\n{}", (net.forward(vec![1.0, 0.0])[0] > 0.5) );
    ///    println!("\n{}", (net.forward(vec![0.0, 1.0])[0] > 0.5) );
    ///    println!("\n{}", (net.forward(vec![1.0, 1.0])[0] > 0.5) );  
    ///
    ///    io::save(&net.layers, "models/xor.json".to_string())?;
    ///    utills::draw_loss(loss_vec, "img/2.png".to_string())?;
    ///    Ok(())
    /// }
    /// ```
    pub fn from_layers(layers: Vec<Layer>) -> Net {
        Net { layers }
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
        Net { layers }
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
        assert_eq!(linear.prev_grads.len(), 3);
        assert_eq!(linear.grads[0].len(), 2);
        assert_eq!(linear.prev_grads[0].len(), 2);
        assert_eq!(linear.local_grads.len(), 3);
        assert_eq!(linear.prev_local_grads.len(), 3);
    }

    #[test]
    fn test_linear_forward1() {
        let mut linear = Layer::new(2, 1, 1.0, activator::sigmoid());

        for j in 0..linear.w.len() {
            for i in 0..linear.w[j].len() {
                linear.w[j][i] = 1.0;
            }
        }

        assert_eq!(linear.forward(&vec![1.0, 1.0])[0], 0.982013790037908442);
        assert_eq!(linear.outputs[0], 4.0);
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
        assert_eq!(result[0], 0.9525741268224334);
        assert_eq!(result[1], 0.9820137900379084);
        assert_eq!(linear.outputs[0], 3.0);
        assert_eq!(linear.outputs[1], 4.0);
    }
}
