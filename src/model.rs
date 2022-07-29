use crate::activator;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[derive(Debug)]
pub struct Layer {
    pub inputs: Array1<f64>,
    pub outputs: Array1<f64>, // need to save this for backward pass
    pub w: Array2<f64>,
    pub b: Array1<f64>,
    pub grads: Array2<f64>,
    pub prev_grads: Array2<f64>,
    pub local_grads: Array1<f64>,
    pub prev_local_grads: Array1<f64>,
    pub act: activator::ActivationContainer,
}

impl Layer {
    pub fn new(
        input_features: usize,
        output_features: usize,
        bias: f64,
        act: activator::ActivationContainer,
    ) -> Layer {
        // initialize weights matrix
        let w = Array::random((output_features, input_features), Uniform::new(-1.0, 1.0));
        let b = Array::<f64, _>::from_elem(output_features, bias);
        let inputs = Array::<f64, _>::zeros(input_features);
        let outputs = Array::<f64, _>::zeros(output_features);

        let local_grads = Array1::zeros(output_features);
        let prev_local_grads = Array1::zeros(output_features);

        let grads = Array2::zeros((output_features, input_features));
        let prev_grads = Array2::zeros((output_features, input_features));

        Layer {
            inputs,
            outputs,
            w,
            b,
            grads,
            prev_grads,
            local_grads,
            prev_local_grads,
            act,
        }
    }

    pub fn forward(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        if inputs.len() != self.inputs.len() {
            panic!("forward: input size is wrong");
        }
        let result = &inputs.dot(&self.w.t()) + &self.b;
        self.outputs = result.clone();
        self.inputs = inputs.clone();
        result.map(|x| (self.act.func)(*x))
    }

    pub fn update(&mut self, lr: f64, momentum: f64) {
        /*         for j in 0..self.w.len() {
            self.b[j] -= momentum * self.prev_local_grads[j] + lr * self.local_grads[j]; // update each neuron bias
            for i in 0..self.w[j].len() {
                self.w[j][i] -= momentum * self.prev_grads[j][i] + lr * self.grads[j][i];
                // update each weights
            }
        } */
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

    pub fn new(architecture: Vec<usize>) -> Net {
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

    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
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
        /*         let linear = Layer::new(2, 3, 1.0, activator::linear());
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
        assert_eq!(linear.prev_local_grads.len(), 3); */
    }

    #[test]
    fn test_linear_forward1() {
        let mut linear = Layer::new(2, 1, 1.0, activator::sigmoid());
        linear.w = linear.w.map(|_| 1.0);

        assert_eq!(linear.forward(&arr1(&[1.0, 1.0]))[0], 0.9525741268224334);
        assert_eq!(linear.outputs[0], 3.0);
    }

    #[test]
    fn test_linear_forward2() {
        let mut linear = Layer::new(2, 2, 1.0, activator::sigmoid());

        for j in 0..2 {
            linear.w[[j, 0]] = (j as f64) + 1.0;
            linear.w[[j, 1]] = (j as f64) + 1.0;
        }

        let result = linear.forward(&arr1(&[0.0, 1.0]));
        assert_eq!(result[0], 0.8807970779778823);
        assert_eq!(result[1], 0.9525741268224334);
        assert_eq!(linear.outputs[0], 2.0);
        assert_eq!(linear.outputs[1], 3.0);
    }
}
