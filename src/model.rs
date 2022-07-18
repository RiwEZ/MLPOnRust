use crate::layer;
use crate::activator;

#[derive(Debug)]
pub struct Net {
    pub layers: Vec<layer::Linear>,
}

impl Net {
    pub fn from_layers(layers: Vec<layer::Linear>) -> Net {
        Net { layers }
    }

    pub fn new(architecture: Vec<u64>) -> Net {
        let mut layers: Vec<layer::Linear> = vec![];
        for i in 1..architecture.len() {
            layers.push(
                layer::Linear::new(
                    architecture[i - 1], 
                    architecture[i], 
                    1f64, activator::sigmoid()))
        }
        Net {layers}
    }

    pub fn zero_grad(&mut self) {
        for l in 0..self.layers.len() {
            self.layers[l].zero_grad();
        }
    }

    pub fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        let mut result = self.layers[0].forward(input);
        for l in 1..self.layers.len() {
            result = self.layers[l].forward(result.clone());
        }
        result
    }

    pub fn update(&mut self, lr: f64, momentum: f64) {
        for l in 0..self.layers.len() {
            self.layers[l].update(lr, momentum);
        }
    }
}