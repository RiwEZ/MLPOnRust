use crate::layer;
use crate::activator;

pub fn mse(output: f64, desired: f64) -> f64 {
    0.5 * (desired - output).powi(2)
}

pub fn mse_der(output: f64, desired: f64) -> f64 {
    desired - output
}

pub struct MSELoss {
    outputs: Vec<f64>,
    desired: Vec<f64>,
    loss: f64,
}

impl MSELoss { 
    pub fn criterion(outputs: Vec<f64>, desired: Vec<f64>) -> MSELoss {
        if outputs.len() != desired.len() {
            panic!("outputs size is not equal to desired size");
        }
        
        let mut loss = 0.0;
        for i in 0..outputs.len() {
            loss += mse(outputs[i], desired[i]);
        }
        loss = loss / (outputs.len() as f64);
        MSELoss {outputs, desired, loss}
    }
    
    pub fn backward(&self, layers: &mut Vec<layer::Linear>) {
        for l in (0..layers.len()).rev() {
            // output layer 
            if l == layers.len() - 1 {
                for i in 0..(layers[l].grads.len() as usize) {
                    // compute grads
                    let delta =
                        mse_der(self.outputs[i], self.desired[i]) * 
                        activator::sigmoid_der(layers[l].outputs[i]);

                    layers[l].grads[i] = delta * layers[l - 1].outputs[i];
                    layers[l].b_grads[i] = delta;
                }
                continue;
            }
            // hidden layer
            for i in 0..layers[l].grads.len() {
                let mut delta = 0f64;
                for k in 0..layers[l + 1].grads.len() {
                    delta += layers[l + 1].grads[k] * layers[l + 1].w[k][i];
                }

                layers[l].grads[i] = delta *
                    activator::sigmoid_der(layers[l].outputs[i]);
                layers[l].b_grads[i] = delta;
            }
        }
    }

    pub fn item(self) -> f64 {
        self.loss
    }
}