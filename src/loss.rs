use crate::layer;

pub struct MSELoss {
    outputs: Vec<f64>,
    desired: Vec<f64>,
    loss: f64,
}

impl MSELoss { 
    pub fn new() -> MSELoss {
        MSELoss { outputs: vec![], desired: vec![], loss: 0.0 }
    }

    fn func(output: f64, desired: f64) -> f64 {
        0.5 * (output - desired).powi(2)
    }

    fn der(output: f64, desired: f64) -> f64 {
        output - desired
    }

    pub fn item(&self) -> f64 {
        self.loss
    }

    pub fn criterion(&mut self, outputs: Vec<f64>, desired: Vec<f64>) {
        if outputs.len() != desired.len() {
            panic!("outputs size is not equal to desired size");
        }
        
        let mut loss = 0.0;
        for i in 0..outputs.len() {
            loss += MSELoss::func(outputs[i], desired[i]);
        }
        self.loss = loss / (outputs.len() as f64);
        self.outputs = outputs;
        self.desired = desired;
    }
    
    pub fn backward(&self, layers: &mut Vec<layer::Linear>) {
        for l in (0..layers.len()).rev() {
            // output layer 
            if l == layers.len() - 1 {
                for j in 0..layers[l].outputs.len() {
                    // compute grads
                    let local_grad =
                        MSELoss::der(self.outputs[j], self.desired[j]) * 
                        (layers[l].act.der)(layers[l].outputs[j]);
                    
                    layers[l].prev_local_grads = layers[l].local_grads.clone(); // copied previous grad before update
                    layers[l].local_grads[j] = local_grad;
                    
                    layers[l].prev_grads = layers[l].grads.clone();
                    // set grads for each weight
                    for k in 0..(layers[l - 1].outputs.len()) {
                        layers[l].grads[j][k] = 
                            (layers[l - 1].act.func)(layers[l - 1].outputs[k]) *
                            local_grad;
                    }
                }
                continue;
            }
            // hidden layer
            for j in 0..layers[l].outputs.len() {
                // calculate local_grad based on previous local_grad
                let mut local_grad = 0f64;
                for i in 0..layers[l + 1].w.len() {
                    for k in 0..layers[l + 1].w[i].len() {
                        local_grad += layers[l + 1].w[i][k] * layers[l + 1].local_grads[i];
                    }
                }
                local_grad = (layers[l].act.der)(layers[l].outputs[j]) * local_grad; 

                layers[l].prev_local_grads = layers[l].local_grads.clone(); // copied previous grad before update
                layers[l].local_grads[j] = local_grad;
                

                layers[l].prev_grads = layers[l].grads.clone();
                // set grads for each weight
                if l == 0 {
                    for k in 0..layers[l].inputs.len() {
                        layers[l].grads[j][k] = layers[l].inputs[k] * local_grad;
                    }
                }
                else {
                    for k in 0..layers[l - 1].outputs.len() {
                        layers[l].grads[j][k] = 
                            (layers[l].act.func)(layers[l - 1].outputs[k]) * // a(l - 1)
                            local_grad;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_func() {
        assert_eq!(MSELoss::func(2.0, 1.0), 0.5);
        assert_eq!(MSELoss::func(5.0, 0.0), 12.5);
    }

    #[test]
    fn test_mse_der() {
        assert_eq!(MSELoss::der(2.0, 1.0), 1.0);
        assert_eq!(MSELoss::der(5.0, 0.0), 5.0);
    }
    
    #[test] 
    fn test_mse() {
        let mut loss = MSELoss::new();

        loss.criterion(vec![2.0, 1.0, 0.0], vec![0.0, 1.0, 2.0]);
        assert_eq!(loss.item(), 4.0/3.0);

        loss.criterion(
            vec![34.0, 37.0, 44.0, 47.0, 48.0], 
            vec![37.0, 40.0, 46.0, 44.0, 46.0]);
        assert_eq!(loss.item(), 3.5);
    }
}