use crate::layer;

pub fn mse(output: f64, desired: f64) -> f64 {
    0.5 * (output - desired).powi(2)
}

pub fn mse_der(output: f64, desired: f64) -> f64 {
    output - desired
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
                for j in 0..layers[l].outputs.len() {
                    // compute grads
                    let local_grad =
                        mse_der(self.outputs[j], self.desired[j]) * 
                        (layers[l].act.der)(layers[l].outputs[j]);
                    
                    // set grads for each weight
                    for k in 0..(layers[l - 1].outputs.len()) {
                        layers[l].grads[j][k] = local_grad * (layers[l - 1].act.func)(layers[l - 1].outputs[k]);
                    }
                    //println!("{}, {}: {:?}", l, j, layers[l].grads[j]);
                    layers[l].local_grads[j] = local_grad;
                }
                continue;
            }
            // hidden layer
            for j in 0..layers[l].outputs.len() {
                let mut local_grad = 0f64;
                // calculate local_grad based on previous local_grad
                for i in 0..layers[l + 1].w.len() {
                    for k in 0..layers[l + 1].w[i].len() {
                        local_grad += layers[l + 1].w[i][k] * layers[l + 1].local_grads[i];
                    }
                }

                local_grad = (layers[l].act.der)(layers[l].outputs[j]) * local_grad; 
                
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
                
                //println!("{}, {}: {:?}", l, j, layers[l].grads[j]);
                layers[l].local_grads[j] = local_grad;
            }
        }
    }

    pub fn item(self) -> f64 {
        self.loss
    }
}