pub mod activator;
pub mod loss;
pub mod layer;
pub mod utills;

// Multilayer Perceptron
#[derive(Debug)]
pub struct Net {
    pub layers: Vec<layer::Linear>,
}

impl Net {
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
            for j in 0..self.layers[l].outputs.len() {
                self.layers[l].local_grads[j] = 0.0;
                for i in 0..self.layers[l].grads[j].len() {
                    self.layers[l].grads[j][i] = 0.0;
                }
            }
        }
    }

    pub fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        let mut result = self.layers[0].forward(input);
        for l in 1..self.layers.len() {
            result = self.layers[l].forward(result.clone());
        }
        result
    }

    pub fn update(&mut self, lr: f64) {
        for l in 0..self.layers.len() {
            for j in 0..self.layers[l].w.len() {
                self.layers[l].b[j] -= lr * self.layers[l].local_grads[j]; // update each neuron bias
                for i in 0..self.layers[l].w[j].len() {
                    self.layers[l].w[j][i] -= lr * self.layers[l].grads[j][i]; // update each weights
                }
            }
        }
    }
}

fn main() {  
    let mut net = Net::new(vec![2, 2, 1]);
    let lr = 0.1;
    let dataset = utills::xor_dataset();

    for j in 0..1 {
        let mut running_loss = 0.0;

        for data in dataset.get_samples() {
            net.zero_grad();
            
            let result = net.forward(data.inputs.clone());
            let loss = loss::MSELoss::criterion(result, data.labels.clone());
            loss.backward(&mut net.layers);
            
            net.update(lr);

            running_loss += loss.item();
        }
        println!("epoch: {}, loss: {}", j + 1, running_loss);
    }
    
    for l in &net.layers {
        println!("\n{:?}", l);
    }
}
