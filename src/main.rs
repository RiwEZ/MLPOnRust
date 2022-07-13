pub mod activator;
pub mod loss;
pub mod layer;

// Multilayer Perceptron
#[derive(Debug)]
pub struct Net {
    layers: Vec<layer::Linear>,
}

impl Net {
    pub fn new(architecture: Vec<u64>) -> Net {
        let mut layers: Vec<layer::Linear> = vec![];
        for i in 1..architecture.len() {
            layers.push(layer::Linear::new(architecture[i - 1], architecture[i], 1f64))
        }
        Net {layers}
    }

    pub fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.layers[0].forward(input);
        for l in 1..self.layers.len() {
            // consume copied of previous layer output
            let prev_out = self.layers[l - 1].outputs.clone();
            self.layers[l].forward(prev_out);
        }
        self.layers[self.layers.len() - 1].outputs.clone()
    }

    pub fn update(&mut self, lr: f64) {
        for l in 0..self.layers.len() {
            for j in 0..self.layers[l].w.len() {
                self.layers[l].b[j] += lr * self.layers[l].b_grads[j]; // update each neuron bias
                for i in 0..self.layers[l].w[j].len() {
                    self.layers[l].w[j][i] += lr * self.layers[l].grads[j] // update each weight
                }
            }
        }
    }
}

fn main() {  
    let mut net = Net::new(vec![2, 2, 1]);
    let lr = 0.01;
    
    for _ in 0..10000 {
        let result = net.forward(vec![0.0, 1.0]);
        let loss = loss::MSELoss::criterion(result, vec![1.0]);
        loss.backward(&mut net.layers);

        net.update(lr);

        println!("{}", loss.item());
    }

    println!("");
    for l in &net.layers {
        println!("{:?}", l);
        println!("");
    }
}
