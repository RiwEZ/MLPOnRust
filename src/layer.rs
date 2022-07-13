use crate::activator;

#[derive(Debug)]
pub struct Linear {
    pub inputs: u64,
    pub outputs: Vec<f64>, // need to save this for backward pass
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub grads: Vec<f64>,
    pub b_grads: Vec<f64>
}

impl Linear {
    pub fn new(input_features: u64, output_features: u64, bias: f64) -> Linear {
        // initialize weights matrix
        let mut weights: Vec<Vec<f64>> = vec![];
        let mut outputs: Vec<f64> = vec![];
        let mut grads: Vec<f64> = vec![];
        let mut b_grads: Vec<f64> = vec![];
        let mut b: Vec<f64> = vec![];

        for _ in 0..output_features {
            grads.push(0.0);
            outputs.push(0.0);
            b_grads.push(0.0);
            b.push(bias);           

            let mut w: Vec<f64> = vec![];
            for _ in 0..input_features {
                // random both positive and negative weight
                w.push(2f64 * rand::random::<f64>() - 1f64);
            }
            weights.push(w)
        }
        Linear { inputs: input_features, outputs, w: weights, b, grads, b_grads }
    }

    pub fn forward(&mut self, input: Vec<f64>) {
        if input.len() as u64 != self.inputs {
            panic!("forward: input size is wrong");
        }
        
        for j in 0..self.outputs.len() {
            let mut sum: f64 = 0.0;
            // loop through input and add w*x + b to sum
            for i in 0..input.len() {
                sum += self.w[j as usize][i] * input[i] + self.b[j]
            }
            self.outputs[j] = activator::sigmoid(sum); // this could be more generic
        }
    } 
}