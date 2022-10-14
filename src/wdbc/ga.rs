//! Genictic Algorithm Utility 
use crate::model::{Layer, Net};

/// create inital population of net from layers 
/// return pop and net total parameters
pub fn init_pop(net: &Net, amount: u32) -> Vec<Vec<f64>> {
    let mut pop: Vec<Vec<f64>> = vec![];
    for _ in 0..(amount) {
        let mut individual: Vec<f64> = vec![];
        for l in &net.layers {
            for output in &l.w {
                for _ in output {
                    // new random weight in range [-1, 1] 
                    individual.push(2f64 * rand::random::<f64>() - 1f64);
                }
            }
            for bias in &l.b {
                individual.push(*bias);
            }
        }
        pop.push(individual);
    }
    pop
}

/// assign individual weigth to net
pub fn assign_ind(net: &mut Net, individual: &Vec<f64>) {
    if net.parameters != individual.len() as u64 {
        panic!["The neural network parameters size is not equal to individual size"];
    }
    let mut idx: usize = 0;

    for l in &mut net.layers {
        for output in &mut l.w {
            for i in 0..output.len() {
                output[i] = individual[idx];
                idx += 1;
            }
        }
        for i in 0..l.b.len() {
            l.b[i] = individual[idx];
            idx += 1;
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::{model, activator};
    use super::*;

    #[test]
    fn test_init_pop() {
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(4, 2, 1.0, activator::sigmoid()));
        layers.push(Layer::new(2, 1, 1.0, activator::sigmoid()));
        let net = Net::from_layers(layers);
        let pop = init_pop(&net, 5);
        
        assert_eq!(pop.len(), 5);
        assert_eq!(pop[0].len() as u64, net.parameters);
        // check if bias is the same.
        assert_eq!(pop[0][8], 1.0);
        assert_eq!(pop[0][9], 1.0);
        assert_eq!(pop[0][12], 1.0);
    }

    #[test]
    fn test_assign_ind() {
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(3, 1, 1.0, activator::sigmoid()));
        layers.push(Layer::new(1, 1, 1.0, activator::sigmoid()));
        let mut net = Net::from_layers(layers);

        let individual: Vec<f64> = vec![2.5, 2.3, 2.1, 1.2, 1.3, 4.0];
        assign_ind(&mut net, &individual);

        // check if network has been mutated correctly or not.
        let mut idx = 0;
        for l in net.layers {
            for output in l.w {
                for w in output {
                    assert_eq!(w, individual[idx]);
                    idx += 1;
                }
            }
            for b in l.b {
                assert_eq!(b, individual[idx]);
                idx += 1;
            }
        }
    }
}
