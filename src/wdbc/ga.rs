//! Genictic Algorithm Utility
use rand::{
    seq::{index::sample, SliceRandom},
    Rng,
};

use crate::model::Net;

#[derive(Clone)]
pub struct Individual {
    pub chromosome: Vec<f64>,
    pub fitness: f64,
}

impl Individual {
    pub fn new(chromosome: Vec<f64>) -> Individual {
        Individual {
            chromosome,
            fitness: 0.0,
        }
    }

    pub fn set_fitness(&mut self, v: f64) {
        self.fitness = v;
    }
}

/// return result of mating of individual in the pool
pub fn mating(pop: &Vec<Individual>) -> Vec<Individual> {
    let mut rand = rand::thread_rng();
    let mut new_pop: Vec<Individual> = vec![];
    for _ in 0..pop.len() {
        let parent: Vec<_> = pop.choose_multiple(&mut rand::thread_rng(), 2).collect();
        let length = parent[0].chromosome.len();
        let indexes: Vec<_> = sample(&mut rand, length, length / 2).into_vec();

        let mut new_chromosome: Vec<f64> = vec![];
        for i in 0..length {
            if indexes.contains(&i) {
                new_chromosome.push(parent[0].chromosome[i]);
            } else {
                new_chromosome.push(parent[1].chromosome[i]);
            }
        }
        new_pop.push(Individual::new(new_chromosome));
    }
    new_pop
}

pub fn mutate(pop: &Vec<Individual>, amount: usize) -> Vec<Individual> {
    let mut new_pop: Vec<Individual> = vec![];
    let mut rand = rand::thread_rng();
    for i in 0..amount {
        let n = rand.gen_range(0..pop[i].chromosome.len() / 2);
        let indexes: Vec<_> = sample(&mut rand, pop[i].chromosome.len(), n).into_vec();
        let mut ind_clone = pop[i].clone();
        for i in indexes {
            let change = 2f64 * rand::random::<f64>() - 1f64;
            ind_clone.chromosome[i] += change;
        }
        new_pop.push(ind_clone);
    }
    new_pop
}

/// create inital population of net from layers
/// return pop and net total parameters
pub fn init_pop(net: &Net, amount: u32) -> Vec<Individual> {
    let mut pop: Vec<Individual> = vec![];
    for _ in 0..(amount) {
        let mut chromosome: Vec<f64> = vec![];
        for l in &net.layers {
            for output in &l.w {
                for _ in output {
                    // new random weight in range [-1, 1]
                    chromosome.push(2f64 * rand::random::<f64>() - 1f64);
                }
            }
            for bias in &l.b {
                chromosome.push(*bias);
            }
        }
        pop.push(Individual::new(chromosome));
    }
    pop
}

/// assign individual weigth to net
pub fn assign_ind(net: &mut Net, individual: &Individual) {
    if net.parameters != individual.chromosome.len() as u64 {
        panic!["The neural network parameters size is not equal to individual size"];
    }
    let mut idx: usize = 0;

    for l in &mut net.layers {
        for output in &mut l.w {
            for i in 0..output.len() {
                output[i] = individual.chromosome[idx];
                idx += 1;
            }
        }
        for i in 0..l.b.len() {
            l.b[i] = individual.chromosome[idx];
            idx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        activator,
        model::{self, Layer},
    };

    #[test]
    fn test_init_pop() {
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(4, 2, 1.0, activator::sigmoid()));
        layers.push(Layer::new(2, 1, 1.0, activator::sigmoid()));
        let net = Net::from_layers(layers);
        let pop = init_pop(&net, 5);

        assert_eq!(pop.len(), 5);
        assert_eq!(pop[0].chromosome.len() as u64, net.parameters);
        // check if bias is the same.
        assert_eq!(pop[0].chromosome[8], 1.0);
        assert_eq!(pop[0].chromosome[9], 1.0);
        assert_eq!(pop[0].chromosome[12], 1.0);
    }

    #[test]
    fn test_assign_ind() {
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(3, 1, 1.0, activator::sigmoid()));
        layers.push(Layer::new(1, 1, 1.0, activator::sigmoid()));
        let mut net = Net::from_layers(layers);

        let individual = Individual::new(vec![2.5, 2.3, 2.1, 1.2, 1.3, 4.0]);
        assign_ind(&mut net, &individual);

        // check if network has been mutated correctly or not.
        let mut idx = 0;
        for l in net.layers {
            for output in l.w {
                for w in output {
                    assert_eq!(w, individual.chromosome[idx]);
                    idx += 1;
                }
            }
            for b in l.b {
                assert_eq!(b, individual.chromosome[idx]);
                idx += 1;
            }
        }
    }

    #[test]
    fn test_mating_and_mutate() {
        let mut pop: Vec<Individual> = vec![];
        for i in 0..4 {
            let v = i as f64 + 1.0;
            pop.push(Individual::new(vec![v, v, v, 1.0]))
        }

        let res = mating(&pop);
        let mut_res = mutate(&pop, 4);
        assert_eq!(res.len(), pop.len());
        assert_eq!(mut_res.len(), pop.len());

        for ind in res {
            println!("{:?}", ind.chromosome);
        }
        for ind in mut_res {
            println!("{:?}", ind.chromosome);
        }
    }
}
