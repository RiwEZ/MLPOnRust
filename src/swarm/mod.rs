use rand::{distributions::Uniform, prelude::Distribution};

use crate::mlp::Net;

#[derive(Debug, Clone)]
pub struct Individual {
    pub best_pos: Vec<f64>,
    pub position: Vec<f64>,
    pub f: f64, // evaluation of this individual
    pub speed: Vec<f64>,
}

impl Individual {
    pub fn new(position: Vec<f64>) -> Individual {
        let mut rand = rand::thread_rng();
        let dist = Uniform::from(-1.0..=1.0);
        let speed: Vec<f64> = position.iter().map(|_i| dist.sample(&mut rand)).collect();
        Individual {
            best_pos: position.clone(),
            position,
            f: f64::MAX,
            speed,
        }
    }

    /// Individual best speed updater
    pub fn ind_update_speed(&mut self, rho: f64) {
        self.speed
            .iter_mut()
            .zip(self.best_pos.iter().zip(self.position.iter()))
            .for_each(|(v, (x_b, x))| {
                *v = *v + rho * (*x_b - *x);
            });
    }

    /// Speed updator with social component included
    pub fn update_speed(&mut self, other_best: &Vec<f64>, rho1: f64, rho2: f64) {
        let w = 1.0;
        self.speed
            .iter_mut()
            .zip(
                self.position
                    .iter()
                    .zip(self.best_pos.iter().zip(other_best.iter())),
            )
            .for_each(|(v, (x, (x_b, x_gb)))| {
                *v = w * *v + rho1 * (*x_b - *x) + rho2 * (*x_gb - *x);
            });
    }

    pub fn change_pos(&mut self) {
        self.position
            .iter_mut()
            .zip(self.speed.iter())
            .for_each(|(x, v)| {
                *x = *x + v;
            });
    }
}

pub fn gen_rho(c: f64) -> f64 {
    let mut rand = rand::thread_rng();
    let dist = Uniform::from(0.0..=1.0);
    dist.sample(&mut rand) * c
}

/// Create inital particles of MLP from layers
///
/// return: particles
pub fn init_particles(net: &Net, amount: u32) -> Vec<Individual> {
    let mut inidividuals: Vec<Individual> = vec![];
    for _ in 0..amount {
        let mut position: Vec<f64> = Vec::with_capacity(net.parameters as usize);
        for l in net.layers.iter() {
            for output in l.w.iter() {
                for _ in output.iter() {
                    // new random weight in range [-1, 1]
                    position.push(2f64 * rand::random::<f64>() - 1f64);
                }
            }
            for bias in l.b.iter() {
                position.push(*bias);
            }
        }
        inidividuals.push(Individual::new(position));
    }
    inidividuals
}

pub struct IndividualGroup {
    pub particles: Vec<Individual>,
    pub lbest_f: f64,
    pub lbest_pos: Vec<f64>,
}

impl IndividualGroup {
    pub fn add(&mut self, individual: Individual) {
        self.particles.push(individual);
    }
}

pub fn init_particles_group(net: &Net, group: usize, group_size: u32) -> Vec<IndividualGroup> {
    (0..group)
        .into_iter()
        .map(|_| {
            let particles = init_particles(&net, group_size + 1);
            IndividualGroup {
                particles: particles[1..].into(),
                lbest_f: f64::MAX,
                lbest_pos: particles[0].position.clone(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{activator, mlp::Layer};

    use super::*;

    #[test]
    fn test_update_speed() {
        fn f(pos: &Vec<f64>) -> f64 {
            pos[0].powi(2) + 2.0 * pos[1]
        }

        let mut p1 = Individual::new(vec![1.0, 1.0]);
        p1.f = 4.0;
        p1.speed = vec![0.5, 0.5];

        let gbest = vec![0.5, 1.0];

        // trainning
        let eval_result = f(&p1.position);
        if eval_result < p1.f {
            p1.f = eval_result;
            p1.best_pos = p1.position.clone();
        }

        p1.update_speed(&gbest, 1.0, 1.0);
        p1.change_pos();

        assert_eq!(p1.speed, vec![0.0, 0.5]);
        assert_eq!(p1.position, vec![1.0, 1.5]);
    }

    #[test]
    fn test_split() {
        let mut layers: Vec<Layer> = vec![];
        layers.push(Layer::new(4, 2, 1.0, activator::sigmoid()));
        layers.push(Layer::new(2, 1, 1.0, activator::sigmoid()));
        let net = Net::from_layers(layers);

        let groups = init_particles_group(&net, 3, 3);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[2].particles.len(), 3);

        let groups = init_particles_group(&net, 2, 5);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].particles.len(), 5);
    }
}
