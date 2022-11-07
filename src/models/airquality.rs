use std::time::Instant;

use crate::{
    activator, loss,
    mlp::{Layer, Net},
    swarm::{self, gen_rho},
    utills::{data, graph, io},
};

const IMGPATH: &str = "img";

pub fn air_8_4_1() {
    fn model() -> Net {
        let mut layers: Vec<Layer> = vec![];
        layers.push(Layer::new(8, 8, 1.0, activator::relu()));
        layers.push(Layer::new(8, 1, 1.0, activator::linear()));
        Net::from_layers(layers)
    }
    air_particle_swarm(&model);
}

pub fn air_particle_swarm(model: &dyn Fn() -> Net) {
    let (dataset_five, dataset_ten) =
        data::airquality_dataset().expect("Something wrong with airquality_dataset");
    let mut loss = loss::Loss::abs_err();
    let max_epoch = 200;

    let mut train_proc: Vec<Vec<(i32, f64)>> = Vec::with_capacity(10);
    for _ in 0..10 {
        train_proc.push(vec![]);
    }

    let start = Instant::now();
    // five day predictor
    for (j, dt) in dataset_five.cross_valid_set(0.1).iter().enumerate() {
        if j > 0 {
            break;
        }

        let (training_set, validation_set) = dt.0.minmax_norm(&dt.1);

        let mut net = model();
        //let mut particles = swarm::init_particles(&net, 20);
        //let mut gbest = particles[0].clone();

        let mut groups = swarm::init_particles_group(&net, 20, 4);

        for i in 0..max_epoch {
            for (k, g) in groups.iter_mut().enumerate() {
                for (_, x) in g.particles.iter_mut().enumerate() {
                    net.set_params(&x.position);
                    let mut run_loss = 0.0;
                    for data in training_set.get_shuffled() {
                        let result = net.forward(&data.inputs);
                        run_loss += loss.criterion(&result, &data.labels);
                    }
                    let mae = run_loss / training_set.len() as f64; // Mean Absolute Error, F(x_i(t))
                    if mae < x.f {
                        x.f = mae;
                        x.best_pos = x.position.clone();
                    }
                    if mae < g.lbest.f {
                        g.lbest.f = mae; // set gbest
                        g.lbest.best_pos = x.position.clone();
                    }
                    x.update_speed(&g.lbest, gen_rho(1.0), gen_rho(1.5));
                    x.change_pos();
                    train_proc[j].push((i, x.f));
                }
                println!("{}, {} lbest : {:.5e}", k, i, g.lbest.f);
            }
        }

        let gbest = &groups
            .iter()
            .reduce(|best, x| if best.lbest.f < x.lbest.f { best } else { x })
            .unwrap()
            .lbest;

        net.set_params(&gbest.position);
        io::save(&net.layers, "models/air/air-8-4-1.json".into()).unwrap();

        let mut run_loss = 0.0;
        for data in validation_set.get_datas() {
            let result = net.forward(&data.inputs);
            let loss = loss.criterion(&result, &data.labels);
            run_loss += loss;
            println!(
                "predict: {:.3}, real: {:.3}, loss: {:.3}",
                result[0], &data.labels[0], loss
            );
        }
        let mae = run_loss / validation_set.len() as f64;
        println!("validation MAE : {:.5}, with gbest f = {}", mae, gbest.f);
    }
    let duration = start.elapsed();
    println!("Time used: {:.3} sec", duration.as_secs_f32());
    graph::draw_ga_progress(&train_proc, format!("img/train_proc.png"), 12.0).unwrap();
}
