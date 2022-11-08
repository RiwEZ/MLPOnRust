use std::time::Instant;

use crate::{
    activator, loss,
    mlp::{Layer, Net},
    swarm::{self, gen_rho},
    utills::{
        data::{self, mean},
        graph, io,
    },
};

const IMGPATH: &str = "img";

pub fn air_8_4_1() {
    fn model() -> Net {
        let mut layers: Vec<Layer> = vec![];
        layers.push(Layer::new(8, 4, 1.0, activator::relu()));
        layers.push(Layer::new(4, 1, 1.0, activator::linear()));
        Net::from_layers(layers)
    }
    air_particle_swarm(&model);
}

pub fn air_particle_swarm(model: &dyn Fn() -> Net) {
    let (dataset_five, dataset_ten) =
        data::airquality_dataset().expect("Something wrong with airquality_dataset");
    let mut loss = loss::Loss::abs_err();
    let max_epoch = 100;

    let mut train_proc: Vec<Vec<(i32, f64)>> = (0..10).into_iter().map(|_| vec![]).collect();

    let start = Instant::now();
    // five day predictor
    for (j, dt) in dataset_five.cross_valid_set(0.1).iter().enumerate() {
        if j > 0 {
            break;
        }

        let (training_set, validation_set) = dt.0.minmax_norm(&dt.1);

        let mut net = model();
        let mut groups = swarm::init_particles_group(&net, 5, 4);

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
                    if mae < g.lbest_f {
                        g.lbest_f = mae; // set gbest
                        g.lbest_pos = x.position.clone();
                    }
                    x.update_speed(&g.lbest_pos, gen_rho(1.0), gen_rho(1.5));
                    x.change_pos();
                    train_proc[j].push((i, x.f));
                }
                println!("{}, {} lbest : {:.5e}", k, i, g.lbest_f);
            }
        }

        let best_group = &groups
            .iter()
            .reduce(|best, x| if best.lbest_f < x.lbest_f { best } else { x })
            .unwrap();

        let gbest = best_group
            .particles
            .iter()
            .reduce(|best, ind| if best.f < ind.f { best } else { ind })
            .unwrap();

        net.set_params(&gbest.best_pos);
        io::save(&net.layers, "models/air/air-8-4-1.json".into()).unwrap();

        let mut mae = 0.0;
        let mut mse = 0.0;
        let mut rscore_divider = 0.0;
        let v_mean = mean(&validation_set.get_label(0));
        let mut mape = 0.0;

        for data in validation_set.get_datas() {
            let result = net.forward(&data.inputs);

            let abs_err = loss.criterion(&result, &data.labels);
            mae += abs_err;
            mape += abs_err / data.labels[0];
            let sqr_err = loss::Loss::square_err().criterion(&result, &data.labels);
            mse += sqr_err;
            rscore_divider += (data.labels[0] - v_mean).powi(2);

            println!(
                "predict: {:.3}, real: {:.3}, loss: {:.3}",
                result[0], &data.labels[0], abs_err
            );
        }
        let rscore = 1.0 - (mse / rscore_divider);
        mape = mape / validation_set.len() as f64;
        mae = mae / validation_set.len() as f64;
        mse = mse / validation_set.len() as f64;
        println!(
            "validation MAE : {:.3}, rmse: {:.3}, rscore: {:.3}, mape: {:.3}, with gbest f = {:.3}",
            mae,
            mse.sqrt(),
            rscore,
            mape,
            gbest.f
        );
    }
    let duration = start.elapsed();
    println!("Time used: {:.3} sec", duration.as_secs_f32());
    graph::draw_ga_progress(&train_proc, format!("img/train_proc.png"), 12.0).unwrap();
}
