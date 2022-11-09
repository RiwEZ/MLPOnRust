use std::time::Instant;

use crate::{
    activator, loss,
    mlp::{Layer, Net},
    swarm::{self, gen_rho},
    utills::{
        data::{self, DataSet},
        graph,
    },
};

const IMGPATH: &str = "report/assignment_4/images";

pub fn air_8_4_1() {
    fn model() -> Net {
        let mut layers: Vec<Layer> = vec![];
        layers.push(Layer::new(8, 4, 1.0, activator::relu()));
        layers.push(Layer::new(4, 1, 1.0, activator::linear()));
        Net::from_layers(layers)
    }
    air_particle_swarm(&model, "air-8-4-1");
}

pub fn air_8_1_1() {
    fn model() -> Net {
        let mut layers: Vec<Layer> = vec![];
        layers.push(Layer::new(8, 1, 1.0, activator::relu()));
        layers.push(Layer::new(1, 1, 1.0, activator::linear()));
        Net::from_layers(layers)
    }
    air_particle_swarm(&model, "air-8-1-1");
}

pub fn air_8_8_4_1() {
    fn model() -> Net {
        let mut layers: Vec<Layer> = vec![];
        layers.push(Layer::new(8, 8, 1.0, activator::relu()));
        layers.push(Layer::new(8, 4, 1.0, activator::relu()));
        layers.push(Layer::new(4, 1, 1.0, activator::linear()));
        Net::from_layers(layers)
    }
    air_particle_swarm(&model, "air-8-8-4-1");
}

pub fn validation_test(net: &mut Net, validation_set: &DataSet, training_set: &DataSet) -> (f64, f64) {
    let mut mae = 0.0;
    for data in validation_set.get_datas() {
        let result = net.forward(&data.inputs);
        let abs_err = loss::Loss::abs_err().criterion(&result, &data.labels);
        mae += abs_err;
    }
    mae = mae / validation_set.len() as f64;

    let mut t_mae = 0.0;
    for data in training_set.get_datas() {
        let result = net.forward(&data.inputs);
        let abs_err = loss::Loss::abs_err().criterion(&result, &data.labels);
        t_mae += abs_err;
    }
    (mae, t_mae/training_set.len() as f64)
}

pub fn pso_fit(model: &dyn Fn() -> Net, dataset: &DataSet, folder: String) -> f32 {
    let mut loss = loss::Loss::abs_err();
    let max_epoch = 100;
    let mut train_proc: Vec<Vec<(i32, f64)>> = (0..10).into_iter().map(|_| vec![]).collect();
    let mut valid_mae: Vec<f64> = vec![];
    let mut train_mae: Vec<f64> = vec![];

    let start = Instant::now();
    for (j, dt) in dataset.cross_valid_set(0.1).iter().enumerate() {
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
        //io::save(&net.layers, "models/air/air-8-4-1.json".into()).unwrap();
        let (v_mae, t_mae) = validation_test(&mut net, &validation_set, &training_set);   
        valid_mae.push(v_mae);
        train_mae.push(t_mae);
    }

    let duration = start.elapsed();
    
    graph::draw_ga_progress(
        &train_proc,
        format!("{}/{}/train_proc.png", IMGPATH, folder),
        12.0,
    )
    .unwrap();

    graph::hist::draw_2hist(
        [&valid_mae, &train_mae],
        "Validation/Training MAE",
        ("Iteration", "Validation/Training MAE"),
        format!("{}/{}/mae.png", IMGPATH, folder),
    ).unwrap();

    duration.as_secs_f32()
}

pub fn air_particle_swarm(model: &dyn Fn() -> Net, folder: &str) {
    let (dataset_five, dataset_ten) =
        data::airquality_dataset().expect("Something wrong with airquality_dataset");

    let t1 = pso_fit(model, &dataset_five, format!("5days/{}", folder));
    let t2 = pso_fit(model, &dataset_ten, format!("10days/{}", folder));

    println!("t1: {:.3} sec, t2: {:.3} sec", t1, t2);
}

