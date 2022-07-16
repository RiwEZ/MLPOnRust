pub mod activator;
pub mod loss;
pub mod layer;
pub mod utills;
pub mod model;
pub mod io;
use plotters::prelude::*;
use std::error::Error;

fn draw_loss(loss_vec: Vec<f64>) -> Result<(), Box<dyn Error>> {
    // plotting loss
    let root = BitMapBackend::new("img/0.png", (640, 480))
        .into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("loss", ("san-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0i64..loss_vec.len() as i64, 0f64..1f64)?;
    
    chart.configure_mesh().draw()?;

    let mut i: i64 = -1;
    chart
        .draw_series(LineSeries::new(loss_vec.into_iter().map(|x| {i += 1; (i, x)}), &RED))?
        .label("loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    /*
    let mut net = model::Net::new(vec![2, 2, 1]);
    let lr = 0.1;
    let dataset = utills::xor_dataset();
    let mut loss = loss::MSELoss::new();
    let mut loss_vec: Vec<f64> = vec![];

    for _ in 0..5000 {
        let mut running_loss = 0.0;

        for data in dataset.get_samples() {
            net.zero_grad();
            
            let result = net.forward(data.inputs.clone());
            loss.criterion(result, data.labels.clone());
            loss.backward(&mut net.layers);
            
            net.update(lr);

            running_loss += loss.item();
        }
        loss_vec.push(running_loss);
    }
    println!("epoch: {}, loss: {}", loss_vec.len(), loss_vec[loss_vec.len() - 1]);
    
    println!("\n{}", (net.forward(vec![0.0, 0.0])[0] > 0.5) );
    println!("\n{}", (net.forward(vec![1.0, 0.0])[0] > 0.5) );
    println!("\n{}", (net.forward(vec![0.0, 1.0])[0] > 0.5) );
    println!("\n{}", (net.forward(vec![1.0, 1.0])[0] > 0.5) );  

    io::save(&net.layers, "models/xor.json".to_string())?;
    draw_loss(loss_vec)?;
    */

    let mut net = io::load("models/xor.json".to_string())?;
    println!("\n{}", (net.forward(vec![0.0, 0.0])[0] > 0.5) );
    println!("\n{}", (net.forward(vec![1.0, 0.0])[0] > 0.5) );
    println!("\n{}", (net.forward(vec![0.0, 1.0])[0] > 0.5) );
    println!("\n{}", (net.forward(vec![1.0, 1.0])[0] > 0.5) );  
    
    Ok(())
}
