use std::error::Error;
use plotters::prelude::*;

pub fn draw_loss(x_vec: Vec<f64>, loss_vec: Vec<f64>, valid_loss_vec: Vec<f64>, path: String) -> Result<(), Box<dyn Error>> {
    let max_loss1 = loss_vec.iter().fold(0.0f64, |max, &val| if val > max {val} else {max});
    let max_loss2= valid_loss_vec.iter().fold(0.0f64, |max, &val| if val > max {val} else {max});
    let max_loss = if max_loss1 > max_loss2 {max_loss1} else {max_loss2};

    let max_x = x_vec.iter().fold(0.0f64, |max, &val| if val > max {val} else {max});
    let min_x = x_vec.iter().fold(0.0f64, |max, &val| if val < max {val} else {max});

    // plotting loss
    let root = BitMapBackend::new(&path, (1024, 768))
        .into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("loss", ("san-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(min_x..max_x, 0f64..max_loss)?;
    
    chart.configure_mesh().draw()?;

    let mut i: usize = 0;
    chart
        .draw_series(LineSeries::new(loss_vec.into_iter().map(|x| {i += 1; (x_vec[i - 1], x)}), &RED))?
        .label("training_loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    i = 0;
    chart
        .draw_series(LineSeries::new(valid_loss_vec.into_iter().map(|x| {i += 1; (x_vec[i - 1], x)}), &BLUE))?
        .label("validation_loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}