use std::error::Error;
use plotters::prelude::*;

/// Draw training loss and validation loss at each epoch (x_vec)
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

    chart
        .draw_series(LineSeries::new(loss_vec.iter().enumerate().map(|(i, x)| {(x_vec[i], *x)}), &RED))?
        .label("training_loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(valid_loss_vec.iter().enumerate().map(|(i, x)| {(x_vec[i], *x)}), &BLUE))?
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

/// Draw cross validation score of given data (validation score for each iteraion)
pub fn draw_cv_scores(data: Vec<f64>, path: String) -> Result<(), Box<dyn Error>> {
    let n = data.len();
    let max_y = data.iter().fold(0.0f64, |max, &val| if val > max {val} else {max}) + 0.1;
    let mean = data.iter().fold(0.0f64, |mean, &val| {mean + val/n as f64});

    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Historgram", ("san-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d((1..n).into_segmented(), 0.0..max_y)?
        .set_secondary_coord(1..n, 0.0..max_y);
        
    chart.configure_mesh()
        .disable_x_mesh()
        .x_label_style(("san-serif, 14").into_font())
        .y_label_style(("san-serif, 14").into_font())
        .y_desc("validation loss") 
        .x_desc("iterations") 
        .axis_desc_style(("sans-serif", 20))
        .draw()?;
    
    let hist = Histogram::vertical(&chart)
        .style(RED.mix(0.5).filled())
        .margin(10)
        .data(data.iter().enumerate().map(|(i, x)| {(i + 1, *x)}));

    chart.draw_series(hist)?;

    chart.draw_secondary_series(
        LineSeries::new(data.iter().enumerate().map(|(i, _)| {(i + 1, mean)}), 
            BLUE.filled().stroke_width(2))
    )?;

    root.present()?;
    Ok(())
}


#[test]
fn check() -> Result<(), Box<dyn Error>> {
    let data = vec![0.944, 0.8, 0.7, 0.6];
    draw_cv_scores(data, "img/hist.png".to_string())?;
    Ok(())
}