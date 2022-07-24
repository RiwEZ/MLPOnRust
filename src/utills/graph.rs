use plotters::coord::Shift;
use plotters::prelude::*;
use std::error::Error;

pub struct LossGraph {
    loss: Vec<Vec<f64>>,
    valid_loss: Vec<Vec<f64>>,
}

impl LossGraph {
    pub fn new() -> LossGraph {
        let loss: Vec<Vec<f64>> = vec![];
        let valid_loss: Vec<Vec<f64>> = vec![];
        LossGraph { loss, valid_loss }
    }

    pub fn add_loss(&mut self, training: Vec<f64>, validation: Vec<f64>) {
        self.loss.push(training);
        self.valid_loss.push(validation);
    }
    /// Draw training loss and validation loss at each epoch (x_vec)
    pub fn draw_loss(
        &self,
        idx: u32,
        root: &DrawingArea<BitMapBackend, Shift>,
        loss_vec: &Vec<f64>,
        valid_loss_vec: &Vec<f64>,
    ) -> Result<(), Box<dyn Error>> {
        let max_loss1 = loss_vec.iter().fold(0f64, |max, &val| val.max(max));
        let max_loss2 = valid_loss_vec.iter().fold(0f64, |max, &val| val.max(max));
        let max_loss = max_loss1.max(max_loss2);

        let min_loss1 = loss_vec.iter().fold(0f64, |min, &val| val.min(min));
        let min_loss2 = valid_loss_vec.iter().fold(0f64, |min, &val| val.min(min));
        let min_loss = 
            if min_loss1.min(min_loss2) > 0.0 {
                0.0
            } else {
                min_loss1.min(min_loss2)
            };

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Loss {}", idx),
                ("Hack", 44, FontStyle::Bold).into_font(),
            )
            .margin(20)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(0..loss_vec.len(), min_loss..max_loss)?;

        chart
            .configure_mesh()
            .y_desc("Loss")
            .x_desc("Epochs")
            .axis_desc_style(("Hack", 20))
            .draw()?;

        chart.draw_series(LineSeries::new(
            loss_vec.iter().enumerate().map(|(i, x)| (i + 1, *x)),
            &RED,
        ))?;

        chart.draw_series(LineSeries::new(
            valid_loss_vec.iter().enumerate().map(|(i, x)| (i + 1, *x)),
            &BLUE,
        ))?;

        root.present()?;
        Ok(())
    }

    pub fn draw(&self, path: String) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(&path, (2000, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        // hardcode for 10 iteraions
        let drawing_areas = root.split_evenly((2, 5));

        let mut loss_iter = self.loss.iter();
        let mut valid_loss_iter = self.valid_loss.iter();

        for (drawing_area, idx) in drawing_areas.iter().zip(1..) {
            if let (Some(loss_vec), Some(valid_loss_vec)) =
                (loss_iter.next(), valid_loss_iter.next())
            {
                self.draw_loss(idx, drawing_area, loss_vec, valid_loss_vec)?;
            }
        }

        Ok(())
    }
}

/// Draw cross validation loss for each iteraion
pub fn draw_loss_scores(loss: Vec<f64>, path: String) -> Result<(), Box<dyn Error>> {
    let n = loss.len();
    let max_y = loss
        .iter()
        .fold(0.0f64, |max, &val| if val > max { val } else { max });
    let mean = loss.iter().fold(0.0f64, |mean, &val| mean + val / n as f64);

    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Cross Validation Loss",
            ("Hack", 44, FontStyle::Bold).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d((1..n).into_segmented(), 0.0..max_y)?
        .set_secondary_coord(1..n, 0.0..max_y);

    chart
        .configure_mesh()
        .disable_x_mesh()
        .y_desc("Validation loss")
        .x_desc("Iterations")
        .axis_desc_style(("Hack", 20))
        .draw()?;

    let hist = Histogram::vertical(&chart)
        .style(RED.mix(0.5).filled())
        .margin(10)
        .data(loss.iter().enumerate().map(|(i, x)| (i + 1, *x)));

    chart.draw_series(hist)?;

    chart
        .draw_secondary_series(LineSeries::new(
            loss.iter().enumerate().map(|(i, _)| (i + 1, mean)),
            BLUE.filled().stroke_width(2),
        ))?
        .label(format!("mean loss: {:.3}", mean))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .label_font(("Hack", 14).into_font())
        .background_style(&WHITE)
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Draw cross validation r2 score for each iteraion
pub fn draw_r2_scores(r2: Vec<f64>, path: String) -> Result<(), Box<dyn Error>> {
    let n = r2.len();
    let mean = r2.iter().fold(0.0f64, |mean, &val| mean + val / n as f64);
    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Cross Validation R2 Scores",
            ("Hack", 44, FontStyle::Bold).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d((1..n).into_segmented(), 0.0..1.0)?
        .set_secondary_coord(1..n, 0.0..1.0);

    chart
        .configure_mesh()
        .disable_x_mesh()
        .y_desc("R2 Scores")
        .x_desc("Iterations")
        .axis_desc_style(("Hack", 20))
        .draw()?;

    let hist = Histogram::vertical(&chart)
        .style(RED.mix(0.5).filled())
        .margin(10)
        .data(r2.iter().enumerate().map(|(i, x)| (i + 1, *x)));

    chart.draw_series(hist)?;

    chart
        .draw_secondary_series(LineSeries::new(
            r2.iter().enumerate().map(|(i, _)| (i + 1, mean)),
            BLUE.filled().stroke_width(2),
        ))?
        .label(format!("mean loss: {:.3}", mean))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .label_font(("Hack", 14).into_font())
        .background_style(&WHITE)
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

#[test]
fn check() -> Result<(), Box<dyn Error>> {
    let data = vec![0.944, 0.8, 0.7, 0.6];
    draw_loss_scores(data, "img/hist.png".to_string())?;
    Ok(())
}
