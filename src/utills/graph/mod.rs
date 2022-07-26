use plotters::coord::Shift;
use plotters::prelude::*;
use std::error::Error;

pub mod hist;

const FONT: &str = "Roboto Mono";
const CAPTION: i32 = 70;
const SERIE_LABEL: i32 = 32;
const AXIS_LABEL: i32 = 40;

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
        max_loss: f64,
    ) -> Result<(), Box<dyn Error>> {
        let min_loss1 = loss_vec.iter().fold(f64::NAN, |min, &val| val.min(min));
        let min_loss2 = valid_loss_vec
            .iter()
            .fold(f64::NAN, |min, &val| val.min(min));
        let min_loss = if min_loss1.min(min_loss2) > 0.0 {
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

    pub fn max_loss(&self) -> f64 {
        f64::max(
            self.loss.iter().fold(f64::NAN, |max, vec| {
                let max_loss = vec.iter().fold(f64::NAN, |max, &val| val.max(max));
                f64::max(max_loss, max)
            }),
            self.valid_loss.iter().fold(f64::NAN, |max, vec| {
                let max_loss = vec.iter().fold(f64::NAN, |max, &val| val.max(max));
                f64::max(max_loss, max)
            }),
        )
    }

    pub fn draw(&self, path: String) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(&path, (2000, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        // hardcode for 10 iteraions
        let drawing_areas = root.split_evenly((2, 5));

        let mut loss_iter = self.loss.iter();
        let mut valid_loss_iter = self.valid_loss.iter();
        let max_loss = self.max_loss();
        for (drawing_area, idx) in drawing_areas.iter().zip(1..) {
            if let (Some(loss_vec), Some(valid_loss_vec)) =
                (loss_iter.next(), valid_loss_iter.next())
            {
                self.draw_loss(idx, drawing_area, loss_vec, valid_loss_vec, max_loss)?;
            }
        }
        Ok(())
    }
}

/// Draw confusion matrix
pub fn draw_confustion(matrix_vec: Vec<[[i32; 2]; 2]>, path: String) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(&path, (2000, 1100)).into_drawing_area();
    root.fill(&WHITE)?;

    let (top, down) = root.split_vertically(1000);

    let mut chart = ChartBuilder::on(&down)
        .margin(20)
        .margin_left(40)
        .margin_right(40)
        .x_label_area_size(40)
        .build_cartesian_2d(0i32..50i32, 0i32..1i32)?;
    chart
        .configure_mesh()
        .disable_y_axis()
        .disable_y_mesh()
        .x_labels(3)
        .label_style((FONT, 40))
        .draw()?;

    chart.draw_series((0..50).map(|x| {
        Rectangle::new(
            [(x, 0), (x + 1, 1)],
            HSLColor(
                240.0 / 360.0 - 240.0 / 360.0 * (x as f64 / 50.0),
                0.7,
                0.1 + 0.4 * x as f64 / 50.0,
            )
            .filled(),
        )
    }))?;
    // hardcode for 10 iteraions
    let drawing_areas = top.split_evenly((2, 5));
    let mut matrix_iter = matrix_vec.iter();
    for (drawing_area, idx) in drawing_areas.iter().zip(1..) {
        if let Some(matrix) = matrix_iter.next() {
            let mut chart = ChartBuilder::on(&drawing_area)
                .caption(
                    format!("Iteration {}", idx),
                    (FONT, 40, FontStyle::Bold).into_font(),
                )
                .margin(20)
                .build_cartesian_2d(0i32..2i32, 2i32..0i32)?
                .set_secondary_coord(0f64..2f64, 2f64..0f64);

            chart
                .configure_mesh()
                .disable_axes()
                .max_light_lines(4)
                .disable_x_mesh()
                .disable_y_mesh()
                .label_style(("Hack", 20))
                .draw()?;

            chart.draw_series(
                matrix
                    .iter()
                    .zip(0..)
                    .map(|(l, y)| l.iter().zip(0..).map(move |(v, x)| (x, y, v)))
                    .flatten()
                    .map(|(x, y, v)| {
                        Rectangle::new(
                            [(x, y), (x + 1, y + 1)],
                            HSLColor(
                                240.0 / 360.0 - 240.0 / 360.0 * (*v as f64 / 50.0),
                                0.7,
                                0.1 + 0.4 * *v as f64 / 50.0,
                            )
                            .filled(),
                        )
                    }),
            )?;

            chart.draw_secondary_series(
                matrix
                    .iter()
                    .zip(0..)
                    .map(|(l, y)| l.iter().zip(0..).map(move |(v, x)| (x, y, v)))
                    .flatten()
                    .map(|(x, y, v)| {
                        let text: String = if x == 0 && y == 0 {
                            format!["TP:{}", v]
                        } else if x == 1 && y == 0 {
                            format!["FP:{}", v]
                        } else if x == 0 && y == 1 {
                            format!["FN:{}", v]
                        } else {
                            format!["TN:{}", v]
                        };

                        Text::new(
                            text,
                            ((2.0 * x as f64 + 0.7) / 2.0, (2.0 * y as f64 + 1.0) / 2.0),
                            FONT.into_font().resize(30.0).color(&WHITE),
                        )
                    }),
            )?;
        }
    }
    root.present()?;
    Ok(())
}

/// Receive each cross-validation vector of each individual fitness value.
pub fn draw_ga_progress(
    cv_fitness: &Vec<Vec<(i32, f64)>>,
    path: String,
    max_y: f64,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(&path, (2000, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_x = cv_fitness[0]
        .iter()
        .reduce(|m, x| if m.0 > x.0 { m } else { x })
        .unwrap()
        .0;

    // This is mostly hardcoded
    let drawing_areas = root.split_evenly((2, 5));
    for ((drawing_area, idx), fitness) in drawing_areas.iter().zip(1..).zip(cv_fitness.iter()) {
        let mut chart = ChartBuilder::on(&drawing_area)
            .caption(
                format!("Iteration {}", idx),
                (FONT, 40, FontStyle::Bold).into_font(),
            )
            .margin(40)
            .x_label_area_size(20)
            .y_label_area_size(30)
            .build_cartesian_2d(0i32..max_x, 0.0..max_y)?;

        chart
            .configure_mesh()
            .x_labels(3)
            .y_labels(2)
            .label_style((FONT, 30))
            .max_light_lines(4)
            .draw()?;

        chart.draw_series(
            fitness
                .iter()
                .map(|x| Circle::new((x.0, x.1), 1, BLUE.mix(0.25).filled())),
        )?;
    }
    root.present()?;
    Ok(())
}
