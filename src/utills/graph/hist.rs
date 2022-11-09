use plotters::prelude::*;
use std::error::Error;

use crate::utills::{graph::*, data};

pub struct Histogram2 {
    datas: [Vec<f64>; 2],
    title: String,
    axes_desc: (String, String),
}

impl Histogram2 {
    pub fn new(datas: [Vec<f64>; 2],title: &str, axes_desc: (&str, &str)) -> Histogram2 {
        Histogram2 {
            datas,
            title: title.into(),
            axes_desc: (axes_desc.0.into(), axes_desc.1.into()),
        }
    }

    pub fn mean(&self) -> Vec<f64> {
        self.datas
            .iter()
            .map(|l| {
                l.iter()
                    .fold(0f64, |mean, &val| mean + val / l.len() as f64)
            })
            .collect()
    }

    pub fn max_x(&self) -> f64 {
        self.datas
            .iter()
            .fold(0f64, |max, l| max.max(l.len() as f64))
    }

    pub fn max_y(&self) -> f64 {
        self.datas
            .iter()
            .fold(f64::MIN, |max, l| {
                let l_max = data::max(l);
                if l_max > max {
                    l_max
                }
                else {
                    max
                }
            })
    }

    pub fn draw_hist(&self, path: String, max_y: f64) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let n = self.max_x();
        let mut chart = ChartBuilder::on(&root)
            .caption(&self.title, (FONT, CAPTION, FontStyle::Bold).into_font())
            .margin(20)
            .x_label_area_size(70)
            .y_label_area_size(90)
            .build_cartesian_2d((1..n as u32).into_segmented(), 0.0..max_y)?
            .set_secondary_coord(0.0..n, 0.0..max_y);

        chart
            .configure_mesh()
            .disable_x_mesh()
            .y_max_light_lines(0)
            .y_desc(&self.axes_desc.1)
            .x_desc(&self.axes_desc.0)
            .axis_desc_style((FONT, AXIS_LABEL))
            .y_labels(3)
            .label_style((FONT, AXIS_LABEL - 10))
            .draw()?;


        let colors = [&RED, &BLUE];

        for (i, dt) in self.datas.iter().enumerate() {
            let color = colors[i % 2];
            let offset = i as f64 * 0.4;
            chart.draw_secondary_series(dt.iter().zip(0..).map(|(y, x)| {
                Rectangle::new(
                    [(x as f64 + 0.1 + offset, *y), (x as f64 + 0.5 + offset, 0f64)],
                    Into::<ShapeStyle>::into(color.mix(0.5)).filled(),
                )
            }))?;
        }

        let v: Vec<usize> = (0..(n + 1.0) as usize).collect();
        let mean = self.mean();
        for (j, m) in mean.iter().enumerate() {
            let color = colors[j % 2];
            chart
                .draw_secondary_series(LineSeries::new(
                    v.iter().map(|i| (*i as f64, *m)),
                    color.filled().stroke_width(2),
                ))?
                .label(format!("mean: {:.3}", m))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.filled()));
        }

        chart
            .configure_series_labels()
            .label_font((FONT, SERIE_LABEL).into_font())
            .background_style(&WHITE)
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }
}

/// Draw histogram of given datas
/// axes_desc - (for x, for y)
pub fn draw_acc_hist(
    datas: &Vec<f64>,
    title: &str,
    axes_desc: (&str, &str),
    path: String,
) -> Result<(), Box<dyn Error>> {
    let n = datas.len();
    let mean = datas
        .iter()
        .fold(0.0f64, |mean, &val| mean + val / n as f64);

    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("Hack", 44, FontStyle::Bold).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d((1..n).into_segmented(), 0.0..1.0)?
        .set_secondary_coord(1..n, 0.0..1.0);

    chart
        .configure_mesh()
        .disable_x_mesh()
        .y_max_light_lines(0)
        .y_desc(axes_desc.1)
        .x_desc(axes_desc.0)
        .axis_desc_style(("Hack", 20))
        .y_labels(3)
        .draw()?;

    let hist = Histogram::vertical(&chart)
        .style(RED.mix(0.5).filled())
        .margin(10)
        .data(datas.iter().enumerate().map(|(i, x)| (i + 1, *x)));

    chart.draw_series(hist)?;

    chart
        .draw_secondary_series(LineSeries::new(
            datas.iter().enumerate().map(|(i, _)| (i + 1, mean)),
            BLUE.filled().stroke_width(2),
        ))?
        .label(format!("mean: {:.3}", mean))
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

pub fn draw_acc_2hist(
    datas: [&Vec<f64>; 2],
    title: &str,
    axes_desc: (&str, &str),
    path: String,
) -> Result<(), Box<dyn Error>> {
    let hist2 = Histogram2::new([datas[0].clone(), datas[1].clone()], title, axes_desc);
    hist2.draw_hist(path, 1.0)
}

pub fn draw_2hist(
    datas: [&Vec<f64>; 2],
    title: &str,
    axes_desc: (&str, &str),
    path: String,
) -> Result<(), Box<dyn Error>> {
    let hist2 = Histogram2::new([datas[0].clone(), datas[1].clone()], title, axes_desc);
    hist2.draw_hist(path, hist2.max_y())
}

