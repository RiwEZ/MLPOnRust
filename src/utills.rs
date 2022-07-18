use rand::prelude::SliceRandom;
use serde::Deserialize;
use std::error::Error;
use plotters::prelude::*;

pub fn draw_loss(loss_vec: Vec<f64>, path: String) -> Result<(), Box<dyn Error>> {
    // plotting loss
    let root = BitMapBackend::new(&path, (1024, 768))
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

#[derive(Debug)]
#[derive(Clone)]
pub struct Data {
    pub inputs: Vec<f64>,
    pub labels: Vec<f64>
}
pub struct DataSet {
    datas: Vec<Data>,
}

impl DataSet {
    pub fn new(datas: Vec<Data>) -> DataSet {
        DataSet {datas}
    }
    
    pub fn cross_valid_set(&self, percent: f64) -> Vec<(DataSet, DataSet)> {
        if percent < 0.0 && percent > 1.0 {
            panic!("argument percent must be in range [0, 1]")
        }
        let k = (percent * (self.datas.len() as f64)).ceil() as usize; // fold size
        let n = (self.datas.len() as f64 / k as f64).ceil() as usize; // number of folds
        let mut set: Vec<(DataSet, DataSet)> = vec![]; 

        let mut curr: usize = 0;
        
        for _ in 0..n {
            let r_pt: usize = if curr + k > self.datas.len() {self.datas.len()} else {curr + k};

            let validation_set: Vec<Data> = self.datas[curr..r_pt].to_vec();
            let training_set: Vec<Data> = 
                if curr > 0 {
                    let mut temp = self.datas[0..curr].to_vec();
                    temp.append(&mut self.datas[r_pt..self.datas.len()].to_vec());
                    temp
                }
                else {
                    self.datas[r_pt..self.datas.len()].to_vec()
                };

            println!("{}, {}", curr, r_pt);
            set.push((DataSet::new(training_set), DataSet::new(validation_set)));
            curr += k
        }
        set
    }
    
    pub fn get_datas(&self) -> Vec<Data> {
        self.datas.clone()
    }

    pub fn get_shuffled(&self) -> Vec<Data> {
        let mut shuffled_datas = self.datas.clone();
        shuffled_datas.shuffle(&mut rand::thread_rng());
        shuffled_datas
    }
}

pub fn xor_dataset() -> DataSet {
    let inputs = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let labels = vec![[0.0], [1.0], [1.0], [0.0]];
    let mut datas: Vec<Data> = vec![];
    for i in 0..4 {
        datas.push(Data {inputs: inputs[i].to_vec(), labels: labels[i].to_vec()});
    }
    
    DataSet::new(datas)
}

pub fn flood_dataset() -> Result<DataSet, Box<dyn Error>> {    
    #[derive(Debug, Deserialize)]
    struct Record {
        s1_t3: u32,
        s1_t2: u32,
        s1_t1: u32,
        s1_t0: u32,
        s2_t3: u32,
        s2_t2: u32,
        s2_t1: u32,
        s2_t0: u32,
        t7: u32
    }

    let mut datas: Vec<Data> = vec![];
    let mut reader = csv::Reader::from_path("data/flood_dataset.csv")?;
    for record in reader.deserialize() {
        let record: Record = record?;
        let mut inputs: Vec<f64> = vec![];
        // station 1
        inputs.push(f64::from(record.s1_t3));
        inputs.push(f64::from(record.s1_t2));
        inputs.push(f64::from(record.s1_t1));
        inputs.push(f64::from(record.s1_t0));
        // station 2
        inputs.push(f64::from(record.s2_t3));
        inputs.push(f64::from(record.s2_t2));
        inputs.push(f64::from(record.s2_t1));
        inputs.push(f64::from(record.s2_t0));

        let labels: Vec<f64> = vec![f64::from(record.t7)];
        datas.push(Data {inputs, labels});
    }
    Ok(DataSet::new(datas))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_test() -> Result<(), Box<dyn Error>> {
        let dt = flood_dataset()?;
        dt.cross_valid_set(0.1);

        //println!("\n{:?}", dt.get_samples());
        //println!("\n{:?}", dt.validation_set());
        Ok(())
    }
}