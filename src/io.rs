use crate::model;
use crate::activator;
use serde_json::{json, Value, to_writer_pretty};
use std::fs::File;
use std::error::Error;
use std::io::Read;

pub fn save(layers: &Vec<model::Layer>, path: String) -> Result<(), Box<dyn Error>> {
    let mut json: Vec<Value>= vec![];

    for l in layers {
        json.push(json!({
            "inputs": l.inputs.len(),
            "outputs": l.outputs.len(),
            "w": l.w,
            "b": l.b,
            "act": l.act.name
        }));
    }
    let result = json!(json);
    let file = File::create(path)?;
    to_writer_pretty(&file, &result)?;
    Ok(())
}

pub fn read_file(path: String) -> Result<String, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

pub fn load(path: String) -> Result<model::Net, Box<dyn Error>> {
    let contents = read_file(path)?;

    let json: Value = serde_json::from_str(&contents)?;
    let mut layers: Vec<model::Layer> = vec![];

    for l in json.as_array().unwrap() {
        // default layer activation is simeple linear f(x) = x
        let mut layer = model::Layer::new(
                l["inputs"].as_u64().unwrap(), 
                l["outputs"].as_u64().unwrap(), 
                0.0, activator::linear());
        // setting activation function
        if l["act"] == "sigmoid" {
            layer.act = activator::sigmoid();
        }
        // setting weights and bias
        let w = l["w"].as_array().unwrap();
        let b = l["b"].as_array().unwrap();
        for j in 0..w.len() {
            layer.b[j] = b[j].as_f64().unwrap();
            let w_j = w[j].as_array().unwrap();
            for i in 0..w_j.len() {
                layer.w[j][i] = w_j[i].as_f64().unwrap();
            }
        } 
        
        layers.push(layer);
    }
    
    Ok(model::Net::from_layers(layers))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_test() -> Result<(), Box<dyn Error>> {
        /*         
        let net = model::Net::new(vec![2, 2, 2]);
        save(& net.layers, "models/xor.json".to_string())?;
        */
        load("models/xor.json".to_string())?;
        Ok(())
    }
}