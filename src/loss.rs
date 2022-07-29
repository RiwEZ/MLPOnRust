use crate::model;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Zip;

pub struct Loss {
    outputs: Array1<f64>,
    desired: Array1<f64>,
    pub func: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    pub der: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
}

impl Loss {
    /// Mean Squared Error
    pub fn mse() -> Loss {
        fn f(output: f64, desired: f64) -> f64 {
            0.5 * (output - desired).powi(2)
        }
        fn func(outputs: &Array1<f64>, desired: &Array1<f64>) -> Array1<f64> {
            let mut result = Array::<f64, _>::zeros(outputs.len());
            Zip::from(&mut result)
                .and(outputs)
                .and(desired)
                .for_each(|v, &o, &d| *v = f(o, d));
            result
        }
        fn f_p(output: f64, desired: f64) -> f64 {
            output - desired
        }
        fn der(outputs: &Array1<f64>, desired: &Array1<f64>) -> Array1<f64> {
            let mut result = Array::<f64, _>::zeros(outputs.len());
            Zip::from(&mut result)
                .and(outputs)
                .and(desired)
                .for_each(|v, &o, &d| *v = f_p(o, d));
            result
        }

        Loss {
            outputs: arr1(&[0.0]),
            desired: arr1(&[0.0]),
            func,
            der,
        }
    }

    /// Binary Cross Entropy
    /*     pub fn bce() -> Loss {
        fn func(output: f64, desired: f64) -> f64 {
            -desired * output.ln() + (1.0 - desired) * (1.0 - output).ln()
        }
        fn der(output: f64, desired: f64) -> f64 {
            -(desired / output - (1.0 - desired) / (1.0 - output))
        }

        Loss {
            outputs: arr1(&[0.0]),
            desired: arr1(&[0.0]),
            func,
            der,
        }
    } */

    pub fn criterion(&mut self, outputs: &Array1<f64>, desired: &Array1<f64>) -> f64 {
        if outputs.len() != desired.len() {
            panic!("outputs size is not equal to desired size");
        }

        let loss = (self.func)(outputs, desired).fold(0.0, |v, x| v + x);
        self.outputs = outputs.clone();
        self.desired = desired.clone();
        loss
    }

    pub fn backward(&self, layers: &mut Vec<model::Layer>) {
        for l in (0..layers.len()).rev() {
            layers[l].prev_local_grads = layers[l].local_grads.clone(); // copied previous grad before update
            layers[l].prev_grads = layers[l].grads.clone();

            let a_der = layers[l].outputs.map(|x| (layers[l].act.der)(*x)); // apply derivative of act func on outputs
            let previous_a = if l > 0 {
                layers[l - 1].outputs.map(|x| (layers[l - 1].act.func)(*x))
            } else {
                layers[l].inputs.clone()
            };
            let mut grads = Array2::<f64>::zeros(layers[l].grads.dim());

            // output layer
            if l == layers.len() - 1 {
                // compute gradient
                let a = self.outputs.map(|x| (layers[l].act.func)(*x)); // apply act func on outputs
                let local_grad = (self.der)(&a, &self.desired) * &a_der;
                layers[l].local_grads = local_grad.clone();

                for (j, mut row) in grads.axis_iter_mut(Axis(0)).enumerate() {
                    for (k, col) in row.iter_mut().enumerate() {
                        *col = previous_a[k] * local_grad[j];
                    }
                }
                continue;
            }
            
            // hidden layer
            // calculate local_grad based on previous local_grad
            let local_grad = layers[l + 1].w.t().dot(&layers[l + 1].local_grads) * &a_der;
            layers[l].local_grads = local_grad.clone();

            for (j, mut row) in grads.axis_iter_mut(Axis(0)).enumerate() {
                for (k, col) in row.iter_mut().enumerate() {
                    *col = previous_a[k] * local_grad[j]
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_func() {
        assert_eq!(
            (Loss::mse().func)(&arr1(&[2.0]), &arr1(&[1.0])),
            arr1(&[0.5])
        );
        assert_eq!(
            (Loss::mse().func)(&arr1(&[5.0]), &arr1(&[0.0])),
            arr1(&[12.5])
        );
    }

    #[test]
    fn test_mse_der() {
        assert_eq!(
            (Loss::mse().der)(&arr1(&[2.0]), &arr1(&[1.0])),
            arr1(&[1.0])
        );
        assert_eq!(
            (Loss::mse().der)(&arr1(&[5.0]), &arr1(&[0.0])),
            arr1(&[5.0])
        );
    }

    #[test]
    fn test_mse() {
        let mut loss = Loss::mse();

        let l = loss.criterion(&arr1(&[2.0, 1.0, 0.0]), &arr1(&[0.0, 1.0, 2.0]));
        assert_eq!(l, 4.0);

        loss.criterion(
            &arr1(&[34.0, 37.0, 44.0, 47.0, 48.0]),
            &arr1(&[37.0, 40.0, 46.0, 44.0, 46.0]),
        );
        assert_eq!(l, 4.0);
    }

    /*     #[test]
    fn test_bce_func() {
        println!("{}", (Loss::bce().func)(0.9, 0.0));
        println!("{}", (Loss::bce().func)(0.9, 1.0));
    } */
}
