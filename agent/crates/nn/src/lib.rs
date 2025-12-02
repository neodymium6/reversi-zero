use std::path::Path;

use tch::{CModule, Device, IValue, Tensor};

pub struct NnModel {
    module: CModule,
}

impl NnModel {
    pub fn load<P: AsRef<Path>>(path: P, device: Device) -> tch::Result<Self> {
        let module = CModule::load_on_device(path, device)?;
        Ok(Self { module })
    }

    pub fn forward(&self, x: &Tensor) -> tch::Result<(Tensor, Tensor)> {
        let input_ivalue = IValue::Tensor(x.shallow_clone());
        let iv = self.module.forward_is(&[input_ivalue])?;

        match iv {
            IValue::Tuple(mut elems) => {
                if elems.len() != 2 {
                    Err(tch::TchError::Kind("Expected tuple of length 2".into()))
                } else {
                    let value_iv = elems.pop().unwrap();
                    let policy_iv = elems.pop().unwrap();

                    let policy = tensor_from_ivalue(policy_iv)?;
                    let value = tensor_from_ivalue(value_iv)?;
                    Ok((policy, value))
                }
            }
            _ => Err(tch::TchError::Kind(
                "Expected TorchScript output to be a tuple".into(),
            )),
        }
    }
}

fn tensor_from_ivalue(iv: IValue) -> tch::Result<Tensor> {
    match iv {
        IValue::Tensor(t) => Ok(t),
        other => Err(tch::TchError::Kind(format!(
            "Expected Tensor, got {other:?}"
        ))),
    }
}
