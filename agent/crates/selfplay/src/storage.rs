use anyhow::Result;
use ndarray::{Array, Array1, Array2, Array4};
use ndarray_npy::write_npy;

use crate::data::TrainingExample;

/// Save training data to NPY files
///
/// Creates three separate files:
/// - `{path}_states.npy`: (N, 3, 8, 8) board states
/// - `{path}_policies.npy`: (N, 64) policy distributions
/// - `{path}_values.npy`: (N,) value targets
///
/// # Arguments
/// * `examples` - Slice of training examples to save
/// * `path` - Base path for output files (without extension)
///
/// # Example
/// ```no_run
/// use reversi_selfplay::storage::save_training_data;
/// use reversi_selfplay::TrainingExample;
///
/// let examples = vec![
///     TrainingExample::new(vec![0.0; 192], vec![0.0; 64], 1.0),
/// ];
/// save_training_data(&examples, "selfplay_data").unwrap();
/// // Creates: selfplay_data_states.npy, selfplay_data_policies.npy, selfplay_data_values.npy
/// ```
pub fn save_training_data(examples: &[TrainingExample], path: &str) -> Result<()> {
    if examples.is_empty() {
        anyhow::bail!("Cannot save empty training data");
    }

    // Extract and flatten states: (N, 3, 8, 8)
    let states: Vec<f32> = examples
        .iter()
        .flat_map(|e| e.state.iter().copied())
        .collect();

    let states_array: Array4<f32> = Array::from_shape_vec((examples.len(), 3, 8, 8), states)?;

    // Extract and flatten policies: (N, 64)
    let policies: Vec<f32> = examples
        .iter()
        .flat_map(|e| e.policy.iter().copied())
        .collect();

    let policies_array: Array2<f32> = Array::from_shape_vec((examples.len(), 64), policies)?;

    // Extract values: (N,)
    let values: Vec<f32> = examples.iter().map(|e| e.value).collect();
    let values_array: Array1<f32> = Array::from_vec(values);

    // Write to separate NPY files
    write_npy(format!("{}_states.npy", path), &states_array)?;
    write_npy(format!("{}_policies.npy", path), &policies_array)?;
    write_npy(format!("{}_values.npy", path), &values_array)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_npy::read_npy;
    use std::fs;

    #[test]
    fn test_save_and_load_training_data() {
        let temp_dir = std::env::temp_dir();
        let base_path = temp_dir.join("test_selfplay");
        let base_path_str = base_path.to_str().unwrap();

        // Create sample data
        let examples = vec![
            TrainingExample::new(vec![1.0; 192], vec![0.5; 64], 1.0),
            TrainingExample::new(vec![0.0; 192], vec![0.1; 64], -1.0),
        ];

        // Save
        save_training_data(&examples, base_path_str).unwrap();

        // Load and verify shapes
        let states_path = format!("{}_states.npy", base_path_str);
        let policies_path = format!("{}_policies.npy", base_path_str);
        let values_path = format!("{}_values.npy", base_path_str);

        let states: Array4<f32> = read_npy(&states_path).unwrap();
        let policies: Array2<f32> = read_npy(&policies_path).unwrap();
        let values: Array1<f32> = read_npy(&values_path).unwrap();

        assert_eq!(states.shape(), &[2, 3, 8, 8]);
        assert_eq!(policies.shape(), &[2, 64]);
        assert_eq!(values.shape(), &[2]);

        // Verify values
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], -1.0);

        // Cleanup
        fs::remove_file(states_path).ok();
        fs::remove_file(policies_path).ok();
        fs::remove_file(values_path).ok();
    }

    #[test]
    fn test_save_empty_data_fails() {
        let examples: Vec<TrainingExample> = vec![];
        let result = save_training_data(&examples, "test");
        assert!(result.is_err());
    }
}
