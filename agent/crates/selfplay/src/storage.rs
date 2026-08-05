use anyhow::Result;
use ndarray::{concatenate, Array, Array1, Array2, Array4, Axis};
use ndarray_npy::{read_npy, write_npy};
use std::path::{Path, PathBuf};

use crate::data::TrainingExample;

struct DatasetTransactionPaths {
    final_paths: [PathBuf; 3],
    next_paths: [PathBuf; 3],
    backup_paths: [PathBuf; 3],
    marker_path: PathBuf,
}

impl DatasetTransactionPaths {
    fn new(dir: &Path) -> Self {
        Self {
            final_paths: [
                dir.join("states.npy"),
                dir.join("policies.npy"),
                dir.join("values.npy"),
            ],
            next_paths: [
                dir.join(".states.npy.next"),
                dir.join(".policies.npy.next"),
                dir.join(".values.npy.next"),
            ],
            backup_paths: [
                dir.join(".states.npy.backup"),
                dir.join(".policies.npy.backup"),
                dir.join(".values.npy.backup"),
            ],
            marker_path: dir.join(".append-transaction"),
        }
    }
}

fn remove_files_if_present(paths: &[PathBuf]) -> Result<()> {
    for path in paths {
        if path.exists() {
            std::fs::remove_file(path)?;
        }
    }
    Ok(())
}

/// Recover the old complete dataset if a previous three-file commit was interrupted.
fn recover_dataset_transaction(paths: &DatasetTransactionPaths) -> Result<()> {
    if !paths.marker_path.exists() {
        remove_files_if_present(&paths.next_paths)?;
        remove_files_if_present(&paths.backup_paths)?;
        return Ok(());
    }

    let marker = std::fs::read_to_string(&paths.marker_path)?;
    match marker.trim() {
        "existing" => {
            if paths.backup_paths.iter().any(|path| !path.is_file()) {
                anyhow::bail!("Cannot recover interrupted dataset transaction: backup is missing");
            }

            // Keep every backup until restoration is fully committed, so this
            // operation can itself be retried after interruption.
            remove_files_if_present(&paths.next_paths)?;
            for ((backup, next), final_path) in paths
                .backup_paths
                .iter()
                .zip(paths.next_paths.iter())
                .zip(paths.final_paths.iter())
            {
                std::fs::hard_link(backup, next)?;
                std::fs::rename(next, final_path)?;
            }
        }
        "new" => {
            remove_files_if_present(&paths.final_paths)?;
            remove_files_if_present(&paths.next_paths)?;
        }
        other => anyhow::bail!("Unknown dataset transaction marker: {other:?}"),
    }

    std::fs::remove_file(&paths.marker_path)?;
    remove_files_if_present(&paths.backup_paths)?;
    Ok(())
}

fn commit_dataset_transaction(
    paths: &DatasetTransactionPaths,
    had_existing_data: bool,
) -> Result<()> {
    if had_existing_data {
        for (final_path, backup) in paths.final_paths.iter().zip(paths.backup_paths.iter()) {
            if let Err(error) = std::fs::hard_link(final_path, backup) {
                remove_files_if_present(&paths.next_paths)?;
                remove_files_if_present(&paths.backup_paths)?;
                return Err(error.into());
            }
        }
    }

    let marker = if had_existing_data { "existing" } else { "new" };
    if let Err(error) = std::fs::write(&paths.marker_path, marker) {
        remove_files_if_present(&paths.next_paths)?;
        remove_files_if_present(&paths.backup_paths)?;
        return Err(error.into());
    }

    for (next, final_path) in paths.next_paths.iter().zip(paths.final_paths.iter()) {
        if let Err(error) = std::fs::rename(next, final_path) {
            recover_dataset_transaction(paths)?;
            return Err(error.into());
        }
    }

    // Removing the marker is the commit point. Backups are cleanup-only after it.
    if let Err(error) = std::fs::remove_file(&paths.marker_path) {
        recover_dataset_transaction(paths)?;
        return Err(error.into());
    }
    remove_files_if_present(&paths.backup_paths)?;
    Ok(())
}

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

    // Create parent directory if it doesn't exist
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
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

/// Append training data to a directory (without prefix)
///
/// Saves data to:
/// - `{dir}/states.npy`
/// - `{dir}/policies.npy`
/// - `{dir}/values.npy`
///
/// Creates the directory if it doesn't exist.
/// Appends to existing files if they exist.
///
/// # Arguments
/// * `examples` - Slice of new training examples to append
/// * `dir` - Directory to save files in
///
/// # Example
/// ```no_run
/// use reversi_selfplay::storage::append_training_data_to_dir;
/// use reversi_selfplay::TrainingExample;
///
/// let examples = vec![
///     TrainingExample::new(vec![0.0; 192], vec![0.0; 64], 1.0),
/// ];
/// append_training_data_to_dir(&examples, "data/selfplay").unwrap();
/// // Creates: data/selfplay/states.npy, data/selfplay/policies.npy, data/selfplay/values.npy
/// ```
pub fn append_training_data_to_dir(examples: &[TrainingExample], dir: &str) -> Result<()> {
    if examples.is_empty() {
        anyhow::bail!("Cannot save empty training data");
    }

    // Create directory if it doesn't exist
    std::fs::create_dir_all(dir)?;

    let transaction_paths = DatasetTransactionPaths::new(Path::new(dir));
    recover_dataset_transaction(&transaction_paths)?;
    let [states_path, policies_path, values_path] = &transaction_paths.final_paths;

    let existing_file_count = transaction_paths
        .final_paths
        .iter()
        .filter(|path| path.is_file())
        .count();
    if existing_file_count != 0 && existing_file_count != transaction_paths.final_paths.len() {
        anyhow::bail!(
            "Incomplete training dataset in {dir}: expected all of states.npy, policies.npy, and values.npy"
        );
    }
    let had_existing_data = existing_file_count == transaction_paths.final_paths.len();

    // Prepare new data arrays
    let new_states: Vec<f32> = examples
        .iter()
        .flat_map(|e| e.state.iter().copied())
        .collect();
    let new_states_array: Array4<f32> =
        Array::from_shape_vec((examples.len(), 3, 8, 8), new_states)?;

    let new_policies: Vec<f32> = examples
        .iter()
        .flat_map(|e| e.policy.iter().copied())
        .collect();
    let new_policies_array: Array2<f32> =
        Array::from_shape_vec((examples.len(), 64), new_policies)?;

    let new_values: Vec<f32> = examples.iter().map(|e| e.value).collect();
    let new_values_array: Array1<f32> = Array::from_vec(new_values);

    // Check if files exist and concatenate if they do
    let final_states = if had_existing_data {
        let existing: Array4<f32> = read_npy(states_path)?;
        concatenate(Axis(0), &[existing.view(), new_states_array.view()])?
    } else {
        new_states_array
    };

    let final_policies = if had_existing_data {
        let existing: Array2<f32> = read_npy(policies_path)?;
        concatenate(Axis(0), &[existing.view(), new_policies_array.view()])?
    } else {
        new_policies_array
    };

    let final_values = if had_existing_data {
        let existing: Array1<f32> = read_npy(values_path)?;
        concatenate(Axis(0), &[existing.view(), new_values_array.view()])?
    } else {
        new_values_array
    };

    if final_states.len_of(Axis(0)) != final_policies.len_of(Axis(0))
        || final_states.len_of(Axis(0)) != final_values.len_of(Axis(0))
    {
        anyhow::bail!("Training dataset arrays have mismatched sample counts");
    }

    // Write the complete next generation before replacing any live file.
    let write_result: Result<()> = (|| {
        write_npy(&transaction_paths.next_paths[0], &final_states)?;
        write_npy(&transaction_paths.next_paths[1], &final_policies)?;
        write_npy(&transaction_paths.next_paths[2], &final_values)?;
        Ok(())
    })();
    if let Err(error) = write_result {
        remove_files_if_present(&transaction_paths.next_paths)?;
        return Err(error);
    }

    commit_dataset_transaction(&transaction_paths, had_existing_data)?;

    Ok(())
}

/// Append training data to existing NPY files (or create new if they don't exist)
///
/// DEPRECATED: Use `append_training_data_to_dir` instead for cleaner directory structure.
///
/// If the files already exist, loads them and concatenates the new data.
/// If they don't exist, creates new files with just the new data.
///
/// # Arguments
/// * `examples` - Slice of new training examples to append
/// * `path` - Base path for output files (without extension)
///
/// # Example
/// ```no_run
/// use reversi_selfplay::storage::append_training_data;
/// use reversi_selfplay::TrainingExample;
///
/// let examples = vec![
///     TrainingExample::new(vec![0.0; 192], vec![0.0; 64], 1.0),
/// ];
/// append_training_data(&examples, "selfplay_data").unwrap();
/// // Appends to existing files or creates new ones
/// ```
pub fn append_training_data(examples: &[TrainingExample], path: &str) -> Result<()> {
    if examples.is_empty() {
        anyhow::bail!("Cannot save empty training data");
    }

    // Create parent directory if it doesn't exist
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let states_path = format!("{}_states.npy", path);
    let policies_path = format!("{}_policies.npy", path);
    let values_path = format!("{}_values.npy", path);

    // Prepare new data arrays
    let new_states: Vec<f32> = examples
        .iter()
        .flat_map(|e| e.state.iter().copied())
        .collect();
    let new_states_array: Array4<f32> =
        Array::from_shape_vec((examples.len(), 3, 8, 8), new_states)?;

    let new_policies: Vec<f32> = examples
        .iter()
        .flat_map(|e| e.policy.iter().copied())
        .collect();
    let new_policies_array: Array2<f32> =
        Array::from_shape_vec((examples.len(), 64), new_policies)?;

    let new_values: Vec<f32> = examples.iter().map(|e| e.value).collect();
    let new_values_array: Array1<f32> = Array::from_vec(new_values);

    // Check if files exist and concatenate if they do
    let final_states = if Path::new(&states_path).exists() {
        let existing: Array4<f32> = read_npy(&states_path)?;
        concatenate(Axis(0), &[existing.view(), new_states_array.view()])?
    } else {
        new_states_array
    };

    let final_policies = if Path::new(&policies_path).exists() {
        let existing: Array2<f32> = read_npy(&policies_path)?;
        concatenate(Axis(0), &[existing.view(), new_policies_array.view()])?
    } else {
        new_policies_array
    };

    let final_values = if Path::new(&values_path).exists() {
        let existing: Array1<f32> = read_npy(&values_path)?;
        concatenate(Axis(0), &[existing.view(), new_values_array.view()])?
    } else {
        new_values_array
    };

    // Write concatenated data
    write_npy(states_path, &final_states)?;
    write_npy(policies_path, &final_policies)?;
    write_npy(values_path, &final_values)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_npy::read_npy;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "reversi-zero-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

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

    #[test]
    fn test_append_to_dir_is_transactional_and_appends() {
        let dir = unique_temp_dir("append");
        let dir_str = dir.to_str().unwrap();
        let first = [TrainingExample::new(vec![1.0; 192], vec![0.5; 64], 1.0)];
        let second = [TrainingExample::new(vec![0.0; 192], vec![0.1; 64], -1.0)];

        append_training_data_to_dir(&first, dir_str).unwrap();
        append_training_data_to_dir(&second, dir_str).unwrap();

        let states: Array4<f32> = read_npy(dir.join("states.npy")).unwrap();
        let policies: Array2<f32> = read_npy(dir.join("policies.npy")).unwrap();
        let values: Array1<f32> = read_npy(dir.join("values.npy")).unwrap();
        assert_eq!(states.shape(), &[2, 3, 8, 8]);
        assert_eq!(policies.shape(), &[2, 64]);
        assert_eq!(values.to_vec(), vec![1.0, -1.0]);
        assert!(!dir.join(".append-transaction").exists());

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_append_refuses_partial_existing_dataset() {
        let dir = unique_temp_dir("partial");
        fs::create_dir_all(&dir).unwrap();
        let states = Array4::<f32>::zeros((1, 3, 8, 8));
        write_npy(dir.join("states.npy"), &states).unwrap();
        let examples = [TrainingExample::new(vec![1.0; 192], vec![0.5; 64], 1.0)];

        let result = append_training_data_to_dir(&examples, dir.to_str().unwrap());

        assert!(result.is_err());
        assert!(dir.join("states.npy").is_file());
        assert!(!dir.join("policies.npy").exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_append_recovers_interrupted_existing_transaction() {
        let dir = unique_temp_dir("recover");
        let dir_str = dir.to_str().unwrap();
        let first = [TrainingExample::new(vec![1.0; 192], vec![0.5; 64], 1.0)];
        append_training_data_to_dir(&first, dir_str).unwrap();

        let paths = DatasetTransactionPaths::new(&dir);
        for (final_path, backup) in paths.final_paths.iter().zip(paths.backup_paths.iter()) {
            fs::hard_link(final_path, backup).unwrap();
        }
        fs::write(&paths.marker_path, "existing").unwrap();
        let interrupted_next = dir.join(".interrupted-values.npy");
        write_npy(&interrupted_next, &Array1::<f32>::from_vec(vec![0.0])).unwrap();
        fs::rename(interrupted_next, &paths.final_paths[2]).unwrap();

        let second = [TrainingExample::new(vec![0.0; 192], vec![0.1; 64], -1.0)];
        append_training_data_to_dir(&second, dir_str).unwrap();

        let values: Array1<f32> = read_npy(dir.join("values.npy")).unwrap();
        assert_eq!(values.to_vec(), vec![1.0, -1.0]);
        assert!(!paths.marker_path.exists());
        assert!(paths.backup_paths.iter().all(|path| !path.exists()));
        fs::remove_dir_all(dir).unwrap();
    }
}
