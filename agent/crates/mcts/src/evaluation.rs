use reversi_core::{BOARD_FEATURE_COUNT, Board};
use reversi_nn::NnModel;
use tch::{Device, Tensor};

use crate::error::{MctsError, Result};

/// Minimal interface required from a policy-value network
pub trait PolicyValueModel {
    fn forward(&self, x: &Tensor) -> tch::Result<(Tensor, Tensor)>;
    fn device(&self) -> Device;

    fn evaluate_batch(
        &self,
        features: &[f32],
        batch_size: usize,
    ) -> tch::Result<(Vec<f32>, Vec<f32>)> {
        if batch_size == 0 || features.len() != batch_size * BOARD_FEATURE_COUNT {
            return Err(tch::TchError::Kind(format!(
                "Expected {batch_size} board feature sets, got {} values",
                features.len()
            )));
        }

        let input = Tensor::from_slice(features).view([batch_size as i64, 3, 8, 8]);
        let (policy_tensor, value_tensor) = self.forward(&input)?;
        let expected_policy_values = batch_size * 64;
        if policy_tensor.numel() != expected_policy_values || value_tensor.numel() != batch_size {
            return Err(tch::TchError::Kind(format!(
                "Unexpected model output sizes: policy={}, value={}, batch={batch_size}",
                policy_tensor.numel(),
                value_tensor.numel()
            )));
        }

        let mut policies = vec![0.0f32; expected_policy_values];
        let mut values = vec![0.0f32; batch_size];
        policy_tensor
            .contiguous()
            .view([-1])
            .copy_data(&mut policies, expected_policy_values);
        value_tensor
            .contiguous()
            .view([-1])
            .copy_data(&mut values, batch_size);
        Ok((policies, values))
    }
}

impl PolicyValueModel for NnModel {
    fn forward(&self, x: &Tensor) -> tch::Result<(Tensor, Tensor)> {
        NnModel::forward(self, x)
    }

    fn device(&self) -> Device {
        NnModel::device(self)
    }
}

/// Evaluate a board position with the neural network
///
/// Returns (policy_logits, value) where:
/// - policy_logits: Vec<f32> of length 64 (logits for each square)
/// - value: f32 in range [-1, 1] (position evaluation)
pub fn evaluate_with_nn<M: PolicyValueModel>(board: &Board, model: &M) -> Result<(Vec<f32>, f32)> {
    let features = board.to_features();
    let (policy, values) = model.evaluate_batch(&features, 1)?;
    let value = values[0];

    if policy.len() != 64 {
        return Err(MctsError::EvaluationFailed(format!(
            "Expected policy of length 64, got {}",
            policy.len()
        )));
    }

    Ok((policy, value))
}

/// Compute softmax over legal moves only (bitmask version)
///
/// Returns a vector of (move_idx, prob) pairs for each set bit in `legal_mask`.
pub fn softmax_legal_moves_mask(logits: &[f32], legal_mask: u64) -> Vec<(usize, f32)> {
    if legal_mask == 0 {
        return Vec::new();
    }

    // First pass: find max logit for numerical stability and collect indices.
    let mut idxs = Vec::with_capacity(32); // typical branching factor
    let mut mask = legal_mask;
    let mut max = f32::NEG_INFINITY;
    while mask != 0 {
        let tz = mask.trailing_zeros() as usize;
        // Bits are stored with index i at bit (63 - i)
        let idx = 63 - tz;
        idxs.push(idx);
        max = max.max(logits[idx]);
        mask &= mask - 1; // clear lowest set bit
    }

    // Stable ordering to match index order expectations
    idxs.sort_unstable();

    // Second pass: compute exp and sum
    let mut exp_sum = 0.0;
    let mut exp_vals = Vec::with_capacity(idxs.len());
    for &idx in &idxs {
        let e = (logits[idx] - max).exp();
        exp_vals.push(e);
        exp_sum += e;
    }

    // Normalize
    idxs.into_iter()
        .zip(exp_vals)
        .map(|(idx, e)| (idx, e / exp_sum))
        .collect()
}

/// Compute softmax over legal moves only (Vec version)
///
/// Illegal moves are masked with -inf before softmax, so they get probability 0
#[allow(dead_code)]
pub fn softmax_legal_moves(logits: &[f32], legal_moves: &[usize]) -> Vec<f32> {
    if legal_moves.is_empty() {
        return Vec::new();
    }

    // Find max for numerical stability
    let max = legal_moves
        .iter()
        .map(|&m| logits[m])
        .fold(f32::NEG_INFINITY, f32::max);

    // Compute exp and sum
    let exp_sum: f32 = legal_moves.iter().map(|&m| (logits[m] - max).exp()).sum();

    // Normalize
    legal_moves
        .iter()
        .map(|&m| (logits[m] - max).exp() / exp_sum)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_softmax_legal_moves() {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let legal_moves = vec![0, 1, 2];

        let probs = softmax_legal_moves(&logits, &legal_moves);

        // Check sum to 1
        let sum: f32 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

        // Check relative ordering (higher logit -> higher prob)
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);

        // Move 3 is illegal, so it gets no probability
        assert_eq!(probs.len(), 3);
    }

    #[test]
    fn test_softmax_single_move() {
        let logits = vec![1.0, 2.0, 3.0];
        let legal_moves = vec![1];

        let probs = softmax_legal_moves(&logits, &legal_moves);

        assert_eq!(probs.len(), 1);
        assert_relative_eq!(probs[0], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_softmax_empty() {
        let logits = vec![1.0, 2.0, 3.0];
        let legal_moves = vec![];

        let probs = softmax_legal_moves(&logits, &legal_moves);

        assert!(probs.is_empty());
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let logits = vec![1000.0, 1001.0, 1002.0];
        let legal_moves = vec![0, 1, 2];

        let probs = softmax_legal_moves(&logits, &legal_moves);

        // Should not overflow or produce NaN
        assert!(probs.iter().all(|&p| p.is_finite()));

        // Sum should still be 1
        let sum: f32 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_softmax_mask_matches_vec() {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let legal_moves = vec![0, 1, 3];

        let mask = (1u64 << 63) | (1u64 << 62) | (1u64 << 60); // indices 0,1,3

        let vec_probs = softmax_legal_moves(&logits, &legal_moves);
        let mask_probs = softmax_legal_moves_mask(&logits, mask);

        assert_eq!(mask_probs.len(), legal_moves.len());
        for ((idx, p_mask), p_vec) in mask_probs.iter().zip(vec_probs.iter()) {
            assert!(legal_moves.contains(idx));
            assert_relative_eq!(*p_mask, *p_vec, epsilon = 1e-6);
        }
    }
}
