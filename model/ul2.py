import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Tuple, Optional, Dict

# Helper functions for UL2 masking strategies

def span_corruption_mask(
    inputs: torch.Tensor,
    mask_ratio: float = 0.15,
    mean_span_length: int = 3,
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies T5-style span corruption masking.
    Replaces contiguous spans of tokens with a single mask token.

    Args:
        inputs: Input tensor (batch_size, seq_len).
        mask_ratio: Approximate fraction of tokens to mask.
        mean_span_length: Average length of masked spans.
        device: Torch device.

    Returns:
        Tuple of (masked_inputs, labels).
        masked_inputs: Input tensor with spans replaced by mask tokens.
        labels: Target tensor for reconstruction (original tokens in masked spans).
    """
    batch_size, seq_len = inputs.shape
    num_tokens_to_mask = int(mask_ratio * seq_len)
    num_spans_to_mask = max(1, round(num_tokens_to_mask / mean_span_length))

    # Create mask: 0 for keep, 1 for mask
    mask = torch.zeros_like(inputs, dtype=torch.bool, device=device)
    masked_indices = []

    for i in range(batch_size):
        span_lengths = torch.poisson(torch.full((num_spans_to_mask,), float(mean_span_length), device=device)).long().clamp(min=1)
        total_masked = span_lengths.sum()

        # Adjust span lengths if total exceeds desired mask count
        if total_masked > num_tokens_to_mask:
            excess = total_masked - num_tokens_to_mask
            span_lengths = torch.maximum(torch.tensor(1, device=device), span_lengths - (excess // num_spans_to_mask))
            # Distribute remaining excess randomly
            for _ in range(excess % num_spans_to_mask):
                idx_to_reduce = random.randrange(num_spans_to_mask)
                if span_lengths[idx_to_reduce] > 1:
                    span_lengths[idx_to_reduce] -= 1

        # Select random start indices for spans, ensuring no overlap
        possible_starts = torch.arange(seq_len - span_lengths.max(), device=device)
        start_indices = possible_starts[torch.randperm(len(possible_starts), device=device)[:num_spans_to_mask]]

        current_masked_indices = []
        for start, length in zip(start_indices, span_lengths):
            end = start + length
            # Ensure spans don't overlap previously masked tokens
            if not mask[i, start:end].any():
                 mask[i, start:end] = True
                 current_masked_indices.extend(range(start, end))

        masked_indices.append(sorted(current_masked_indices)) # Store indices for label creation

    # Create masked inputs (replace masked tokens with a sentinel/mask token ID)
    # Assuming mask token ID is -1 for now, replace later if needed
    mask_token_id = -1 # Placeholder - should be set from tokenizer
    masked_inputs = inputs.clone()
    masked_inputs[mask] = mask_token_id

    # Create labels (original tokens at masked positions)
    labels = torch.full_like(inputs, -100) # -100 is often used for ignored tokens in loss
    for i in range(batch_size):
        if masked_indices[i]:
            indices_tensor = torch.tensor(masked_indices[i], device=device)
            labels[i, indices_tensor] = inputs[i, indices_tensor]

    return masked_inputs, labels


def prefix_lm_mask(
    inputs: torch.Tensor,
    prefix_ratio_range: Tuple[float, float] = (0.1, 0.9),
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Prefix Language Modeling (R-objective) masking.
    Splits sequence into prefix and suffix; predicts suffix based on prefix.

    Args:
        inputs: Input tensor (batch_size, seq_len).
        prefix_ratio_range: Min and max ratio for prefix length.
        device: Torch device.

    Returns:
        Tuple of (masked_inputs, labels).
        masked_inputs: Input tensor (prefix part).
        labels: Target tensor (suffix part).
    """
    batch_size, seq_len = inputs.shape
    labels = torch.full_like(inputs, -100) # Ignore all tokens initially
    masked_inputs_list = []

    for i in range(batch_size):
        # Randomly choose prefix length
        prefix_ratio = random.uniform(*prefix_ratio_range)
        prefix_len = max(1, int(prefix_ratio * seq_len))
        suffix_len = seq_len - prefix_len

        # Prefix is the input
        masked_inputs_list.append(inputs[i, :prefix_len])

        # Suffix is the target
        labels[i, prefix_len:] = inputs[i, prefix_len:]

    # Pad masked inputs to the original sequence length for consistency if needed,
    # or handle variable lengths in the main model/training loop.
    # For simplicity here, we'll assume the model handles variable lengths or padding is done elsewhere.
    # Returning the original inputs for now, labels indicate the task.
    # A more robust implementation might return only the prefix and handle padding/attention masks upstream.
    masked_inputs = inputs # Placeholder - actual input might be just the prefix

    return masked_inputs, labels


class UL2Loss(nn.Module):
    """
    Calculates the combined loss for UL2 training, incorporating causal LM
    and the specific denoising objective loss based on the task applied.
    """
    def __init__(self,
                 objective_weights: Dict[str, float],
                 causal_lm_weight: float = 0.5,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.objective_weights = objective_weights
        self.causal_lm_weight = causal_lm_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
        self.vocab_size = None # Will be set based on logits shape

    def forward(self,
                logits: torch.Tensor,
                labels: torch.Tensor,
                denoising_labels: Optional[torch.Tensor] = None,
                task: Optional[str] = None) -> torch.Tensor:
        """
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size).
            labels: Standard causal LM labels (batch_size, seq_len).
            denoising_labels: Labels specific to the denoising task (R, X, S) (batch_size, seq_len).
            task: The denoising task applied ('R', 'X', 'S', or None for only causal LM).

        Returns:
            Combined loss tensor.
        """
        if self.vocab_size is None:
            self.vocab_size = logits.shape[-1]

        # 1. Causal Language Modeling Loss (always computed)
        causal_loss = self.cross_entropy_loss(
            logits.view(-1, self.vocab_size),
            labels.view(-1)
        )

        # 2. Denoising Loss (if applicable)
        denoising_loss = torch.tensor(0.0, device=logits.device)
        if task is not None and denoising_labels is not None:
            task_weight = self.objective_weights.get(task, 0.0)
            if task_weight > 0:
                denoising_loss = self.cross_entropy_loss(
                    logits.view(-1, self.vocab_size),
                    denoising_labels.view(-1)
                )
                denoising_loss = denoising_loss * task_weight
        else:
            task_weight = 0.0 # No denoising task applied

        # 3. Combine Losses
        # Adjust causal LM weight based on whether a denoising task was performed
        current_causal_weight = self.causal_lm_weight if task is not None else 1.0
        total_loss = (current_causal_weight * causal_loss) + denoising_loss

        # Log individual losses (optional, e.g., using wandb)
        # wandb.log({
        #     "causal_loss": causal_loss.item(),
        #     f"{task}_denoising_loss": denoising_loss.item() / task_weight if task_weight > 0 else 0.0,
        #     "total_ul2_loss": total_loss.item()
        # })

        return total_loss


class UL2DataProcessor(nn.Module):
    """
    Applies UL2 denoising objectives to input data based on a schedule.
    This module modifies the input batch *before* it goes into the model.
    """
    def __init__(self,
                 objective_mix: List[str] = ['R', 'X', 'S'],
                 objective_weights: Dict[str, float] = {'R': 0.33, 'X': 0.33, 'S': 0.34},
                 dynamic_schedule: bool = True,
                 initial_task: str = 'R',
                 mask_token_id: int = -1, # Replace with actual mask token ID from tokenizer
                 pad_token_id: int = -100, # Replace with actual pad token ID
                 span_mask_ratio: float = 0.15,
                 mean_span_length: int = 3,
                 prefix_ratio_range: Tuple[float, float] = (0.1, 0.9)):
        super().__init__()
        self.objective_mix = objective_mix
        self.objective_weights = objective_weights # Used for sampling if dynamic
        self.dynamic_schedule = dynamic_schedule
        self.current_task = initial_task
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.span_mask_ratio = span_mask_ratio
        self.mean_span_length = mean_span_length
        self.prefix_ratio_range = prefix_ratio_range
        self.training_step = 0

        if not self.dynamic_schedule and not self.objective_mix:
             raise ValueError("Objective mix must be provided for non-dynamic schedule.")
        if self.dynamic_schedule and not self.objective_weights:
             raise ValueError("Objective weights must be provided for dynamic schedule.")

    def select_task(self) -> str:
        """Selects the next denoising task based on the schedule."""
        if self.dynamic_schedule:
            # Sample based on weights
            tasks = list(self.objective_weights.keys())
            weights = list(self.objective_weights.values())
            self.current_task = random.choices(tasks, weights=weights, k=1)[0]
        else:
            # Cycle through the mix
            task_index = self.training_step % len(self.objective_mix)
            self.current_task = self.objective_mix[task_index]

        self.training_step += 1
        return self.current_task

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Applies the selected UL2 objective to the input batch.

        Args:
            batch: Dictionary containing 'input_ids' and potentially 'labels'.
                   Assumes 'labels' are standard causal LM labels (shifted inputs).

        Returns:
            Modified batch dictionary including:
            - 'input_ids': Potentially modified input IDs based on the task.
            - 'labels': Original causal LM labels.
            - 'denoising_labels': Labels specific to the applied denoising task.
            - 'ul2_task': The task applied ('R', 'X', 'S').
        """
        input_ids = batch['input_ids']
        device = input_ids.device

        # Ensure standard causal LM labels exist if not provided
        if 'labels' not in batch:
             # Create standard causal LM labels (shift inputs)
             labels = input_ids.clone()
             labels[:, :-1] = input_ids[:, 1:]
             labels[:, -1] = self.pad_token_id # Pad last token label
             batch['labels'] = labels

        task = self.select_task()
        denoising_labels = torch.full_like(input_ids, self.pad_token_id) # Default ignore

        if task == 'X': # Span Corruption
            # Note: span_corruption_mask currently uses -1 as mask_token_id placeholder
            # Need to update it or replace here if mask_token_id is different.
            masked_inputs, denoising_labels = span_corruption_mask(
                input_ids,
                mask_ratio=self.span_mask_ratio,
                mean_span_length=self.mean_span_length,
                device=device
            )
            # Replace placeholder mask token ID if necessary
            if self.mask_token_id != -1:
                 masked_inputs[masked_inputs == -1] = self.mask_token_id
            batch['input_ids'] = masked_inputs

        elif task == 'R': # Prefix LM
            # Prefix LM modifies labels, not necessarily inputs directly
            # The model needs to know to only attend to the prefix.
            # We'll pass the original inputs but set labels for the suffix.
            _, denoising_labels = prefix_lm_mask(
                input_ids,
                prefix_ratio_range=self.prefix_ratio_range,
                device=device
            )
            # Input IDs remain unchanged for R, but attention mask needs modification upstream.

        elif task == 'S': # Sequential Denoising (similar to X but might have different params)
            # Using span corruption logic for S as a placeholder
            masked_inputs, denoising_labels = span_corruption_mask(
                input_ids,
                mask_ratio=self.span_mask_ratio, # Could use different params for S
                mean_span_length=self.mean_span_length,
                device=device
            )
            if self.mask_token_id != -1:
                 masked_inputs[masked_inputs == -1] = self.mask_token_id
            batch['input_ids'] = masked_inputs

        batch['denoising_labels'] = denoising_labels
        batch['ul2_task'] = task

        return batch

# Example Usage (Conceptual - needs integration into training loop)
# config = { ... ul2 config from train_llm.py ... }
# ul2_processor = UL2DataProcessor(**ul2_config)
# ul2_loss_fn = UL2Loss(objective_weights=config.ul2_objective_weights,
#                       causal_lm_weight=config.ul2_causal_lm_weight)

# In training loop:
# for batch in dataloader:
#     batch = batch.to(device)
#     processed_batch = ul2_processor(batch) # Apply UL2 objective
#
#     # Model forward pass (needs to handle attention mask for 'R' task)
#     # Need to generate appropriate attention mask based on processed_batch['ul2_task']
#     attention_mask = create_attention_mask(processed_batch['input_ids'], processed_batch['ul2_task'])
#     outputs = model(input_ids=processed_batch['input_ids'], attention_mask=attention_mask)
#     logits = outputs.logits
#
#     # Calculate loss
#     loss = ul2_loss_fn(
#         logits=logits,
#         labels=processed_batch['labels'], # Original causal LM labels
#         denoising_labels=processed_batch['denoising_labels'],
#         task=processed_batch['ul2_task']
#     )
#
#     loss.backward()
#     optimizer.step()
#     ...