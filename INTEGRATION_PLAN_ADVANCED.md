# Integration Plan: Advanced Features for ValkyrieLLM Training

This document outlines the plan to integrate four advanced features into the `training/train_llm.py` script:

1.  **UL2-style Mixture-of-Denoisers:** Incorporate multiple denoising objectives during pre-training.
2.  **Retro-style Retrieval Integration:** Augment the model with external knowledge retrieved from a document corpus.
3.  **CoLT5 Conditional Computation:** Enable parts of the model to execute conditionally to improve efficiency.
4.  **Hyena Operator Support:** Integrate the Hyena operator as an alternative or supplement to standard attention for long-context modeling.

## Analysis of Existing Code (`training/train_llm.py`)

*   **Configuration:** Uses `AdvancedTrainingConfig` dataclass and `argparse`. Highly configurable.
*   **Model Setup:** `setup_advanced_model` function constructs `ValkyrieLLM` by composing modules (CoreModel, RWKV, GNN, RAG, MoE, reasoning, etc.). Key integration point.
*   **Data Loading:** `load_dataset_from_config` handles data sources and tokenization. Needs modification for new data requirements.
*   **Training Loop:** `train` function uses `TrainingEngine` for the core loop, optimization, etc. Loss calculation and data handling need updates.
*   **Parallelism:** Supports DDP, FSDP, Pipeline Parallelism. New features must be compatible.
*   **Existing Components:** Includes RAG, MoE, RWKV, reasoning modules, FlashAttention, checkpointing. Interactions must be considered.

## Integration Plan Phases

### Phase 1: Foundational Setup & Configuration

1.  **Update Configuration (`AdvancedTrainingConfig` & `parse_args`):**
    *   Add boolean flags: `use_ul2`, `use_retro`, `use_colt5`, `use_hyena`.
    *   Add specific parameters:
        *   **UL2:** `ul2_denoiser_types: List[str]` (e.g., `["prefix", "suffix", "mask"]`), `ul2_objective_probabilities: Dict[str, float]`.
        *   **Retro:** `retro_index_path: str`, `retro_num_neighbors: int`, `retro_chunk_size: int`, `retro_retriever_type: str` (e.g., "faiss", "external_api"), `retro_add_cross_attention: bool`.
        *   **CoLT5:** `colt5_conditional_layers: Union[List[int], str]` (e.g., `[1, 3, 5]` or `"ffn"`), `colt5_routing_strategy: str` (e.g., "token_level", "sequence_level"), `colt5_router_type: str` (e.g., "mlp", "threshold").
        *   **Hyena:** `hyena_filter_order: int`, `hyena_embedding_dim: int`, `hyena_use_in_layers: Union[List[int], str]` (e.g., `[0, 2, 4]` or `"all"`), `hyena_kernel_size: int`.
    *   Update `__post_init__` in `AdvancedTrainingConfig` for defaults and validation.
    *   Add corresponding arguments in `parse_args`.

### Phase 2: UL2 Mixture-of-Denoisers Integration

1.  **Data Preparation (`data/` or `training/data_loaders.py`):**
    *   Implement functions to apply different denoising strategies (span corruption, prefix/suffix masking) based on `ul2_objective_probabilities`.
    *   Modify `tokenize_function` or add a new preprocessing step in `load_huggingface_dataset`/`load_file_dataset` to apply strategies dynamically.
    *   Ensure the batch contains information about the applied objective (e.g., a type field or specific mask patterns).
2.  **Model Modification (`setup_advanced_model`, `model/ul2.py`):**
    *   Refine/use `model/ul2.py`'s `UL2Denoiser` module.
    *   In `setup_advanced_model`, if `config.use_ul2`:
        *   Instantiate `UL2Denoiser`.
        *   Integrate it into `ValkyrieLLM`, potentially as separate output heads or by modifying the main forward pass to route based on objective type.
3.  **Loss Calculation (`training/training_engine.py` or `train` function):**
    *   Adapt the loss computation to handle different objectives per sample (e.g., masked prediction loss, standard next-token prediction).
    *   Potentially weight the losses based on objective frequency or importance.

### Phase 3: Retro-style Retrieval Integration

1.  **Retrieval Component (`model/retro.py`, `utils/retrieval.py`):**
    *   Implement/integrate a `Retriever` class (e.g., using `faiss-gpu` or an API client).
    *   Handle loading/connecting to the index specified by `retro_index_path`.
    *   Implement logic to fetch `retro_num_neighbors` chunks based on input query embeddings.
2.  **Data Handling (`training/data_loaders.py`, `model/retro.py`):**
    *   Modify dataset loading or collation:
        *   Generate query embeddings from input sequences.
        *   Use the `Retriever` to fetch relevant document chunks.
        *   Augment the input batch with retrieved chunk data (`retrieved_input_ids`, `retrieved_attention_mask`).
3.  **Model Architecture (`setup_advanced_model`, `model/transformer.py`, `model/attention.py`):**
    *   Instantiate the `Retriever` in `setup_advanced_model` if `config.use_retro`.
    *   Modify `ValkyrieLLM`/`CoreModel` forward pass to accept retrieved chunks.
    *   If `config.retro_add_cross_attention` is true:
        *   Adapt `model/attention.py/EnhancedAttention` or create `RetroAttention` to include cross-attention layers where the main sequence attends to retrieved chunks.
    *   Ensure embedding layers and positional encoding handle the potentially larger effective input.
4.  **Training Loop (`training/training_engine.py`):**
    *   Pass augmented batches correctly to the model.

### Phase 4: CoLT5 Conditional Computation Integration

1.  **Conditional Modules (`model/transformer.py`, `model/feed_forward.py`, `model/attention.py`, `model/colt5.py`):**
    *   Identify target modules based on `config.colt5_conditional_layers`.
    *   Implement gating/routing mechanisms (e.g., small MLP router, thresholding) within these modules based on `config.colt5_router_type`.
    *   Modify module `forward` passes for conditional execution (e.g., `output = gate * module(input) + (1 - gate) * input` or complete skipping).
    *   Develop helper modules in `model/colt5.py` as needed.
2.  **Model Setup (`setup_advanced_model`):**
    *   Pass flags/configs to sub-modules during construction to enable conditional computation if `config.use_colt5`.
3.  **Training Considerations:**
    *   Verify correct gradient flow through conditional paths.
    *   Consider auxiliary loss for learned gates.
    *   Test compatibility with FSDP/Pipeline Parallelism (potential load balancing issues).

### Phase 5: Hyena Operator Integration

1.  **Hyena Implementation (`model/hyena.py`, `model/attention_mechanisms.py`):**
    *   Implement the `HyenaOperator` module, including:
        *   Efficient long convolutions (FFT-based).
        *   Gating mechanisms.
        *   Filter parameterization.
    *   Design it as a potential replacement for `SelfAttention`.
2.  **Model Architecture (`setup_advanced_model`, `model/transformer.py`, `model/config.py`):**
    *   Add Hyena parameters to `TransformerConfig`.
    *   In `setup_advanced_model` and `TransformerBlock`, instantiate `HyenaOperator` based on `config.use_hyena` and `config.hyena_use_in_layers`.
    *   Adapt or disable incompatible positional embeddings (e.g., RoPE might need adjustment).
    *   Determine interaction with FlashAttention (likely mutually exclusive or requires hybrid approach).
3.  **Training & Efficiency:**
    *   Benchmark performance and memory usage.
    *   Ensure numerical stability (FFTs).

### Phase 6: Testing, Validation & Documentation

1.  **Unit Tests (`tests/`):** Test new modules (`UL2Denoiser`, `Retriever`, `RetroAttention`, `ConditionalLayer`, `HyenaOperator`) in isolation.
2.  **Integration Tests (`tests/`):** Run short training sessions enabling each feature individually and combined. Check for crashes, NaNs, basic learning.
3.  **Evaluation (`evaluation/`):** Adapt/define metrics (ROUGE, retrieval accuracy, perplexity, downstream tasks). Run evaluations using `evaluation/evaluator.py`.
4.  **Documentation:**
    *   Update `README.md` or create `README_ADVANCED_FEATURES.md`.
    *   Add code comments.
    *   Update `ValkyrieLLM_Technical_Overview.md`.

## Execution Order & Dependencies

*   Start with Phase 1 (Configuration).
*   Phases 2-5 can proceed somewhat in parallel, but integration points require coordination.
*   Retro requires a working retrieval backend/index.
*   Hyena integration needs careful consideration regarding FlashAttention and positional embeddings.
*   CoLT5 requires testing with parallelism.
*   Phase 6 (Testing/Docs) is ongoing throughout and finalized at the end.