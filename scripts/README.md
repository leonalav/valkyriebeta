# Data Processing Scripts

## Workflow Options

### Pre-Training Workflow
1. `download_datasets.py` - Downloads and prepares raw datasets
   ```bash
   python download_datasets.py --save-format parquet --output-dir data/raw
   ```

2. `prepare_data.py` - Processes raw data into train/eval/inference splits
   ```bash
   python prepare_data.py
   ```

3. `process_data.py` - Final processing step before training
   ```bash
   python process_data.py --input_dir data/processed --output_dir data/ready --mode train
   ```

### Post-Training Student Model Workflow
1. Use `process_data.py` with student mode to prepare data for your custom LLM:
   ```bash
   python process_data.py \
     --input_dir data/processed \
     --output_dir data/student_ready \
     --mode student \
     --student_model_path path/to/your/custom/model.safetensors \
     --teacher_model_path path/to/teacher/model \  # Optional
     --distillation_temp 2.0  # Optional
   ```

## Important Notes
- Pre-training workflow: Run scripts BEFORE training
- Student model workflow: Run after your custom model is trained
- For inference: Use `--mode inference`
- For student models: Use `--mode student` with appropriate model paths
