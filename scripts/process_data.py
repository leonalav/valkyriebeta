import os
import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer
from data.data_processor import UnifiedDataProcessor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description='Process data files for training, inference, or student model')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--cache_dir', type=str, default='.cache', help='Directory for caching')
    parser.add_argument('--model_name', type=str, 
                       default='google/gemma-2b-27b-it', 
                       help='Model name for tokenizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--mode', choices=['train', 'inference', 'preprocess', 'student'], 
                      default='train', help='Processing mode')
    parser.add_argument('--student_model_path', type=str, 
                      help='Path to student model (required for student mode)')
    parser.add_argument('--teacher_model_path', type=str,
                      help='Path to teacher model (optional for student mode)')
    parser.add_argument('--distillation_temp', type=float, default=1.0,
                      help='Temperature for knowledge distillation')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize appropriate tokenizer based on mode
        if args.mode == 'student':
            if not args.student_model_path:
                raise ValueError("student_model_path is required for student mode")
            logger.info(f"Loading student model tokenizer from {args.student_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(args.student_model_path)
        else:
            logger.info(f"Loading tokenizer from {args.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Initialize processor with mode-specific settings
        processor_kwargs = {
            'tokenizer': tokenizer,
            'cache_dir': args.cache_dir,
            'max_seq_length': args.max_seq_length
        }
        
        if args.mode == 'student':
            processor_kwargs.update({
                'is_student': True,
                'teacher_model_path': args.teacher_model_path,
                'distillation_temp': args.distillation_temp
            })
        
        processor = UnifiedDataProcessor(**processor_kwargs)
        
        # Get input files
        input_files = []
        for ext in ['.jsonl', '.parquet']:
            input_files.extend(
                str(f) for f in Path(args.input_dir).glob(f"**/*{ext}")
            )
        
        if not input_files:
            raise ValueError(f"No .jsonl or .parquet files found in {args.input_dir}")
            
        logger.info(f"Found {len(input_files)} input files")
        
        # Process data with mode-specific handling
        result = processor.process_data_files(
            data_files=input_files,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            is_inference=(args.mode == 'inference'),
            preprocess_only=(args.mode == 'preprocess'),
            is_student=(args.mode == 'student')
        )
        
        # Log results
        if args.mode == 'preprocess':
            logger.info(f"Preprocessed {len(result)} examples")
        else:
            logger.info(
                f"Processed {result['num_examples']} examples\n"
                f"Preprocessed data saved to: {result['preprocessed_dir']}"
            )
            
            # Log cache statistics
            cache_stats = processor.get_cache_stats()
            logger.info(
                f"Cache statistics:\n"
                f"- Size: {cache_stats['cache_size_mb']:.2f}MB\n"
                f"- Files: {cache_stats['num_cached_files']}"
            )
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        # Cleanup
        if 'processor' in locals():
            processor.cleanup()

if __name__ == '__main__':
    main()