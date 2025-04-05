from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    
    # Basic quantization settings
    enable_quantization: bool = True
    bits: int = 4  # 4-bit quantization
    group_size: int = 128
    
    # Quantization scheme
    scheme: str = "symmetric"  # or "asymmetric"
    dtype: str = "nf4"  # or "fp4", "int4"
    
    # QLoRA specific settings
    use_qlora: bool = True
    qlora_r: int = 8  # Rank for LoRA
    qlora_alpha: int = 32
    qlora_dropout: float = 0.1
    
    # Double quantization
    double_quant: bool = True
    double_group_size: int = 256
    
    # Layer-wise settings
    quantize_embeddings: bool = False
    quantize_attention: bool = True
    quantize_linear: bool = True
    
    # Mixed-precision settings
    compute_dtype: str = "bfloat16"
    kv_cache_dtype: str = "float16"
    
    def validate(self):
        """Validate configuration"""
        valid_schemes = ["symmetric", "asymmetric"]
        valid_dtypes = ["nf4", "fp4", "int4"]
        
        if self.scheme not in valid_schemes:
            raise ValueError(f"Invalid quantization scheme: {self.scheme}")
            
        if self.dtype not in valid_dtypes:
            raise ValueError(f"Invalid quantization dtype: {self.dtype}")
            
        if self.bits not in [4, 8]:
            raise ValueError("Only 4-bit and 8-bit quantization supported")
