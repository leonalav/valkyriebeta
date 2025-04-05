import torch
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CUDAGraphsHandler:
    """Manages CUDA graphs for optimized forward pass"""
    
    def __init__(self, model: torch.nn.Module, batch_size: int, seq_length: int):
        self.model = model
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.static_shapes = True
        self.graphs: Dict[str, Any] = {}
        self.static_inputs: Dict[str, torch.Tensor] = {}
        
    def warmup(self, sample_input: Dict[str, torch.Tensor]):
        """Warmup CUDA graphs with sample input"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping graph warmup")
            return
            
        try:
            # Create static inputs
            self.static_inputs = {
                k: torch.zeros_like(v, device='cuda')
                for k, v in sample_input.items()
            }
            
            # Warmup model
            self.model.zero_grad(set_to_none=True)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):  # Multiple warmup iterations
                    outputs = self.model(**self.static_inputs)
            torch.cuda.current_stream().wait_stream(s)
            
            # Create graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                graph_outputs = self.model(**self.static_inputs)
            
            self.graphs['forward'] = {
                'graph': g,
                'outputs': graph_outputs
            }
            
            logger.info("CUDA graphs warmup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during CUDA graphs warmup: {str(e)}")
            self.graphs.clear()
            
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Any:
        """Execute forward pass using CUDA graph if possible"""
        if not self.graphs:
            return self.model(**inputs)
            
        if self._can_use_graph(inputs):
            # Copy inputs to static buffers
            with torch.no_grad():
                for k, v in inputs.items():
                    self.static_inputs[k].copy_(v)
                    
            # Run graph
            self.graphs['forward']['graph'].replay()
            return self.graphs['forward']['outputs']
        else:
            return self.model(**inputs)
            
    def _can_use_graph(self, inputs: Dict[str, torch.Tensor]) -> bool:
        """Check if inputs are compatible with graph"""
        if not self.static_inputs:
            return False
            
        return all(
            v.shape == self.static_inputs[k].shape
            for k, v in inputs.items()
        )
        
    def cleanup(self):
        """Cleanup graph resources"""
        self.graphs.clear()
        self.static_inputs.clear()
