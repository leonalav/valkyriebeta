"""
Just Read Twice (JRT) Processor

This module implements the JRT technique which improves in-context learning
by presenting information multiple times in different orders to the model.
This approach is especially beneficial for recurrent architectures and models
with linear attention which may have difficulty retaining context information.

Reference: "Just Read Twice: Towards Reliable In-Context Learning
with Linear-Time Sequence Models" (Arora et al., 2024)
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class JRTProcessor:
    """
    Processor that implements the Just Read Twice technique for improved
    in-context learning by repeating context information in different orders.
    """
    
    def __init__(
        self, 
        repetitions: int = 2,
        chunk_size: Optional[int] = None,
        delimiter: str = "\n\n",
        instruction_aware: bool = True,
        reverse_order: bool = True,
        interleave: bool = False,
        preserve_task: bool = True
    ):
        """
        Initialize the JRT processor.
        
        Args:
            repetitions: Number of times to repeat the context (default: 2)
            chunk_size: Size of chunks to split the content into (default: None)
            delimiter: Delimiter to use between chunks (default: "\n\n")
            instruction_aware: Whether to treat the first part as instructions (default: True)
            reverse_order: Whether to reverse the order of chunks in repetition (default: True)
            interleave: Whether to interleave chunks rather than sequential repetition (default: False)
            preserve_task: Whether to keep the task/question at the end (default: True)
        """
        self.repetitions = max(1, repetitions)
        self.chunk_size = chunk_size
        self.delimiter = delimiter
        self.instruction_aware = instruction_aware
        self.reverse_order = reverse_order
        self.interleave = interleave
        self.preserve_task = preserve_task
    
    def _split_instruction_content(self, text: str) -> Tuple[str, str, str]:
        """
        Split text into instruction, content, and task components.
        
        Args:
            text: Input text to process
            
        Returns:
            instruction: The instruction part (if any)
            content: The main content part
            task: The final task/question part (if any)
        """
        # Common instruction patterns to identify
        instruction_patterns = [
            # GPT-style instruction pattern
            r"^(.*?(?:instructions?|guidelines?|task|scenario|context):.*?\n\n)(.*?)(\n\n.*?(?:question|task|problem|what|how|why|when|where|who|answer|solve|explain|analyze|evaluate|compare|discuss):.*?$|$)",
            # General colon-based separator
            r"^(.*?:\s*\n\n)(.*?)(\n\n.*?(?:question|task|problem|answer|solve):.*?$|$)",
            # Numbered instruction pattern
            r"^(.*?(?:1\.|I\.|Step 1:).*?\n\n)(.*?)(\n\n.*?(?:question|task|problem|what|how|why|when|where|who):.*?$|$)",
        ]
        
        for pattern in instruction_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                instruction, content, task = match.groups()
                return instruction, content, task
        
        # If no clear pattern is found and preserve_task is True,
        # try to identify the question/task at the end
        if self.preserve_task:
            task_patterns = [
                r"(.*?)(\n\n.*?(?:question|task|problem|what|how|why|when|where|who|answer|solve|explain|analyze|evaluate|compare|discuss):.*?$)",
            ]
            
            for pattern in task_patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    content, task = match.groups()
                    return "", content, task
        
        # If no pattern matches or preserve_task is False, return everything as content
        return "", text, ""
    
    def _chunk_content(self, content: str) -> List[str]:
        """
        Split content into chunks for repetition.
        
        Args:
            content: Content to split into chunks
            
        Returns:
            chunks: List of content chunks
        """
        if not content.strip():
            return []
            
        if self.chunk_size is None:
            # Split by double newlines or other natural boundaries
            chunks = re.split(r'\n\n+|\. +(?=[A-Z])', content)
            # Filter out empty chunks
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            return chunks
        else:
            # Split by character count
            words = content.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_length = len(word)
                if current_length + word_length + len(current_chunk) > self.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    current_chunk.append(word)
                    current_length += word_length
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            return chunks
    
    def process(self, text: str) -> str:
        """
        Process text using the Just Read Twice technique.
        
        Args:
            text: Input text to process
            
        Returns:
            processed_text: Text with repeated content for improved in-context learning
        """
        if self.repetitions <= 1:
            return text
        
        # Split text into instruction, content, and task parts
        instruction, content, task = "", text, ""
        if self.instruction_aware:
            instruction, content, task = self._split_instruction_content(text)
        
        # Split content into chunks
        chunks = self._chunk_content(content)
        
        if not chunks:
            return text
        
        # Prepare different orderings of chunks
        chunk_orderings = []
        
        # First ordering: original order
        chunk_orderings.append(chunks)
        
        # Additional orderings based on configuration
        for i in range(1, self.repetitions):
            if self.reverse_order:
                # Reverse the chunks
                new_order = chunks[::-1]
            else:
                # Shuffle chunks using a deterministic pattern based on index
                new_order = chunks[i:] + chunks[:i]
            
            chunk_orderings.append(new_order)
        
        # Combine chunks based on repetition strategy
        if self.interleave:
            # Interleave chunks from different orderings
            interleaved_chunks = []
            max_idx = max(len(ordering) for ordering in chunk_orderings)
            
            for i in range(max_idx):
                for ordering in chunk_orderings:
                    if i < len(ordering):
                        interleaved_chunks.append(ordering[i])
            
            processed_chunks = interleaved_chunks
        else:
            # Sequential repetition: first all chunks in first order, then all in second order, etc.
            processed_chunks = []
            for ordering in chunk_orderings:
                processed_chunks.extend(ordering)
        
        # Reconstruct the text
        processed_content = self.delimiter.join(processed_chunks)
        processed_text = instruction + processed_content + task
        
        return processed_text

def apply_jrt(
    text: str, 
    repetitions: int = 2, 
    chunk_size: Optional[int] = None,
    delimiter: str = "\n\n",
    instruction_aware: bool = True,
    reverse_order: bool = True,
    interleave: bool = False,
    preserve_task: bool = True
) -> str:
    """
    Apply Just Read Twice processing to improve in-context learning.
    
    Args:
        text: Input text to process
        repetitions: Number of times to repeat the context
        chunk_size: Size of chunks to split the content into
        delimiter: Delimiter to use between chunks
        instruction_aware: Whether to treat the first part as instructions
        reverse_order: Whether to reverse the order of chunks in repetition
        interleave: Whether to interleave chunks rather than sequential repetition
        preserve_task: Whether to keep the task/question at the end
        
    Returns:
        processed_text: Text with repeated content for improved in-context learning
    """
    processor = JRTProcessor(
        repetitions=repetitions,
        chunk_size=chunk_size,
        delimiter=delimiter,
        instruction_aware=instruction_aware,
        reverse_order=reverse_order,
        interleave=interleave,
        preserve_task=preserve_task
    )
    return processor.process(text) 