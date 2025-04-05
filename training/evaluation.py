import os
import logging
import torch
import numpy as np
import time
import math
from typing import Dict, List, Any, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on a validation set
    
    Args:
        model: The model to evaluate
        dataloader: The validation dataloader
        device: The device to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Get loss
            loss = outputs.loss
            
            # Update metrics
            total_loss += loss.item()
            total_steps += 1
    
    # Calculate metrics
    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }

def evaluate_reasoning_capabilities(model, tokenizer, args):
    """
    Evaluate the model's reasoning capabilities
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        args: Command line arguments
        
    Returns:
        Dictionary with reasoning evaluation metrics
    """
    logger.info("Evaluating reasoning capabilities")
    
    # Set model to evaluation mode
    model.eval()
    
    # Define reasoning benchmarks
    if hasattr(args, 'reasoning_benchmark_file') and args.reasoning_benchmark_file:
        # Load benchmark from file
        import json
        with open(args.reasoning_benchmark_file, 'r') as f:
            benchmarks = json.load(f)
    else:
        # Use default benchmark
        benchmarks = [
            {
                "question": "If x = 5 and y = 3, what is x + y?",
                "reasoning_steps": [
                    "We have x = 5",
                    "We have y = 3",
                    "We need to compute x + y",
                    "x + y = 5 + 3 = 8"
                ],
                "answer": "8"
            },
            {
                "question": "If a rectangle has a length of 10 meters and a width of 5 meters, what is its area?",
                "reasoning_steps": [
                    "We have a rectangle with length = 10 meters",
                    "We have a rectangle with width = 5 meters",
                    "The area of a rectangle is length × width",
                    "Area = 10 × 5 = 50 square meters"
                ],
                "answer": "50 square meters"
            }
        ]
    
    # Track metrics
    total_correct = 0
    total_questions = len(benchmarks)
    total_steps_correct = 0
    total_steps = 0
    
    # Process each benchmark
    for benchmark in benchmarks:
        # Prepare input prompt
        prompt = f"Question: {benchmark['question']}\nReasoning step by step:"
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        # Generate reasoning
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 200,
                temperature=0.7,
                num_return_sequences=1,
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract reasoning and answer
        reasoning = generated_text[len(prompt):]
        
        # Check if reasoning steps are correct
        steps_correct = 0
        for step in benchmark["reasoning_steps"]:
            if step.lower() in reasoning.lower():
                steps_correct += 1
        
        # Check if answer is correct
        answer_correct = benchmark["answer"].lower() in reasoning.lower()
        
        # Update metrics
        if answer_correct:
            total_correct += 1
        
        total_steps_correct += steps_correct
        total_steps += len(benchmark["reasoning_steps"])
    
    # Calculate metrics
    accuracy = total_correct / total_questions
    step_accuracy = total_steps_correct / total_steps
    
    results = {
        "accuracy": accuracy,
        "step_accuracy": step_accuracy,
        "correct": total_correct,
        "total": total_questions,
    }
    
    logger.info(f"Reasoning evaluation results: {results}")
    
    return results

def evaluate_sequence_modeling(model, tokenizer, chunk_size=1024, args=None):
    """
    Evaluate sequence modeling capability with focus on RWKV models
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        chunk_size: Size of chunks for evaluation
        args: Additional arguments
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info("Evaluating sequence modeling capabilities")
    
    # Set model to evaluation mode
    model.eval()
    
    results = {}
    
    try:
        # Define test sequences of varying lengths for evaluation
        test_sequences = [
            "The quick brown fox jumps over the lazy dog.",
            "A long time ago in a galaxy far, far away, there was a brave Jedi knight who fought against the dark side of the Force.",
            "Machine learning is a subfield of artificial intelligence that focuses on developing systems that can learn from data without being explicitly programmed. It encompasses a range of techniques including neural networks, decision trees, and reinforcement learning.",
        ]
        
        # Add user-provided test sequences if available
        if hasattr(args, 'sequence_modeling_benchmark_file') and args.sequence_modeling_benchmark_file:
            import json
            with open(args.sequence_modeling_benchmark_file, 'r') as f:
                user_sequences = json.load(f)
                test_sequences.extend(user_sequences)
        
        # Generate longer sequence for stress testing
        long_sequence = " ".join([test_sequences[0]] * 50)
        test_sequences.append(long_sequence)
        
        total_ppl = 0.0
        total_latency = 0.0
        sequence_results = []
        
        for i, sequence in enumerate(test_sequences):
            sequence_length = len(sequence)
            
            # Tokenize sequence
            inputs = tokenizer(sequence, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            labels = input_ids.clone()
            
            # Measure perplexity
            with torch.no_grad():
                start_time = time.time()
                
                # For very long sequences, use chunked processing
                if input_ids.size(1) > chunk_size and hasattr(model, 'forward_chunked'):
                    outputs = model.forward_chunked(input_ids, labels)
                else:
                    outputs = model(input_ids, labels=labels)
                
                latency = time.time() - start_time
                
                loss = outputs.loss
                ppl = torch.exp(loss).item()
                
                total_ppl += ppl
                total_latency += latency
                
                # Save individual sequence results
                sequence_results.append({
                    "length": sequence_length,
                    "tokens": input_ids.size(1),
                    "perplexity": ppl,
                    "latency_ms": latency * 1000
                })
                
                logger.info(f"Sequence {i+1} ({input_ids.size(1)} tokens): "
                            f"PPL={ppl:.4f}, Latency={latency*1000:.2f}ms")
        
        # Average metrics
        avg_ppl = total_ppl / len(test_sequences)
        avg_latency = total_latency / len(test_sequences)
        
        # Test state caching for RWKV models
        if hasattr(model, 'process_with_state'):
            state_results = _evaluate_with_state_caching(model, tokenizer, chunk_size)
            results.update(state_results)
        
        # Evaluate chunked processing efficiency
        if hasattr(model, 'forward_chunked'):
            chunk_results = _evaluate_chunked_processing(model, tokenizer, chunk_size)
            results.update(chunk_results)
        
        # Save main results
        results["perplexity"] = avg_ppl
        results["latency_ms"] = avg_latency * 1000
        results["sequence_results"] = sequence_results
        
        logger.info(f"Sequence modeling evaluation complete: "
                    f"PPL={avg_ppl:.4f}, Latency={avg_latency*1000:.2f}ms")
        
    except Exception as e:
        logger.error(f"Error in sequence modeling evaluation: {str(e)}")
        results["error"] = str(e)
    
    return results

def _evaluate_with_state_caching(model, tokenizer, chunk_size):
    """
    Evaluate the model's capability to maintain state across chunks
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        chunk_size: Size of chunks to process
        
    Returns:
        Results dictionary
    """
    logger.info("Evaluating state caching capabilities")
    
    # Create a long coherent text for testing state caching
    text = """
    The history of artificial intelligence begins with ancient myths and stories of artificial beings endowed with intelligence or consciousness by master craftsmen. However, the modern scientific field of AI research only began in the mid-20th century, specifically in 1956 when the Dartmouth Conference was held. This conference, organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon, is widely considered the foundational event for artificial intelligence as a field.
    
    In the early days, AI researchers made significant progress, creating programs that could solve algebraic word problems, prove theorems in geometry, and learn to speak English. There was widespread optimism that general artificial intelligence was just around the corner. However, progress slowed in the 1970s, leading to what was called the "AI winter," when funding and interest in AI research declined.
    
    The field revived in the 1980s with the commercial success of expert systems and renewed interest in neural networks. However, a second AI winter followed in the late 1980s and early 1990s as expectations again outpaced reality.
    
    The modern era of AI began in the 2010s with the rise of deep learning, fueled by the availability of vast amounts of data, powerful computing resources, and breakthroughs in neural network techniques. This led to significant advances in computer vision, natural language processing, and reinforcement learning, with systems like AlphaGo demonstrating superhuman performance in specific domains.
    
    Today, AI technology has become integrated into our daily lives, from voice assistants to recommendation systems to autonomous vehicles. Research continues to advance, with ongoing exploration of new architectures, learning techniques, and approaches to creating more general, robust AI systems.
    """
    
    # Tokenize the text
    tokens = tokenizer.encode(text)
    
    # Process in chunks with and without state
    with_state_loss = _process_with_state(model, tokens, chunk_size)
    without_state_loss = _process_without_state(model, tokens, chunk_size)
    
    # Calculate perplexity
    with_state_ppl = math.exp(with_state_loss)
    without_state_ppl = math.exp(without_state_loss)
    
    # Calculate state benefit ratio
    if without_state_loss > 0:
        state_benefit = without_state_loss / with_state_loss
    else:
        state_benefit = 1.0
    
    results = {
        "with_state_perplexity": with_state_ppl,
        "without_state_perplexity": without_state_ppl,
        "state_benefit_ratio": state_benefit
    }
    
    logger.info(f"State caching evaluation: With state PPL={with_state_ppl:.4f}, "
                f"Without state PPL={without_state_ppl:.4f}, Benefit={state_benefit:.2f}x")
    
    return results

def _process_with_state(model, tokens, chunk_size):
    """Process tokens with state caching"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Initialize state
    batch_size = 1
    device = next(model.parameters()).device
    if hasattr(model, 'reset_state'):
        model.reset_state(batch_size)
        state = model.state
    else:
        state = None
    
    with torch.no_grad():
        for i in range(0, len(tokens), chunk_size):
            # Get chunk
            chunk = tokens[i:i+chunk_size]
            chunk_tensor = torch.tensor([chunk], device=device)
            
            # Process chunk
            if hasattr(model, 'process_with_state'):
                outputs, state = model.process_with_state(chunk_tensor, state)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            else:
                # Fallback for models that don't support state caching
                outputs = model(chunk_tensor, labels=chunk_tensor)
                loss = outputs.loss
            
            # Add to total
            total_loss += loss.item() * len(chunk)
            total_tokens += len(chunk)
    
    # Return average loss
    return total_loss / total_tokens

def _process_without_state(model, tokens, chunk_size):
    """Process tokens without state caching"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(tokens), chunk_size):
            # Get chunk
            chunk = tokens[i:i+chunk_size]
            chunk_tensor = torch.tensor([chunk], device=next(model.parameters()).device)
            
            # Process chunk without state
            outputs = model(chunk_tensor, labels=chunk_tensor)
            loss = outputs.loss
            
            # Add to total
            total_loss += loss.item() * len(chunk)
            total_tokens += len(chunk)
    
    # Return average loss
    return total_loss / total_tokens

def _evaluate_chunked_processing(model, tokenizer, chunk_size):
    """
    Evaluate the model's efficiency with different chunk sizes
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        chunk_size: Base chunk size
        
    Returns:
        Results dictionary
    """
    logger.info("Evaluating chunked processing efficiency")
    
    # Create a long sequence for testing
    text = " ".join(["This is a test sequence for chunked processing."] * 100)
    
    # Tokenize
    tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([tokens], device=next(model.parameters()).device)
    
    # Test with different chunk sizes
    chunk_sizes = [chunk_size // 4, chunk_size // 2, chunk_size, chunk_size * 2]
    chunk_results = []
    
    model.eval()
    with torch.no_grad():
        # First test with no chunking
        start_time = time.time()
        outputs = model(tokens_tensor, labels=tokens_tensor)
        full_latency = time.time() - start_time
        full_loss = outputs.loss.item()
        full_ppl = math.exp(full_loss)
        
        # Test with different chunk sizes
        for size in chunk_sizes:
            if hasattr(model, 'set_chunk_size'):
                model.set_chunk_size(size)
            
            if hasattr(model, 'forward_chunked'):
                start_time = time.time()
                outputs = model.forward_chunked(tokens_tensor, tokens_tensor)
                latency = time.time() - start_time
                loss = outputs.loss
                ppl = math.exp(loss) if isinstance(loss, float) else torch.exp(loss).item()
                
                # Calculate efficiency
                speed_ratio = full_latency / latency
                memory_estimate = size / chunk_size  # Rough estimate of relative memory usage
                
                chunk_results.append({
                    "chunk_size": size,
                    "perplexity": ppl,
                    "latency_ms": latency * 1000,
                    "speed_ratio": speed_ratio,
                    "memory_ratio": memory_estimate
                })
                
                logger.info(f"Chunk size {size}: PPL={ppl:.4f}, Latency={latency*1000:.2f}ms, "
                            f"Speed ratio={speed_ratio:.2f}x, Memory ratio={memory_estimate:.2f}x")
    
    return {
        "full_processing_perplexity": full_ppl,
        "full_processing_latency_ms": full_latency * 1000,
        "chunked_processing_results": chunk_results
    }

def evaluate_advanced_capabilities(model, tokenizer, args, testbench_file=None):
    """
    Evaluate the model's advanced capabilities including numerical precision,
    verifiable computation, and adaptive reasoning.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        args: Command line arguments
        testbench_file: Optional path to a test benchmark file
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating advanced capabilities")
    
    # Set model to evaluation mode
    model.eval()
    
    results = {
        "gnn": {},
        "numerical_precision": {},
        "verification": {},
        "adaptive_reasoning": {}
    }
    
    # Load test cases
    test_cases = load_advanced_test_cases(testbench_file)
    
    # Evaluate GNN capabilities if enabled
    if hasattr(model, 'gnn_integration_enabled') and model.gnn_integration_enabled:
        results["gnn"] = evaluate_gnn_capabilities(model, tokenizer, test_cases["gnn"])
    else:
        results["gnn"]["enabled"] = False
        logger.info("GNN capabilities not enabled, skipping evaluation")
    
    # Evaluate numerical precision capabilities if enabled
    if hasattr(model, 'has_numerical_precision') and model.has_numerical_precision:
        results["numerical_precision"] = evaluate_numerical_precision(model, tokenizer, test_cases["numerical_precision"])
    else:
        results["numerical_precision"]["enabled"] = False
        logger.info("Numerical precision not enabled, skipping evaluation")
    
    # Evaluate verification capabilities if enabled
    if hasattr(model, 'has_verification') and model.has_verification:
        results["verification"] = evaluate_verification_capabilities(model, tokenizer, test_cases["verification"])
    else:
        results["verification"]["enabled"] = False
        logger.info("Verification capabilities not enabled, skipping evaluation")
    
    # Evaluate adaptive reasoning capabilities if enabled
    if hasattr(model, 'has_adaptive_reasoning') and model.has_adaptive_reasoning:
        results["adaptive_reasoning"] = evaluate_adaptive_reasoning(model, tokenizer, test_cases["adaptive_reasoning"])
    else:
        results["adaptive_reasoning"]["enabled"] = False
        logger.info("Adaptive reasoning not enabled, skipping evaluation")
    
    logger.info("Advanced capabilities evaluation complete")
    return results

def load_advanced_test_cases(testbench_file=None):
    """
    Load test cases for advanced capabilities evaluation
    
    Args:
        testbench_file: Optional path to a test benchmark file
        
    Returns:
        Dictionary with test cases for each capability
    """
    import json
    
    # Default test cases
    default_test_cases = {
        "gnn": [
            {
                "text": "The network has nodes A, B, C, D connected as follows: A connects to B and C, B connects to D, C connects to D. What is the shortest path from A to D?",
                "graph": {
                    "nodes": ["A", "B", "C", "D"],
                    "edges": [["A", "B"], ["A", "C"], ["B", "D"], ["C", "D"]]
                },
                "expected_answer": "2 steps (A to B to D or A to C to D)"
            },
            {
                "text": "In a social network, Alice knows Bob, Charlie, and Dave. Bob knows Eve. Charlie knows Frank. Dave knows Eve and Frank. How many connections are there in total?",
                "graph": {
                    "nodes": ["Alice", "Bob", "Charlie", "Dave", "Eve", "Frank"],
                    "edges": [
                        ["Alice", "Bob"], ["Alice", "Charlie"], ["Alice", "Dave"],
                        ["Bob", "Eve"], ["Charlie", "Frank"], ["Dave", "Eve"], ["Dave", "Frank"]
                    ]
                },
                "expected_answer": "7 connections"
            }
        ],
        "numerical_precision": [
            {
                "text": "Calculate 0.1 + 0.2 with high precision.",
                "expected_answer": "0.3",
                "precision_required": True
            },
            {
                "text": "Calculate the square root of 2 to 10 decimal places.",
                "expected_answer": "1.4142135624",
                "precision_required": True
            },
            {
                "text": "Compute 1/3 + 1/4 + 1/5 as a fraction.",
                "expected_answer": "47/60",
                "precision_required": True
            }
        ],
        "verification": [
            {
                "text": "Prove that the sum of the first n odd numbers is n².",
                "expected_answer": "n²",
                "verification_required": True
            },
            {
                "text": "Solve the equation 2x + 5 = 15, showing all steps.",
                "expected_answer": "x = 5",
                "verification_required": True
            }
        ],
        "adaptive_reasoning": [
            {
                "text": "How might increasing global temperatures affect ocean currents?",
                "reasoning_type": "causal",
                "complexity": "high"
            },
            {
                "text": "If all A are B, and all B are C, what can we conclude about A and C?",
                "reasoning_type": "logical",
                "complexity": "medium",
                "expected_answer": "All A are C"
            },
            {
                "text": "What ethical considerations should be taken into account when designing facial recognition systems?",
                "reasoning_type": "ethical",
                "complexity": "high"
            }
        ]
    }
    
    # If a testbench file is provided, load it and merge with defaults
    if testbench_file and os.path.exists(testbench_file):
        try:
            with open(testbench_file, 'r') as f:
                custom_test_cases = json.load(f)
                
            # Merge with defaults (custom cases take precedence)
            for capability, tests in custom_test_cases.items():
                if capability in default_test_cases:
                    default_test_cases[capability].extend(tests)
                else:
                    default_test_cases[capability] = tests
            
            logger.info(f"Loaded custom test cases from {testbench_file}")
        except Exception as e:
            logger.error(f"Error loading custom test cases: {e}")
    
    return default_test_cases

def evaluate_gnn_capabilities(model, tokenizer, test_cases):
    """
    Evaluate GNN capabilities
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        test_cases: List of test cases
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating GNN capabilities")
    
    results = {
        "enabled": True,
        "tests": [],
        "correct": 0,
        "total": len(test_cases)
    }
    
    for i, test_case in enumerate(test_cases):
        try:
            # Extract test case information
            text = test_case["text"]
            graph = test_case["graph"]
            expected_answer = test_case["expected_answer"]
            
            # Prepare graph data 
            # This is a simplified version - in a real scenario, 
            # graph data would be processed using a library like PyTorch Geometric
            import torch
            
            # Convert nodes to indices
            node_map = {node: i for i, node in enumerate(graph["nodes"])}
            
            # Create edge index 
            edge_index = []
            for edge in graph["edges"]:
                src, dst = edge
                edge_index.append([node_map[src], node_map[dst]])
                # For undirected graphs, add reverse edge as well
                edge_index.append([node_map[dst], node_map[src]])
                
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            
            # Create simple node features (one-hot encoding)
            num_nodes = len(graph["nodes"])
            node_features = torch.eye(num_nodes)
            
            # Create graph data dictionary
            graph_data = {
                "node_features": node_features.to(model.device),
                "edge_index": edge_index.to(model.device),
                "edge_attr": None,
                "batch": torch.zeros(num_nodes, dtype=torch.long).to(model.device)
            }
            
            # Encode input text
            input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
            attention_mask = torch.ones_like(input_ids).to(model.device)
            
            # Generate output with graph data
            with torch.no_grad():
                outputs = model.forward(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    graph_data=graph_data,
                    return_dict=True
                )
                
                generated_ids = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 100,
                    attention_mask=attention_mask,
                    graph_data=graph_data,
                    temperature=0.7
                )
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract answer from generated text
            answer = generated_text[len(text):].strip()
            
            # Check if the answer is correct
            correct = expected_answer.lower() in answer.lower()
            if correct:
                results["correct"] += 1
            
            # Store test results
            results["tests"].append({
                "id": i,
                "text": text,
                "expected_answer": expected_answer,
                "actual_answer": answer,
                "correct": correct
            })
            
            logger.info(f"GNN test {i+1}: {'✓' if correct else '✗'}")
            
        except Exception as e:
            logger.error(f"Error evaluating GNN test case {i}: {e}")
            results["tests"].append({
                "id": i,
                "text": test_case["text"],
                "error": str(e),
                "correct": False
            })
    
    # Calculate accuracy
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]
    else:
        results["accuracy"] = 0.0
    
    logger.info(f"GNN evaluation complete: Accuracy = {results['accuracy']:.2f}")
    return results

def evaluate_numerical_precision(model, tokenizer, test_cases):
    """
    Evaluate numerical precision capabilities
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        test_cases: List of test cases
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating numerical precision capabilities")
    
    results = {
        "enabled": True,
        "tests": [],
        "correct": 0,
        "total": len(test_cases)
    }
    
    for i, test_case in enumerate(test_cases):
        try:
            # Extract test case information
            text = test_case["text"]
            expected_answer = test_case["expected_answer"]
            precision_required = test_case.get("precision_required", True)
            
            # Encode input text
            input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
            attention_mask = torch.ones_like(input_ids).to(model.device)
            
            # Generate output with requirement for precision
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 50,
                    attention_mask=attention_mask,
                    require_math=precision_required,
                    temperature=0.2,  # Lower temperature for more precise answers
                    num_return_sequences=1
                )
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract answer from generated text
            answer = generated_text[len(text):].strip()
            
            # Check if the answer is correct
            if precision_required:
                # For precision tests, check if the expected answer is exactly in the response
                correct = expected_answer in answer
            else:
                # For more general tests, use a looser matching
                correct = expected_answer.lower() in answer.lower()
            
            if correct:
                results["correct"] += 1
            
            # Store test results
            results["tests"].append({
                "id": i,
                "text": text,
                "expected_answer": expected_answer,
                "actual_answer": answer,
                "correct": correct,
                "precision_required": precision_required
            })
            
            logger.info(f"Numerical test {i+1}: {'✓' if correct else '✗'}")
            
        except Exception as e:
            logger.error(f"Error evaluating numerical precision test case {i}: {e}")
            results["tests"].append({
                "id": i,
                "text": test_case["text"],
                "error": str(e),
                "correct": False
            })
    
    # Calculate accuracy
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]
    else:
        results["accuracy"] = 0.0
    
    logger.info(f"Numerical precision evaluation complete: Accuracy = {results['accuracy']:.2f}")
    return results

def evaluate_verification_capabilities(model, tokenizer, test_cases):
    """
    Evaluate verification capabilities
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        test_cases: List of test cases
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating verification capabilities")
    
    results = {
        "enabled": True,
        "tests": [],
        "correct": 0,
        "verified_correct": 0,  # Correctly computed AND verified
        "total": len(test_cases)
    }
    
    for i, test_case in enumerate(test_cases):
        try:
            # Extract test case information
            text = test_case["text"]
            expected_answer = test_case["expected_answer"]
            verification_required = test_case.get("verification_required", True)
            
            # Encode input text
            input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
            attention_mask = torch.ones_like(input_ids).to(model.device)
            
            # Generate output with verification
            with torch.no_grad():
                # First get the cached info with verification
                outputs = model.forward(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    require_math=verification_required,
                    return_dict=True
                )
                
                # Check if verification info is present
                cache = outputs.get("cache", {})
                has_verification = "verification_info" in cache
                
                # Generate text
                generated_ids = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 100,
                    attention_mask=attention_mask,
                    require_math=verification_required,
                    temperature=0.3
                )
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract answer from generated text
            answer = generated_text[len(text):].strip()
            
            # Check if the expected answer is in the response
            correct = expected_answer.lower() in answer.lower()
            
            # Also check if the verification was successful
            verified = has_verification and cache.get("verification_info", {}).get("verified", False)
            verified_correct = correct and verified
            
            if correct:
                results["correct"] += 1
            
            if verified_correct:
                results["verified_correct"] += 1
            
            # Store test results
            results["tests"].append({
                "id": i,
                "text": text,
                "expected_answer": expected_answer,
                "actual_answer": answer,
                "correct": correct,
                "verified": verified,
                "verified_correct": verified_correct
            })
            
            logger.info(f"Verification test {i+1}: {'✓' if correct else '✗'} (Verified: {'✓' if verified else '✗'})")
            
        except Exception as e:
            logger.error(f"Error evaluating verification test case {i}: {e}")
            results["tests"].append({
                "id": i,
                "text": test_case["text"],
                "error": str(e),
                "correct": False,
                "verified": False,
                "verified_correct": False
            })
    
    # Calculate accuracy
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]
        results["verification_accuracy"] = results["verified_correct"] / results["total"]
    else:
        results["accuracy"] = 0.0
        results["verification_accuracy"] = 0.0
    
    logger.info(f"Verification evaluation complete: "
                f"Accuracy = {results['accuracy']:.2f}, "
                f"Verified Accuracy = {results['verification_accuracy']:.2f}")
    return results

def evaluate_adaptive_reasoning(model, tokenizer, test_cases):
    """
    Evaluate adaptive reasoning capabilities
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        test_cases: List of test cases
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating adaptive reasoning capabilities")
    
    results = {
        "enabled": True,
        "tests": [],
        "strategies_used": {},
        "total": len(test_cases)
    }
    
    # Define simple rubric for qualitative assessment
    def assess_reasoning_quality(text, reasoning_type, complexity):
        # Very basic heuristic scoring
        score = 0
        
        # Check for reasoning keywords based on type
        if reasoning_type == "causal":
            keywords = ["because", "therefore", "thus", "consequently", "as a result", "due to", "leads to", "affects"]
            for keyword in keywords:
                if keyword in text.lower():
                    score += 1
        elif reasoning_type == "logical":
            keywords = ["if", "then", "implies", "conclusion", "premise", "follows that", "given that", "syllogism"]
            for keyword in keywords:
                if keyword in text.lower():
                    score += 1
        elif reasoning_type == "ethical":
            keywords = ["ethical", "moral", "principle", "rights", "justice", "fairness", "virtue", "consequence"]
            for keyword in keywords:
                if keyword in text.lower():
                    score += 1
        
        # Check for reasoning depth
        sentences = text.split(".")
        valid_sentences = [s for s in sentences if len(s.strip()) > 10]
        
        # Adjust score based on complexity
        if complexity == "low":
            score = min(5, score + min(3, len(valid_sentences)))
        elif complexity == "medium":
            score = min(5, score + min(3, len(valid_sentences) / 2))
        elif complexity == "high":
            score = min(5, score + min(3, len(valid_sentences) / 3))
        
        # Normalize to 0-5 scale
        return min(5, max(0, score))
    
    for i, test_case in enumerate(test_cases):
        try:
            # Extract test case information
            text = test_case["text"]
            reasoning_type = test_case.get("reasoning_type", "general")
            complexity = test_case.get("complexity", "medium")
            expected_answer = test_case.get("expected_answer", None)
            
            # Encode input text
            input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
            attention_mask = torch.ones_like(input_ids).to(model.device)
            
            # Generate output with adaptive reasoning
            with torch.no_grad():
                # First get the cached info with reasoning
                outputs = model.forward(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    require_reasoning=True,
                    return_dict=True
                )
                
                # Check which reasoning strategy was used
                cache = outputs.get("cache", {})
                reasoning_info = cache.get("reasoning_info", {})
                strategy_used = reasoning_info.get("strategy", "unknown")
                
                # Track strategies used
                if strategy_used not in results["strategies_used"]:
                    results["strategies_used"][strategy_used] = 0
                results["strategies_used"][strategy_used] += 1
                
                # Generate text
                generated_ids = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 300,  # Longer for reasoning
                    attention_mask=attention_mask,
                    require_reasoning=True,
                    temperature=0.7,
                    num_return_sequences=1
                )
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract answer from generated text
            answer = generated_text[len(text):].strip()
            
            # Check if the answer is correct (if expected answer is provided)
            correct = None
            if expected_answer is not None:
                correct = expected_answer.lower() in answer.lower()
            
            # Assess reasoning quality
            reasoning_score = assess_reasoning_quality(answer, reasoning_type, complexity)
            
            # Store test results
            test_result = {
                "id": i,
                "text": text,
                "reasoning_type": reasoning_type,
                "complexity": complexity,
                "actual_answer": answer,
                "strategy_used": strategy_used,
                "reasoning_score": reasoning_score
            }
            
            if expected_answer is not None:
                test_result["expected_answer"] = expected_answer
                test_result["correct"] = correct
            
            results["tests"].append(test_result)
            
            log_msg = f"Reasoning test {i+1}: Strategy={strategy_used}, Score={reasoning_score:.1f}/5.0"
            if correct is not None:
                log_msg += f", Correct={'✓' if correct else '✗'}"
            logger.info(log_msg)
            
        except Exception as e:
            logger.error(f"Error evaluating adaptive reasoning test case {i}: {e}")
            results["tests"].append({
                "id": i,
                "text": test_case["text"],
                "error": str(e)
            })
    
    # Calculate average reasoning score
    reasoning_scores = [t.get("reasoning_score", 0) for t in results["tests"] if "reasoning_score" in t]
    if reasoning_scores:
        results["avg_reasoning_score"] = sum(reasoning_scores) / len(reasoning_scores)
    else:
        results["avg_reasoning_score"] = 0.0
    
    # Calculate accuracy if expected answers were provided
    correct_tests = [t for t in results["tests"] if t.get("correct", False)]
    tests_with_expected = [t for t in results["tests"] if "expected_answer" in t]
    
    if tests_with_expected:
        results["accuracy"] = len(correct_tests) / len(tests_with_expected)
    
    logger.info(f"Adaptive reasoning evaluation complete: "
                f"Avg Score = {results['avg_reasoning_score']:.2f}/5.0, "
                f"Strategies used: {results['strategies_used']}")
    
    return results 