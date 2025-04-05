"""
Function Calling Framework

This module implements a framework for function calling capabilities, allowing the model
to interact with external tools and APIs. It includes:
1. Function registry for defining available tools
2. Function parsing for extracting function calls from model outputs
3. Function execution for calling external APIs
4. Schema validation for function parameters

This capability significantly extends the model's utility by enabling it to take
actions in external systems rather than being limited to generating text.
"""

import re
import json
import logging
import inspect
from typing import Dict, List, Any, Optional, Union, Callable, Type, get_type_hints
from dataclasses import dataclass, field
import traceback

logger = logging.getLogger(__name__)

@dataclass
class FunctionSchema:
    """Schema for a callable function"""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required_parameters: List[str] = field(default_factory=list)
    return_type: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation"""
        schema_dict = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_parameters
            }
        }
        
        if self.return_type:
            schema_dict["return"] = self.return_type
            
        return schema_dict
    
    @classmethod
    def from_function(cls, func: Callable, description: Optional[str] = None) -> 'FunctionSchema':
        """Create a schema from a Python function"""
        # Get function name and docstring
        name = func.__name__
        func_description = description or func.__doc__ or f"Function {name}"
        
        # Get parameter types from type hints
        type_hints = get_type_hints(func)
        signature = inspect.signature(func)
        
        parameters = {}
        required_parameters = []
        
        for param_name, param in signature.parameters.items():
            # Skip self parameter for methods
            if param_name == 'self':
                continue
                
            param_type = type_hints.get(param_name, Any)
            param_schema = _type_to_schema(param_type)
            
            # Check if parameter has a default value
            has_default = param.default is not inspect.Parameter.empty
            
            parameters[param_name] = param_schema
            
            if not has_default:
                required_parameters.append(param_name)
        
        # Get return type schema
        return_type = None
        if 'return' in type_hints:
            return_type = _type_to_schema(type_hints['return'])
        
        return cls(
            name=name,
            description=func_description,
            parameters=parameters,
            required_parameters=required_parameters,
            return_type=return_type
        )

def _type_to_schema(type_hint: Type) -> Dict[str, Any]:
    """Convert a Python type hint to a JSON schema"""
    # Handle simple types
    if type_hint is str:
        return {"type": "string"}
    elif type_hint is int:
        return {"type": "integer"}
    elif type_hint is float:
        return {"type": "number"}
    elif type_hint is bool:
        return {"type": "boolean"}
    elif type_hint is List[str]:
        return {"type": "array", "items": {"type": "string"}}
    elif type_hint is List[int]:
        return {"type": "array", "items": {"type": "integer"}}
    elif type_hint is List[float]:
        return {"type": "array", "items": {"type": "number"}}
    elif type_hint is Dict[str, Any]:
        return {"type": "object"}
    
    # Handle Union types (e.g., Optional)
    origin = getattr(type_hint, "__origin__", None)
    if origin is Union:
        args = getattr(type_hint, "__args__", [])
        # Handle Optional[X] which is Union[X, None]
        if len(args) == 2 and args[1] is type(None):
            schema = _type_to_schema(args[0])
            schema["nullable"] = True
            return schema
    
    # Handle List[X]
    if origin is list:
        args = getattr(type_hint, "__args__", [Any])
        return {"type": "array", "items": _type_to_schema(args[0])}
    
    # Handle Dict[K, V]
    if origin is dict:
        args = getattr(type_hint, "__args__", [str, Any])
        return {
            "type": "object",
            "additionalProperties": _type_to_schema(args[1])
        }
    
    # Default to string for unknown types
    return {"type": "string"}

@dataclass
class FunctionCall:
    """Represents a function call extracted from model output"""
    name: str
    arguments: Dict[str, Any]
    raw_text: str

class FunctionRegistry:
    """Registry for functions that can be called by the model"""
    
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.schemas: Dict[str, FunctionSchema] = {}
    
    def register(self, func: Optional[Callable] = None, *, name: Optional[str] = None, 
                 description: Optional[str] = None) -> Callable:
        """Register a function in the registry"""
        def decorator(f):
            func_name = name or f.__name__
            schema = FunctionSchema.from_function(f, description)
            
            self.functions[func_name] = f
            self.schemas[func_name] = schema
            
            return f
        
        # Called as a decorator with arguments
        if func is None:
            return decorator
        
        # Called as a simple decorator
        return decorator(func)
    
    def get_function(self, name: str) -> Optional[Callable]:
        """Get a registered function by name"""
        return self.functions.get(name)
    
    def get_schema(self, name: str) -> Optional[FunctionSchema]:
        """Get the schema for a registered function"""
        return self.schemas.get(name)
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered functions"""
        return [schema.to_dict() for schema in self.schemas.values()]

class FunctionParser:
    """Parser for extracting function calls from model output"""
    
    def __init__(self, registry: FunctionRegistry):
        self.registry = registry
    
    def parse(self, text: str) -> List[FunctionCall]:
        """
        Parse function calls from text.
        
        Supports multiple formats:
        1. JSON format: {"name": "function_name", "arguments": {...}}
        2. Markdown code block format: ```json\n{"name": "function_name", "arguments": {...}}\n```
        3. Function call format: function_name(arg1="value1", arg2="value2")
        """
        function_calls = []
        
        # Try to extract JSON function calls
        function_calls.extend(self._parse_json_format(text))
        
        # Try to extract code block function calls
        function_calls.extend(self._parse_code_blocks(text))
        
        # Try to extract function call syntax
        function_calls.extend(self._parse_function_syntax(text))
        
        return function_calls
    
    def _parse_json_format(self, text: str) -> List[FunctionCall]:
        """Parse JSON format function calls"""
        calls = []
        
        # Look for patterns that might be JSON objects
        json_candidates = re.findall(r'(\{[^{}]*"name"[^{}]*"arguments"[^{}]*\})', text)
        
        for candidate in json_candidates:
            try:
                data = json.loads(candidate)
                if "name" in data and "arguments" in data:
                    name = data["name"]
                    
                    # Verify the function exists in registry
                    if self.registry.get_function(name):
                        calls.append(FunctionCall(
                            name=name,
                            arguments=data["arguments"],
                            raw_text=candidate
                        ))
            except json.JSONDecodeError:
                continue
        
        return calls
    
    def _parse_code_blocks(self, text: str) -> List[FunctionCall]:
        """Parse function calls in code blocks"""
        calls = []
        
        # Find code blocks that look like function calls
        code_blocks = re.findall(r'```(?:json|python)?\s*\n(.*?)\n```', text, re.DOTALL)
        
        for block in code_blocks:
            try:
                # Try to parse as JSON
                data = json.loads(block)
                if isinstance(data, dict) and "name" in data and "arguments" in data:
                    name = data["name"]
                    
                    # Verify the function exists in registry
                    if self.registry.get_function(name):
                        calls.append(FunctionCall(
                            name=name,
                            arguments=data["arguments"],
                            raw_text=block
                        ))
            except json.JSONDecodeError:
                # If not valid JSON, try to parse function call syntax
                function_syntax_calls = self._parse_function_syntax(block)
                calls.extend(function_syntax_calls)
        
        return calls
    
    def _parse_function_syntax(self, text: str) -> List[FunctionCall]:
        """Parse function call syntax: function_name(arg1="value1", arg2=42)"""
        calls = []
        
        # Pattern to match function call syntax
        pattern = r'(\w+)\s*\(((?:[^,]+=[^,]+(?:,\s*)?)+)\)'
        matches = re.findall(pattern, text)
        
        for name, args_str in matches:
            # Verify the function exists in registry
            if not self.registry.get_function(name):
                continue
                
            # Parse arguments
            arguments = {}
            args_pattern = r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|\d+|\w+)'
            arg_matches = re.findall(args_pattern, args_str)
            
            for arg_name, arg_value in arg_matches:
                # Remove quotes from string values
                if (arg_value.startswith('"') and arg_value.endswith('"')) or \
                   (arg_value.startswith("'") and arg_value.endswith("'")):
                    arg_value = arg_value[1:-1]
                # Convert numeric values
                elif arg_value.isdigit():
                    arg_value = int(arg_value)
                elif arg_value.lower() == 'true':
                    arg_value = True
                elif arg_value.lower() == 'false':
                    arg_value = False
                
                arguments[arg_name] = arg_value
            
            calls.append(FunctionCall(
                name=name,
                arguments=arguments,
                raw_text=f"{name}({args_str})"
            ))
        
        return calls

class FunctionExecutor:
    """Executor for function calls"""
    
    def __init__(self, registry: FunctionRegistry):
        self.registry = registry
    
    def execute(self, function_call: FunctionCall) -> Dict[str, Any]:
        """
        Execute a function call.
        
        Args:
            function_call: The function call to execute
            
        Returns:
            result: Dictionary with execution results
        """
        function_name = function_call.name
        arguments = function_call.arguments
        
        # Get the function from registry
        func = self.registry.get_function(function_name)
        if not func:
            return {
                "status": "error",
                "error": f"Function '{function_name}' not found in registry"
            }
        
        # Validate arguments
        schema = self.registry.get_schema(function_name)
        validation_result = self._validate_arguments(schema, arguments)
        if "error" in validation_result:
            return {
                "status": "error",
                "error": validation_result["error"]
            }
        
        # Execute the function
        try:
            result = func(**arguments)
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": f"Execution error: {str(e)}"
            }
    
    def _validate_arguments(self, schema: FunctionSchema, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate function arguments against the schema"""
        # Check required parameters
        for param in schema.required_parameters:
            if param not in arguments:
                return {"error": f"Missing required parameter: {param}"}
        
        # Validate parameter types
        for param_name, param_value in arguments.items():
            if param_name not in schema.parameters:
                return {"error": f"Unknown parameter: {param_name}"}
            
            param_schema = schema.parameters[param_name]
            validation_result = self._validate_parameter(param_value, param_schema)
            if not validation_result:
                return {"error": f"Invalid value for parameter {param_name}: {param_value}"}
        
        return {"status": "valid"}
    
    def _validate_parameter(self, value: Any, schema: Dict[str, Any]) -> bool:
        """Validate a parameter value against its schema"""
        schema_type = schema.get("type")
        
        if schema_type == "string":
            return isinstance(value, str)
        elif schema_type == "integer":
            return isinstance(value, int)
        elif schema_type == "number":
            return isinstance(value, (int, float))
        elif schema_type == "boolean":
            return isinstance(value, bool)
        elif schema_type == "array":
            if not isinstance(value, list):
                return False
            
            # Validate array items if schema specifies item type
            if "items" in schema and value:
                for item in value:
                    if not self._validate_parameter(item, schema["items"]):
                        return False
        
        # For object type or unknown types, accept anything
        return True

class FunctionCallingManager:
    """Manager for the function calling system"""
    
    def __init__(self):
        self.registry = FunctionRegistry()
        self.parser = FunctionParser(self.registry)
        self.executor = FunctionExecutor(self.registry)
    
    def register_function(self, func=None, **kwargs):
        """Register a function"""
        return self.registry.register(func, **kwargs)
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered functions"""
        return self.registry.get_all_schemas()
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text to extract and execute function calls.
        
        Args:
            text: Model output text that may contain function calls
            
        Returns:
            result: Processing results including function calls and execution results
        """
        # Extract function calls
        function_calls = self.parser.parse(text)
        
        results = []
        for call in function_calls:
            # Execute each function call
            execution_result = self.executor.execute(call)
            
            results.append({
                "function": call.name,
                "arguments": call.arguments,
                "raw_text": call.raw_text,
                "result": execution_result
            })
        
        return {
            "original_text": text,
            "function_calls_detected": len(function_calls) > 0,
            "function_calls": results
        }
    
    def create_function_prompt(self) -> str:
        """Create a prompt instructing the model on available functions"""
        schemas = self.get_all_schemas()
        
        if not schemas:
            return ""
        
        prompt = "You can use the following functions by calling them in your response:\n\n"
        
        for schema in schemas:
            prompt += f"Function: {schema['name']}\n"
            prompt += f"Description: {schema['description']}\n"
            
            prompt += "Parameters:\n"
            properties = schema['parameters']['properties']
            required = schema['parameters'].get('required', [])
            
            for param_name, param_schema in properties.items():
                required_str = " (required)" if param_name in required else ""
                param_type = param_schema.get('type', 'any')
                param_desc = param_schema.get('description', '')
                
                prompt += f"  - {param_name}: {param_type}{required_str}"
                if param_desc:
                    prompt += f" - {param_desc}"
                prompt += "\n"
            
            prompt += "\n"
        
        prompt += "To call a function, use the following format:\n"
        prompt += "```json\n{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}\n```\n\n"
        
        return prompt

# Example usage:
"""
# Create a function calling manager
function_manager = FunctionCallingManager()

# Register functions
@function_manager.register_function(description="Search for information on the web")
def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    # Implementation would connect to a search API
    return [{"title": "Example result", "url": "https://example.com", "snippet": "Example text"}]

@function_manager.register_function(description="Get current weather for a location")
def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    # Implementation would connect to a weather API
    return {"temperature": 22, "conditions": "sunny", "location": location}

# Use in model pipeline
model_output = "To answer your question, I need to check the weather.\n\n```json\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"New York\"}}\n```"
result = function_manager.process_text(model_output)

# Execute functions and get results
if result["function_calls_detected"]:
    # Process results and incorporate into response
    weather_data = result["function_calls"][0]["result"]["result"]
    response = f"The current temperature in {weather_data['location']} is {weather_data['temperature']}°C and conditions are {weather_data['conditions']}."
""" 