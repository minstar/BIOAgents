"""Tool framework for BIOAgents domains.

Provides the base toolkit class, tool decorator, and tool type definitions.
Follows τ²-bench patterns adapted for medical domains.
"""

import inspect
import json
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, get_type_hints

from pydantic import BaseModel, Field

from bioagents.environment.db import DB

TOOL_ATTR = "__bioagent_tool__"
TOOL_TYPE_ATTR = "__bioagent_tool_type__"

T = TypeVar("T", bound=DB)


class ToolType(str, Enum):
    """Type of a tool."""
    READ = "read"        # Read-only (queries, lookups)
    WRITE = "write"      # Modifies state (orders, prescriptions)
    THINK = "think"      # Internal reasoning
    GENERIC = "generic"  # General purpose (transfer, calculate)


def is_tool(tool_type: ToolType = ToolType.READ):
    """Decorator to mark a method as an available tool.
    
    Args:
        tool_type: The type of tool (READ, WRITE, THINK, GENERIC)
    
    Usage:
        @is_tool(ToolType.READ)
        def search_pubmed(self, query: str) -> list[dict]:
            ...
    """
    def decorator(func):
        setattr(func, TOOL_ATTR, True)
        setattr(func, TOOL_TYPE_ATTR, tool_type)
        return func
    return decorator


class ToolDefinition(BaseModel):
    """OpenAI-compatible tool definition for function calling."""
    type: str = "function"
    function: dict = Field(description="Function specification")

    @classmethod
    def from_method(cls, name: str, method: Callable) -> "ToolDefinition":
        """Create a tool definition from a method's signature and docstring."""
        sig = inspect.signature(method)
        doc = inspect.getdoc(method) or ""
        
        # Parse docstring for description and args
        lines = doc.strip().split("\n")
        description = lines[0] if lines else name
        
        # Build parameters from signature
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            # Get type annotation
            param_type = "string"
            annotation = param.annotation
            if annotation != inspect.Parameter.empty:
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object",
                }
                # Handle basic types
                origin = getattr(annotation, "__origin__", None)
                if origin is list:
                    param_type = "array"
                elif origin is dict:
                    param_type = "object"
                elif annotation in type_map:
                    param_type = type_map[annotation]
                else:
                    param_type = "string"
            
            # Parse description from docstring Args section
            param_desc = f"Parameter: {param_name}"
            in_args = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("Args:"):
                    in_args = True
                    continue
                if in_args and stripped.startswith(f"{param_name}:"):
                    param_desc = stripped.split(":", 1)[1].strip()
                    break
                if in_args and stripped.startswith("Returns:"):
                    break
            
            properties[param_name] = {
                "type": param_type,
                "description": param_desc,
            }
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return cls(
            type="function",
            function={
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        )


class _ToolKitMeta(type):
    """Metaclass that collects @is_tool decorated methods."""
    
    def __init__(cls, name, bases, attrs):
        func_tools = {}
        for attr_name, method in attrs.items():
            if isinstance(method, property):
                method = method.fget
            if hasattr(method, TOOL_ATTR):
                func_tools[attr_name] = method
        
        # Store discovered tools
        cls._declared_tools = func_tools
        super().__init__(name, bases, attrs)


class ToolKitBase(metaclass=_ToolKitMeta):
    """Base class for domain tool kits.
    
    Subclass this to define domain-specific tools:
    
        class ClinicalTools(ToolKitBase):
            db: ClinicalDB
            
            @is_tool(ToolType.READ)
            def get_patient_info(self, patient_id: str) -> dict:
                ...
    """
    
    def __init__(self, db: Optional[T] = None):
        self.db: Optional[T] = db
    
    @property
    def tools(self) -> Dict[str, Callable]:
        """Get all registered tool methods."""
        result = {}
        # Collect from all classes in MRO
        for klass in type(self).__mro__:
            if hasattr(klass, "_declared_tools"):
                for name in klass._declared_tools:
                    if name not in result:
                        result[name] = getattr(self, name)
        return result
    
    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name with given arguments."""
        if tool_name not in self.tools:
            available = list(self.tools.keys())
            raise ValueError(
                f"Tool '{tool_name}' not found. Available tools: {available}"
            )
        return self.tools[tool_name](**kwargs)
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool exists."""
        return tool_name in self.tools
    
    def get_tool_type(self, tool_name: str) -> ToolType:
        """Get the type of a tool."""
        method = self.tools.get(tool_name)
        if method is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return getattr(method, TOOL_TYPE_ATTR, ToolType.GENERIC)
    
    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get OpenAI-compatible tool definitions for all tools."""
        definitions = []
        for name, method in sorted(self.tools.items()):
            definitions.append(ToolDefinition.from_method(name, method))
        return definitions
    
    def get_tool_definitions_dict(self) -> list[dict]:
        """Get tool definitions as a list of dicts (for JSON serialization)."""
        return [td.model_dump() for td in self.get_tool_definitions()]
    
    def get_db_hash(self) -> Optional[str]:
        """Get hash of the underlying database."""
        if self.db is None:
            return None
        return self.db.get_hash()
    
    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the toolkit."""
        tools = self.tools
        return {
            "num_tools": len(tools),
            "num_read_tools": sum(1 for n in tools if self.get_tool_type(n) == ToolType.READ),
            "num_write_tools": sum(1 for n in tools if self.get_tool_type(n) == ToolType.WRITE),
            "num_think_tools": sum(1 for n in tools if self.get_tool_type(n) == ToolType.THINK),
            "num_generic_tools": sum(1 for n in tools if self.get_tool_type(n) == ToolType.GENERIC),
            "tool_names": sorted(tools.keys()),
        }
