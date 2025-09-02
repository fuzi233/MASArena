"""
Multi-Agent Systems for Benchmarking

This package dynamically discovers and registers agent systems.
By importing this package, all agent system modules are automatically loaded,
which triggers the registration of their respective systems via decorators.
"""

import pkgutil
import importlib
import traceback

# Import the base classes and registry
from .base import AgentSystem, AgentSystemRegistry, create_agent_system

# --- Dynamic Discovery and Registration ---
# Iterate over all modules in the current package path
# and import them. This is what triggers the registration decorators
# in each agent system file to run and register themselves.
for _, name, _ in pkgutil.iter_modules(__path__):
    # Ensure we don't try to import the base module itself again
    # or any other non-agent system modules.
    if name not in ['base', 'format_prompts']:
            importlib.import_module(f".{name}", __package__)

# --- Public API ---
# Expose the populated registry and convenient dictionaries for the application.

# Get all registered agent systems
AVAILABLE_AGENT_SYSTEMS = AgentSystemRegistry.get_all_systems()

# Define what gets imported when a user does 'from mas_arena.agents import *'
__all__ = [
    "AgentSystem",
    "AgentSystemRegistry",
    "create_agent_system",
    "AVAILABLE_AGENT_SYSTEMS",
]
