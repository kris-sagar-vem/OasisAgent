"""Deterministic triage-and-allocation engine. No I/O, no LLM, no cloud."""
from .registry import load_builtins

load_builtins()
