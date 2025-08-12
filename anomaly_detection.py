#!/usr/bin/env python3
"""
Clean entry point for the Credit Card Anomaly Detection Application.

This is the single entry point that provides all functionality
while keeping the codebase organized in the src/ package.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.cli.main import cli

if __name__ == "__main__":
    cli()
