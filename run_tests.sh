#!/bin/bash
# Test runner script for voice-agi-mcp

set -e

echo "================================================"
echo "Voice-AGI MCP Test Suite"
echo "================================================"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -q fastmcp edge-tts httpx pynput pytest pytest-asyncio pytest-cov pytest-mock
fi

# Set PYTHONPATH
export PYTHONPATH=src

# Run tests
echo ""
echo "Running tests..."
echo ""

python -m pytest tests/ -v \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml \
    -k "not mcp_tools and not error_handling" \
    "$@"

echo ""
echo "================================================"
echo "Test run complete!"
echo "Coverage report: htmlcov/index.html"
echo "================================================"
