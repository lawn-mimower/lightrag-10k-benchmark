#!/bin/bash

# Script to run Gemini API benchmark with proper configuration

echo "========================================"
echo "Gemini-3-Flash-Preview API Benchmark"
echo "========================================"
echo ""

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY environment variable is not set"
    echo "Please set it using:"
    echo "  export GEMINI_API_KEY='your-api-key-here'"
    exit 1
fi

echo "✓ GEMINI_API_KEY is set"
echo ""

# Default configuration
SAMPLE_SIZE=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --samples N    Number of questions to test (default: 10)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Model: gemini-3-flash-preview"
echo "  Sample Size: $SAMPLE_SIZE questions"
echo "  Modes: local, global, naive, hybrid, mix"
echo "  Total API Calls: $(($SAMPLE_SIZE * 5))"
echo ""

# Run the benchmark
echo "Starting benchmark..."
echo ""

SAMPLE_SIZE="$SAMPLE_SIZE" python3 benchmark_gemini_api.py

echo ""
echo "Benchmark complete!"