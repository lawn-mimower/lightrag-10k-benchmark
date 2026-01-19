#!/bin/bash

echo "=================================="
echo "SIMPLE RAGAS BATCH EVALUATION"
echo "=================================="
echo ""
echo "Configuration:"
echo "• 1 file at a time (sequential)"
echo "• 5 modes evaluated in parallel per file"
echo "• 120 second timeout per evaluation"
echo "• Automatic checkpoint/resume support"
echo ""

# Activate virtual environment if it exists
if [ -d "ml-env" ]; then
    echo "Activating ml-env..."
    source ml-env/bin/activate
elif [ -d "venv" ]; then
    echo "Activating venv..."
    source venv/bin/activate
fi

# Check if checkpoint exists
if [ -f "batch_ragas_checkpoint.json" ]; then
    echo "📚 Found checkpoint - will resume from last position"
    echo ""
fi

# Run the evaluation
python3 batch_ragas_evaluation_simple.py

echo ""
echo "Complete! Check batch_ragas_evaluation_results_simple.json for results."