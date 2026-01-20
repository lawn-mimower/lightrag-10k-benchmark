#!/bin/bash

echo "Setting up visualization dependencies for Gemini benchmark"
echo "=========================================================="
echo ""

# Check if matplotlib is installed
python3 -c "import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing matplotlib..."
    pip install matplotlib
else
    echo "✓ matplotlib is already installed"
fi

# Check if seaborn is installed
python3 -c "import seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing seaborn..."
    pip install seaborn
else
    echo "✓ seaborn is already installed"
fi

# Check if numpy is installed
python3 -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing numpy..."
    pip install numpy
else
    echo "✓ numpy is already installed"
fi

echo ""
echo "All dependencies installed!"
echo ""
echo "You can now run:"
echo "  python3 visualize_gemini_benchmark.py"
echo ""
echo "Or specify a specific benchmark file:"
echo "  python3 visualize_gemini_benchmark.py gemini_benchmark_20240120_123456.json"