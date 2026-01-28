#!/bin/bash

# Ensure the script is run as root (for global installation)
if [ "$EUID" -ne 0 ]; then 
  echo "Please run as root (use sudo)"
  exit 1
fi

echo "=============================================="
echo "   md2pdf Universal Installer"
echo "=============================================="

# --- 1. Install System Dependencies ---
echo "[1/3] Installing System Dependencies (Pandoc)..."
apt-get update -qq
apt-get install -y pandoc python3-pip > /dev/null

# --- 2. Install Python Dependencies Globally ---
echo "[2/3] Installing Python Dependencies (WeasyPrint)..."
# We use --break-system-packages to force global install on modern Ubuntu
# (Safe here because we are explicitly setting up a global tool)
pip3 install weasyprint --break-system-packages

# --- 3. Create the Universal Script ---
echo "[3/3] Creating /usr/local/bin/md2pdf..."

# We embed the Python code directly into the destination file
cat << 'EOF' > /usr/local/bin/md2pdf
#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse
import shutil
import tempfile

# --- Embedded CSS for Professional Tables ---
CSS_CONTENT = """
body { font-family: "Helvetica", "Arial", sans-serif; font-size: 10pt; margin: 2cm; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
th, td { border: 1px solid #444; padding: 6px; text-align: left; vertical-align: top; }
th { background-color: #f2f2f2; font-weight: bold; }
h1, h2, h3 { color: #2c3e50; }
code { background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
"""

def convert(input_path, output_path):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.css', delete=False) as temp_css:
        temp_css.write(CSS_CONTENT)
        css_path = temp_css.name

    try:
        cmd = [
            "pandoc", input_path, "-o", output_path,
            "--pdf-engine=weasyprint", f"--css={css_path}",
            "--metadata", "margin=1in"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Success: {output_path}")
        else:
            print(f"✗ Error converting {input_path}:\n{result.stderr}")
            sys.exit(1)
    finally:
        if os.path.exists(css_path): os.remove(css_path)

def main():
    parser = argparse.ArgumentParser(description="Convert Markdown to PDF globally")
    parser.add_argument("input", help="Input markdown file")
    parser.add_argument("output", nargs="?", help="Output PDF file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("Error: File not found.")
        sys.exit(1)

    output = args.output if args.output else os.path.splitext(args.input)[0] + ".pdf"
    convert(args.input, output)

if __name__ == "__main__":
    main()
EOF

# --- 4. Make it Executable ---
chmod +x /usr/local/bin/md2pdf

echo "=============================================="
echo "Success! Installation Complete."
echo "You can now run 'md2pdf' from ANY folder."
echo "Usage: md2pdf myfile.md"
echo "=============================================="
