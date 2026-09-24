#!/usr/bin/env python3
"""
Script to analyze the 'references' column from finder_train.parquet
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the parquet file
print("Loading finder_train.parquet...")
# Usage: python analyze_references.py [finder_train.parquet] [output.png]
df = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else 'finder_train.parquet')

print(f"Total rows in dataset: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print("\n" + "="*80)

# Extract the references column
print("\n1. REFERENCES COLUMN - BASIC INFO")
print("="*80)
references_df = pd.DataFrame({'references': df['references']})
print(references_df.info())

# Create additional metrics for the references
print("\n2. REFERENCES COLUMN - DESCRIPTIVE STATISTICS")
print("="*80)

# Calculate the number of references per row
references_df['num_references'] = references_df['references'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)

# Calculate total character length of all references per row
references_df['total_char_length'] = references_df['references'].apply(
    lambda x: sum(len(ref) for ref in x) if isinstance(x, (list, np.ndarray)) else 0
)

# Calculate average character length per reference
references_df['avg_char_length_per_ref'] = references_df.apply(
    lambda row: row['total_char_length'] / row['num_references'] if row['num_references'] > 0 else 0,
    axis=1
)

# Display describe() on the numeric metrics
print("\nNumeric statistics for references:")
print(references_df[['num_references', 'total_char_length', 'avg_char_length_per_ref']].describe())

print("\n" + "="*80)
print("\n3. SAMPLE DATA")
print("="*80)
print(f"\nFirst 3 rows of references (showing first 200 chars of each reference):")
for idx in range(min(3, len(references_df))):
    refs = references_df.iloc[idx]['references']
    print(f"\nRow {idx}: {len(refs)} reference(s)")
    for i, ref in enumerate(refs):
        print(f"  Reference {i+1}: {ref[:200]}...")

print("\n" + "="*80)
print("\n4. CHARACTER LENGTH DISTRIBUTION (GAUSSIAN CURVE)")
print("="*80)

# Create visualization of character length distribution
fig, ax = plt.subplots(figsize=(12, 6))

# Get the total character length data
char_lengths = references_df['total_char_length'].values

# Create histogram
n, bins, patches = ax.hist(char_lengths, bins=50, density=True, alpha=0.7,
                           color='skyblue', edgecolor='black', label='Actual Distribution')

# Fit a gaussian (normal) distribution to the data
mu, std = stats.norm.fit(char_lengths)

# Create x values for the gaussian curve
x = np.linspace(char_lengths.min(), char_lengths.max(), 100)

# Calculate the gaussian curve
gaussian_curve = stats.norm.pdf(x, mu, std)

# Plot the gaussian curve
ax.plot(x, gaussian_curve, 'r-', linewidth=2, label=f'Gaussian Fit\n(μ={mu:.1f}, σ={std:.1f})')

# Add labels and title
ax.set_xlabel('Total Character Length', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Distribution of Reference Character Lengths with Gaussian Fit', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Save the plot
output_file = sys.argv[2] if len(sys.argv) > 2 else 'references_char_length_distribution.png'
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nGaussian curve visualization saved to: {output_file}")

# Print statistical information
print(f"\nGaussian fit parameters:")
print(f"  Mean (μ): {mu:.2f} characters")
print(f"  Standard Deviation (σ): {std:.2f} characters")
print(f"  68% of data falls between: {mu-std:.2f} and {mu+std:.2f}")
print(f"  95% of data falls between: {mu-2*std:.2f} and {mu+2*std:.2f}")

# Calculate skewness and kurtosis to assess normality
skewness = stats.skew(char_lengths)
kurtosis = stats.kurtosis(char_lengths)
print(f"\nDistribution shape metrics:")
print(f"  Skewness: {skewness:.3f} (0 = perfectly symmetric)")
print(f"  Kurtosis: {kurtosis:.3f} (0 = normal distribution)")

print("\n" + "="*80)
print("Analysis complete!")
