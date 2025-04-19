#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_benchmark_results(csv_file):
    """
    Read benchmark results from CSV and create plots.
    """
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        return
    
    # Read CSV file
    data = pd.read_csv(csv_file)
    
    # Extract kernel names from columns (remove _Time, _GFLOPS, _BW_Percent suffixes)
    kernel_names = []
    for col in data.columns:
        if col.endswith('_GFLOPS'):
            kernel_names.append(col[:-7])
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Colors and markers for each kernel
    colors = ['blue', 'orange', 'green', 'red']
    markers = ['o', 's', '^', 'D']
    
    # GFLOPS Comparison (left plot)
    for i, kernel in enumerate(kernel_names):
        ax1.plot(data['MatrixSize'], data[f'{kernel}_GFLOPS'], 
                marker=markers[i], linestyle='-' if i == 0 or i == 2 else '--', 
                color=colors[i], label=kernel)
    
    ax1.set_xlabel('Matrix Size (D³)')
    ax1.set_ylabel('Achieved GFLOPS')
    ax1.set_title('3D Convolution: GFLOPS Comparison')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_xscale('log')
    
    # Memory Bandwidth Comparison (right plot)
    for i, kernel in enumerate(kernel_names):
        ax2.plot(data['MatrixSize'], data[f'{kernel}_BW_Percent'], 
                marker=markers[i], linestyle='-' if i == 0 or i == 2 else '--', 
                color=colors[i], label=kernel)
    
    ax2.set_xlabel('Matrix Size (D³)')
    ax2.set_ylabel('Achieved Memory Bandwidth (%)')
    ax2.set_title('3D Convolution: Memory Bandwidth Comparison')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.set_xscale('log')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=300)
    print(f"Plot saved as benchmark_comparison.png")
    
    # Create additional plot comparing speedups relative to naive implementation
    plt.figure(figsize=(10, 6))
    
    baseline = data[f'{kernel_names[0]}_GFLOPS']
    for i, kernel in enumerate(kernel_names[1:], 1):  # Skip the first (naive) kernel
        speedup = data[f'{kernel}_GFLOPS'] / baseline
        plt.plot(data['MatrixSize'], speedup, 
                marker=markers[i], linestyle='-', color=colors[i], 
                label=f'{kernel} vs. {kernel_names[0]}')
    
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Matrix Size (D³)')
    plt.ylabel('Speedup (X times)')
    plt.title('3D Convolution: Speedup Relative to Naive Implementation')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('speedup_comparison.png', dpi=300)
    print(f"Speedup plot saved as speedup_comparison.png")
    
    # Try to display plot if running in interactive mode
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    plot_benchmark_results("benchmark_results.csv")
