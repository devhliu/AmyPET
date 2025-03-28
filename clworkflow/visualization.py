"""
Visualization tools for centiloid results.

This module provides functions for visualizing centiloid calculation results.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def identity_line(ax=None, ls='--', *args, **kwargs):
    """
    Plot identity line for correlation analysis.
    
    Args:
        ax: Matplotlib axis
        ls: Line style
        *args, **kwargs: Additional arguments for plot
        
    Returns:
        Matplotlib axis
    """
    ax = ax or plt.gca()
    identity, = ax.plot([], [], ls=ls, *args, **kwargs)

    def callback(axes):
        low_x, high_x = ax.get_xlim()
        low_y, high_y = ax.get_ylim()
        low = min(low_x, low_y)
        high = max(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(ax)
    ax.callbacks.connect('xlim_changed', callback)
    ax.callbacks.connect('ylim_changed', callback)
    return ax

def plot_centiloid_comparison(pib_values, new_values, tracer_name, 
                             reference_region='wc', outpath=None):
    """
    Plot comparison between PiB and new tracer centiloid values.
    
    Args:
        pib_values: List of PiB centiloid values
        new_values: List of new tracer centiloid values
        tracer_name: Name of the new tracer
        reference_region: Reference region used
        outpath: Output directory for saving the plot
        
    Returns:
        Matplotlib figure
    """
    from scipy.stats import linregress
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = linregress(pib_values, new_values)
    r2 = r_value**2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot data points
    ax.scatter(pib_values, new_values, c='black')
    
    # Plot identity line
    identity_line(ax=ax, ls='--', c='b')
    
    # Plot regression line
    x_range = np.linspace(min(pib_values), max(pib_values), 100)
    y_range = slope * x_range + intercept
    ax.plot(x_range, y_range, 'r-', label=f'y = {slope:.3f}x + {intercept:.3f}')
    
    # Add labels and title
    ax.set_xlabel('PiB Centiloid Values')
    ax.set_ylabel(f'{tracer_name} Centiloid Values')
    ax.set_title(f'Comparison of PiB and {tracer_name} Centiloid Values\n'
                f'Reference Region: {reference_region}')
    
    # Add R² value
    ax.text(0.05, 0.95, f'$R^2 = {r2:.4f}$', transform=ax.transAxes, 
            fontsize=12, verticalalignment='top')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure if outpath is provided
    if outpath:
        outpath = Path(outpath)
        outpath.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath / f"{tracer_name}_vs_pib_{reference_region}.png", dpi=300)
    
    return fig

def plot_group_comparison(results, groups, reference_region='wc', outpath=None):
    """
    Plot comparison of centiloid values between different groups.
    
    Args:
        results: Dictionary of centiloid results
        groups: Dictionary mapping group names to subject filters
        reference_region: Reference region to use
        outpath: Output directory for saving the plot
        
    Returns:
        Matplotlib figure
    """
    from scipy.stats import ttest_ind
    
    # Extract centiloid values for each group
    group_values = {}
    for group_name, filter_key in groups.items():
        group_values[group_name] = []
        for subject_key, subject_data in results.items():
            if filter_key in subject_key.lower():
                group_values[group_name].append(subject_data['cl'][reference_region])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot box plots
    positions = range(len(group_values))
    box = ax.boxplot([group_values[g] for g in groups], positions=positions, patch_artist=True)
    
    # Customize box plots
    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
    
    # Add individual data points
    for i, (group_name, values) in enumerate(group_values.items()):
        # Add jitter to x-positions
        x = np.random.normal(i, 0.05, size=len(values))
        ax.scatter(x, values, alpha=0.6, s=20, edgecolor='black', linewidth=0.5)
    
    # Add labels and title
    ax.set_xlabel('Group')
    ax.set_ylabel(f'Centiloid Value ({reference_region})')
    ax.set_title(f'Comparison of Centiloid Values Between Groups\n'
                f'Reference Region: {reference_region}')
    
    # Set x-tick labels
    ax.set_xticks(positions)
    ax.set_xticklabels(groups.keys())
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add statistics
    if len(groups) == 2:
        group_names = list(groups.keys())
        t_stat, p_value = ttest_ind(
            group_values[group_names[0]], 
            group_values[group_names[1]], 
            equal_var=False
        )
        ax.text(0.5, 0.95, f't-test: p = {p_value:.4f}', transform=ax.transAxes, 
                fontsize=12, horizontalalignment='center', verticalalignment='top')
    
    # Save figure if outpath is provided
    if outpath:
        outpath = Path(outpath)
        outpath.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath / f"group_comparison_{reference_region}.png", dpi=300)
    
    return fig