import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.gridspec as gridspec
import pymc as pm
import arviz as az


# Configuration
DATA_FILE = 'data.csv'
OUTPUT_DIR = 'final'


# Mapping dictionaries
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}


CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}


PERCENTILES = [10, 30, 50, 70, 90]


#Suggested by AI. Implemented partially manually.
def simulate_ddm(params, n_trials, dt=0.001, noise_std=1.0):
    """
    Simulate the Drift Diffusion Model (DDM)
   
    Parameters:
    params (dict): DDM parameters including:
        - drift_rate_v (float): Drift rate
        - boundary_sep_a (float): Boundary separation
        - non_decision_t0 (float): Non-decision time
        - relative_bias_b (float): Relative starting point bias (0-1)
    n_trials (int): Number of trials to simulate
    dt (float): Time step size
    noise_std (float): Standard deviation of noise
   
    Returns:
    rts (ndarray): Response times
    choices (ndarray): Choices (0 for lower boundary, 1 for upper boundary)
    """
    # Extract parameters
    v = params['drift_rate_v']
    a = params['boundary_sep_a']
    t0 = params['non_decision_t0']
    b = params['relative_bias_b'] * a  # Convert to absolute bias
   
    # Initialize arrays
    rts = np.zeros(n_trials)
    choices = np.zeros(n_trials)
   
    for i in range(n_trials):
        # Initialize evidence at starting point
        evidence = b
        time_elapsed = 0
       
        # Simulate until a boundary is hit
        while True:
            # Update evidence: drift + noise
            evidence += v * dt + noise_std * np.sqrt(dt) * np.random.randn()
            time_elapsed += dt
           
            # Check boundaries
            if evidence >= a:
                choices[i] = 1  # Upper boundary choice
                break
            elif evidence <= 0:
                choices[i] = 0  # Lower boundary choice
                break
       
        # Add non-decision time
        rts[i] = time_elapsed + t0
   
    return rts, choices


def compute_delta_plot(rts, choices, percentiles=PERCENTILES):
    """Compute delta plot values with robust error handling""" #
    if len(rts) == 0:
        return percentiles, np.full(len(percentiles), np.nan)
   
    # Separate correct and error responses
    correct_mask = choices == 1
    error_mask = choices == 0
   
    # Calculate percentiles
    correct_perc = np.percentile(rts[correct_mask], percentiles) if sum(correct_mask) > 0 else np.full(len(percentiles), np.nan)
    error_perc = np.percentile(rts[error_mask], percentiles) if sum(error_mask) > 0 else np.full(len(percentiles), np.nan)
   
    # Calculate delta values (error RT - correct RT)
    delta = error_perc - correct_perc
   
    return percentiles, delta


def create_parameter_delta_plots():
    """Create delta plots for diffusion model parameters"""
    # Define parameter sets for the four comparisons
    param_sets = [
        # δ (Drift Rate) comparison
        {
            "name": "δ (Drift Rate)",
            "baseline": {"drift_rate_v": 2.0, "boundary_sep_a": 1.0,
                        "non_decision_t0": 0.25, "relative_bias_b": 0.5},
            "contrast": {"drift_rate_v": 1.0, "boundary_sep_a": 1.0,
                        "non_decision_t0": 0.25, "relative_bias_b": 0.5},
            "color_base": "#1f77b4",  # Blue
            "color_contrast": "#ff7f0e"  # Orange
        },
        # α (Boundary Separation) comparison
        {
            "name": "α (Boundary Separation)",
            "baseline": {"drift_rate_v": 1.0, "boundary_sep_a": 1.0,
                        "non_decision_t0": 0.25, "relative_bias_b": 0.5},
            "contrast": {"drift_rate_v": 1.0, "boundary_sep_a": 2.0,
                        "non_decision_t0": 0.25, "relative_bias_b": 0.5},
            "color_base": "#2ca02c",  # Green
            "color_contrast": "#d62728"  # Red
        },
        # τ (Non-decision Time) comparison - CORRECTED
        {
            "name": "τ (Non-decision Time)",
            "baseline": {"drift_rate_v": 1.0, "boundary_sep_a": 1.0,
                        "non_decision_t0": 0.25, "relative_bias_b": 0.5},
            "contrast": {"drift_rate_v": 1.0, "boundary_sep_a": 1.0,
                        "non_decision_t0": 0.50, "relative_bias_b": 0.5},
            "color_base": "#9467bd",  # Purple
            "color_contrast": "#8c564b"  # Brown
        },
        # β (Bias) comparison
        {
            "name": "β (Bias)",
            "baseline": {"drift_rate_v": 1.0, "boundary_sep_a": 1.0,
                        "non_decision_t0": 0.25, "relative_bias_b": 0.5},
            "contrast": {"drift_rate_v": 1.0, "boundary_sep_a": 1.0,
                        "non_decision_t0": 0.25, "relative_bias_b": 0.7},
            "color_base": "#e377c2",  # Pink
            "color_contrast": "#7f7f7f"  # Gray
        }
    ]


    # Create figure with improved layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Theoretical Delta Plots: Diffusion Model Parameter Effects",
                fontsize=20, fontweight='bold', y=0.97)
   
    # Set consistent y-axis limits Worked on partially by AI
    for ax in axes.flat:
        ax.set_ylim(-0.2, 0.8)
        ax.axhline(0, color="black", linestyle="--", alpha=0.4)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xticks(PERCENTILES)
        ax.set_xlabel("Response Time Percentile", fontsize=12)
        ax.set_ylabel("Δ RT (Error - Correct) (s)", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
   
    # Run simulations and create plots Suggested by AI
    for i, params in enumerate(param_sets):
        ax = axes[i//2, i%2]
       
        # Simulate baseline condition
        rts_base, choices_base = simulate_ddm(params["baseline"], 50000)
        perc_base, delta_base = compute_delta_plot(rts_base, choices_base)
       
        # Simulate contrast condition
        rts_contrast, choices_contrast = simulate_ddm(params["contrast"], 50000)
        perc_contrast, delta_contrast = compute_delta_plot(rts_contrast, choices_contrast)
       
        # Calculate error rates for annotation
        base_error_rate = np.mean(choices_base == 0) * 100
        contrast_error_rate = np.mean(choices_contrast == 0) * 100
       
        # Plot baseline
        ax.plot(perc_base, delta_base, 'o-', linewidth=2.5, markersize=8,
                color=params["color_base"],
                label=f"Baseline")
       
        # Plot contrast
        ax.plot(perc_contrast, delta_contrast, 's--', linewidth=2.5, markersize=8,
                color=params["color_contrast"],
                label=f"Contrast")
       
        # Add annotations with parameter values
        base_params = "\n".join([f"{k}: {v}" for k, v in params["baseline"].items()])
        contrast_params = "\n".join([f"{k}: {v}" for k, v in params["contrast"].items()])
       
        ax.text(0.05, 0.95, f"Baseline:\n{base_params}\nError: {base_error_rate:.1f}%",
                transform=ax.transAxes, color=params["color_base"], fontsize=9,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        ax.text(0.55, 0.95, f"Contrast:\n{contrast_params}\nError: {contrast_error_rate:.1f}%",
                transform=ax.transAxes, color=params["color_contrast"], fontsize=9,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
       
        # Format plot
        ax.set_title(f"Parameter: {params['name']}", fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", framealpha=0.8, fontsize=10)
   
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = Path(OUTPUT_DIR) / 'theoretical_delta_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Theoretical delta plots saved to {output_path}")


def main():
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
   
    # Create theoretical delta plots
    create_parameter_delta_plots()  
   
    print("DDM simulation and delta plot creation complete!")


if __name__ == "__main__":
    main()

