#!/usr/bin/env python3
"""
generate_figures.py — Generate all mandatory plots for federated learning experiments.

Usage:
    # Generate all figures with FedAsync as proposed method
    python generate_figures.py
    
    # Generate with FedCS as proposed method instead
    python generate_figures.py --proposed fedasync
    
    # Show plots interactively (in addition to saving)
    python generate_figures.py --show

Output:
    - All figures saved to results/plots/
    - Results table saved to results/metrics/results_table.csv
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    generate_all_figures,
    plot_figure1_accuracy,
    plot_figure2_loss,
    plot_figure3_system_heterogeneity,
    plot_figure4_iid_vs_noniid,
    plot_figure5_fedavg_vs_proposed,
    generate_results_table,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate all mandatory plots for federated learning experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_figures.py
      → Generates all 5 figures using FedAsync as the proposed method
  
  python generate_figures.py --proposed fedcs
      → Generates all 5 figures using FedCS as the proposed method
  
  python generate_figures.py --show
      → Generates and displays all figures interactively
        """
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to results directory (default: results)"
    )
    
    parser.add_argument(
        "--proposed",
        type=str,
        default="fedasync",
        choices=["fedasync", "fedcs"],
        help="Proposed method to compare against FedAvg (default: fedasync)"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (in addition to saving)"
    )
    
    parser.add_argument(
        "--figure",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Generate only a specific figure (1-5)"
    )
    
    args = parser.parse_args()
    
    results_dir = args.results_dir
    proposed_method = args.proposed
    show_plots = args.show
    
    print("\n" + "=" * 80)
    print("MANDATORY FEDERATED LEARNING PLOTS")
    print("=" * 80)
    print(f"\nResults directory: {results_dir}")
    print(f"Proposed method: {proposed_method.upper()}")
    print(f"Display plots: {show_plots}\n")
    
    try:
        if args.figure:
            # Generate specific figure only
            if args.figure == 1:
                print("Generating Figure 1: Global Accuracy vs Communication Rounds...")
                plot_figure1_accuracy(results_dir, show_plots)
            elif args.figure == 2:
                print("Generating Figure 2: Global Loss vs Communication Rounds...")
                plot_figure2_loss(results_dir, show_plots)
            elif args.figure == 3:
                print("Generating Figure 3: Category 7 - System Heterogeneity Metrics...")
                plot_figure3_system_heterogeneity(results_dir, show_plots)
            elif args.figure == 4:
                print("Generating Figure 4: IID vs Non-IID Comparison...")
                plot_figure4_iid_vs_noniid(results_dir, show_plots)
            elif args.figure == 5:
                print(f"Generating Figure 5: FedAvg vs {proposed_method.upper()}...")
                plot_figure5_fedavg_vs_proposed(results_dir, proposed_method, show_plots)
        else:
            # Generate all figures
            print("📊 Generating ALL mandatory figures...\n")
            
            print("  [1/6] Figure 1: Global Accuracy vs Communication Rounds...")
            plot_figure1_accuracy(results_dir, show_plots)
            
            print("  [2/6] Figure 2: Global Loss vs Communication Rounds...")
            plot_figure2_loss(results_dir, show_plots)
            
            print("  [3/6] Figure 3: Category 7 - System Heterogeneity Metrics...")
            plot_figure3_system_heterogeneity(results_dir, show_plots)
            
            print("  [4/6] Figure 4: IID vs Non-IID Comparison...")
            plot_figure4_iid_vs_noniid(results_dir, show_plots)
            
            print(f"  [5/6] Figure 5: FedAvg vs {proposed_method.upper()}...")
            plot_figure5_fedavg_vs_proposed(results_dir, proposed_method, show_plots)
            
            print("  [6/6] Generating Results Table...")
            generate_results_table(results_dir)
            
            print("\n" + "=" * 80)
            print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
            print("=" * 80)
            print(f"\n📁 Saved to: {results_dir}/plots/")
            print(f"📊 Results table: {results_dir}/metrics/results_table.csv")
            print("\nGenerated files:")
            print("  • figure1_accuracy_vs_rounds.{pdf,png}")
            print("  • figure2_loss_vs_rounds.{pdf,png}")
            print("  • figure3_system_heterogeneity.{pdf,png}")
            print("  • figure4_iid_vs_noniid.{pdf,png}")
            print(f"  • figure5_fedavg_vs_{proposed_method}.{{pdf,png}}")
            print("  • results_table.csv")
            print()
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
