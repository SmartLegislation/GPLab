#!/usr/bin/env python3
"""
Medical Insurance Policy Simulation Dashboard Launcher

This script launches the Streamlit dashboard for visualizing simulation results.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit dashboard"""
    
    # Get the current directory (should be src/web)
    current_dir = Path(__file__).parent
    
    # Check if we're in the right directory
    if not (current_dir / "app.py").exists():
        print("❌ Error: app.py not found in current directory")
        print(f"Current directory: {current_dir}")
        print("Please run this script from the src/web directory")
        sys.exit(1)
    
    # Check if results directory exists
    results_dir = current_dir.parent / "results"
    if not results_dir.exists():
        print("⚠️  Warning: Results directory not found")
        print(f"Expected location: {results_dir}")
        print("Please run a simulation first to generate results")
        print()
    
    # Check for required packages
    required_packages = [
        "streamlit",
        "plotly", 
        "pandas",
        "pyyaml",
        "openai"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print()
        print("Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        print("Or install all requirements with:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    print("🔬 GPLab Policy Simulation Dashboard")
    print("=" * 50)
    print("🚀 Starting Streamlit application...")
    print()
    print("📊 Dashboard will open in your default browser")
    print("🌐 URL: http://localhost:8501")
    print()
    print("💡 Tips:")
    print("  - Use Ctrl+C to stop the server")
    print("  - Refresh the page if you encounter issues")
    print("  - Check the terminal for error messages")
    print()
    
    # Launch Streamlit
    try:
        # Change to the web directory
        os.chdir(current_dir)
        
        # Run Streamlit with optimized settings
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "false"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 