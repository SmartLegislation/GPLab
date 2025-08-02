import streamlit as st
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

def show_simulation_selector(results_dir: Path) -> Optional[str]:
    """Display simulation selector and return selected simulation ID"""
    
    # Get available simulations
    simulations = get_available_simulations(results_dir)
    
    if not simulations:
        st.sidebar.error("❌ No simulation results found")
        st.sidebar.info("💡 Run a simulation first to generate results")
        return None
    
    # Sort simulations by date (newest first)
    simulations.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Create selection options
    simulation_options = []
    simulation_map = {}
    
    for sim in simulations:
        display_name = f"🕒 {sim['display_name']} ({sim['agent_count']} agents)"
        simulation_options.append(display_name)
        simulation_map[display_name] = sim['id']
    
    # Simulation selector
    st.sidebar.markdown("### 📊 Select Simulation")
    
    # Default to the most recent simulation
    default_index = 0
    
    selected_display = st.sidebar.selectbox(
        "Available Simulations:",
        simulation_options,
        index=default_index,
        key="simulation_selector"
    )
    
    if selected_display:
        selected_id = simulation_map[selected_display]
        
        # Display simulation info
        selected_sim = next(sim for sim in simulations if sim['id'] == selected_id)
        
        with st.sidebar.expander("📋 Simulation Details", expanded=False):
            st.write(f"**ID:** {selected_sim['id']}")
            st.write(f"**Date:** {selected_sim['date_str']}")
            st.write(f"**Agents:** {selected_sim['agent_count']}")
            st.write(f"**Epochs:** {selected_sim['epochs']}")
            st.write(f"**Subsystems:** {', '.join(selected_sim['subsystems'])}")
            
            # Database file info
            db_size = selected_sim['db_size']
            st.write(f"**DB Size:** {format_file_size(db_size)}")
        
        return selected_id
    
    return None

def get_available_simulations(results_dir: Path) -> List[Dict[str, Any]]:
    """Get list of available simulation results"""
    simulations = []
    
    try:
        # Scan results directory for simulation folders
        for item in results_dir.iterdir():
            if item.is_dir():
                db_file = item / "results.db"
                if db_file.exists():
                    sim_info = extract_simulation_info(item, db_file)
                    if sim_info:
                        simulations.append(sim_info)
    except Exception as e:
        st.error(f"Error scanning results directory: {e}")
    
    return simulations

def extract_simulation_info(sim_dir: Path, db_file: Path) -> Optional[Dict[str, Any]]:
    """Extract simulation information from directory and database"""
    try:
        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent
        sys.path.append(str(project_root))
        
        from src.web.utils.database_manager import DatabaseManager
        
        sim_id = sim_dir.name
        
        # Parse timestamp from simulation ID
        timestamp = parse_simulation_timestamp(sim_id)
        
        # Get database info
        db_size = db_file.stat().st_size
        
        # Quick database check
        db_manager = DatabaseManager(str(db_file))
        agent_count = db_manager.get_agent_count()
        max_epoch = db_manager.get_max_epoch()
        subsystems = db_manager.get_subsystem_names()
        db_manager.close()
        
        # Format display name
        if timestamp:
            date_str = timestamp.strftime("%Y-%m-%d %H:%M")
            display_name = f"{sim_id.split('_')[0]}_{date_str}"
        else:
            date_str = "Unknown"
            display_name = sim_id
        
        return {
            'id': sim_id,
            'timestamp': timestamp or datetime.min,
            'date_str': date_str,
            'display_name': display_name,
            'agent_count': agent_count,
            'epochs': (max_epoch + 1) if max_epoch is not None else 0,
            'subsystems': [s for s in subsystems if s != "token_usage"],
            'db_size': db_size,
            'db_path': str(db_file)
        }
        
    except Exception as e:
        print(f"Error extracting info for {sim_dir}: {e}")
        return None

def parse_simulation_timestamp(sim_id: str) -> Optional[datetime]:
    """Parse timestamp from simulation ID"""
    try:
        # Expected format: prefix_YYYYMMDD_HHMMSS
        parts = sim_id.split('_')
        if len(parts) >= 3:
            date_part = parts[-2]  # YYYYMMDD
            time_part = parts[-1]  # HHMMSS
            
            if len(date_part) == 8 and len(time_part) == 6:
                timestamp_str = f"{date_part}_{time_part}"
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except Exception:
        pass
    
    return None

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}" 