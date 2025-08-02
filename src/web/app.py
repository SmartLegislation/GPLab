import streamlit as st
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from pages.simulation_selector import show_simulation_selector
from pages.agent_viewer import show_agent_viewer
from pages.subsystem_viewer import show_subsystem_viewer
from pages.intelligent_analysis import show_intelligent_analysis
from utils.database_manager import DatabaseManager
from utils.config_loader import ConfigLoader

# Configure Streamlit page
st.set_page_config(
    page_title="🔬 GPLab Policy Simulation Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Hide Streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .subsystem-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 GPLab Policy Simulation Dashboard</h1>
        <p>A Generative Agent-Based Framework for Policy Simulation and Evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'selected_simulation' not in st.session_state:
        st.session_state.selected_simulation = None
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'config_loader' not in st.session_state:
        st.session_state.config_loader = None
    if 'results_directory' not in st.session_state:
        # Default to src/results instead of results
        st.session_state.results_directory = str(project_root / "src" / "results")
    
    # Sidebar navigation
    st.sidebar.markdown("## 🧭 Navigation")
    
    # Results directory selection
    st.sidebar.markdown("### 📁 Results Directory")
    
    # Directory input options
    dir_option = st.sidebar.radio(
        "Choose directory option:",
        ["📂 Use Default", "🔍 Browse Directory", "✏️ Custom Path"],
        key="dir_option"
    )
    
    if dir_option == "📂 Use Default":
        results_dir = project_root / "src" / "results"
        st.sidebar.info(f"Using default: `src/results`")
    elif dir_option == "🔍 Browse Directory":
        custom_path = st.sidebar.text_input(
            "Enter results directory path:",
            value=st.session_state.results_directory,
            help="Enter the full path to your results directory"
        )
        if custom_path:
            results_dir = Path(custom_path)
            st.session_state.results_directory = custom_path
        else:
            results_dir = project_root / "src" / "results"
    else:  # Custom Path
        custom_path = st.sidebar.text_input(
            "Enter custom path:",
            value=st.session_state.results_directory,
            help="Enter the full path to your results directory"
        )
        if custom_path:
            results_dir = Path(custom_path)
            st.session_state.results_directory = custom_path
        else:
            results_dir = project_root / "src" / "results"
    
    # Display current directory status
    if results_dir.exists():
        st.sidebar.success(f"✅ Directory found: {results_dir.name}")
        # Count simulation folders
        sim_count = len([d for d in results_dir.iterdir() if d.is_dir() and (d / "results.db").exists()])
        st.sidebar.info(f"📊 {sim_count} simulation(s) available")
    else:
        st.sidebar.error(f"❌ Directory not found: {results_dir}")
        st.sidebar.info("💡 Please check the path or run a simulation first")
        return
    
    # Simulation selection
    simulation_id = show_simulation_selector(results_dir)
    
    if simulation_id:
        # Initialize database manager and config loader
        if st.session_state.selected_simulation != simulation_id:
            st.session_state.selected_simulation = simulation_id
            db_path = results_dir / simulation_id / "results.db"
            st.session_state.db_manager = DatabaseManager(str(db_path))
            
            # Try to load config - check multiple possible config files
            config_paths = [
                project_root / "config" / "config_exp1.yaml",
                project_root / "config" / "config_medical_insurance.yaml",
                project_root / "config" / "config.yaml"
            ]
            
            config_loaded = False
            for config_path in config_paths:
                if config_path.exists():
                    st.session_state.config_loader = ConfigLoader(str(config_path))
                    config_loaded = True
                    break
            
            if not config_loaded:
                st.sidebar.warning("⚠️ No configuration file found")
        
        # Navigation menu
        page = st.sidebar.selectbox(
            "📋 Select View",
            ["🏠 Overview", "👥 Agent Information", "🔧 Subsystem Analysis", "🧠 Intelligent Analysis"],
            key="navigation"
        )
        
        # Display selected page
        if page == "🏠 Overview":
            show_overview()
        elif page == "👥 Agent Information":
            show_agent_viewer()
        elif page == "🔧 Subsystem Analysis":
            show_subsystem_viewer()
        elif page == "🧠 Intelligent Analysis":
            show_intelligent_analysis()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔬 GPLab Simulation Platform**")
    st.sidebar.markdown("*Policy Simulation and Evaluation*")

def show_overview():
    """Display simulation overview"""
    st.markdown("## 📊 Simulation Overview")
    
    if not st.session_state.db_manager:
        st.error("Database not loaded")
        return
    
    db = st.session_state.db_manager
    
    # Get basic simulation info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        agent_count = db.get_agent_count()
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Total Agents</h3>
            <h2 style="color: #667eea;">{agent_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        epoch_count = db.get_max_epoch()
        st.markdown(f"""
        <div class="metric-card">
            <h3>⏰ Total Epochs</h3>
            <h2 style="color: #f5576c;">{epoch_count + 1 if epoch_count is not None else 0}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        subsystem_count = len(db.get_subsystem_names())
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔧 Subsystems</h3>
            <h2 style="color: #f093fb;">{subsystem_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        simulation_id = st.session_state.selected_simulation
        st.markdown(f"""
        <div class="metric-card">
            <h3>🆔 Simulation ID</h3>
            <p style="color: #764ba2; font-weight: bold;">{simulation_id}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("### 📈 Quick Statistics")
    
    # Get subsystem metrics
    subsystem_names = db.get_subsystem_names()
    if subsystem_names:
        for subsystem in subsystem_names:
            if subsystem != "token_usage":  # Skip token usage for overview
                with st.expander(f"📊 {subsystem} Summary"):
                    metrics = db.get_subsystem_metrics(simulation_id, subsystem)
                    if metrics:
                        # Display key metrics
                        if 'final_evaluation' in metrics:
                            eval_data = metrics['final_evaluation']
                            st.json(eval_data)
                        else:
                            st.json(metrics)
                    else:
                        st.info("No metrics available for this subsystem")

if __name__ == "__main__":
    main() 