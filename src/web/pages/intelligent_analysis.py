import streamlit as st
import json
import yaml
from typing import Dict, Any, List
import openai
from openai import OpenAI
import time

def show_intelligent_analysis():
    """Display intelligent analysis using LLM"""
    
    if not st.session_state.db_manager:
        st.error("❌ Database not loaded")
        return
    
    db = st.session_state.db_manager
    config_loader = st.session_state.config_loader
    
    st.markdown("## 🧠 Intelligent Analysis")
    st.markdown("*AI-powered analysis of simulation results using Large Language Models*")
    
    # LLM Configuration Section
    st.markdown("### 🤖 LLM Configuration")
    
    # Get default LLM configs from simulation config
    default_configs = []
    if config_loader:
        default_configs = config_loader.get_llm_configs()
    
    # Configuration source selection
    config_source = st.radio(
        "Choose LLM configuration source:",
        ["🔧 Use Simulation Config", "⚙️ Custom Configuration"],
        key="analysis_config_source"
    )
    
    selected_config = None
    
    if config_source == "🔧 Use Simulation Config" and default_configs:
        # Let user select from available LLM configs
        config_names = [f"Config {i+1}: {cfg.get('model', 'Unknown')}" for i, cfg in enumerate(default_configs)]
        selected_idx = st.selectbox(
            "Select LLM configuration:",
            range(len(config_names)),
            format_func=lambda x: config_names[x],
            key="analysis_llm_config_select"
        )
        selected_config = default_configs[selected_idx]
        
        st.success("✅ Using simulation LLM configuration")
        with st.expander("📋 Current LLM Configuration", expanded=False):
            st.json({
                'model': selected_config.get('model', 'Unknown'),
                'base_url': selected_config.get('base_url', ''),
                'temperature': selected_config.get('temperature', 0.7),
                'description': selected_config.get('description', ''),
                'weight': selected_config.get('weight', 1.0)
            })
    
    elif config_source == "🔧 Use Simulation Config" and not default_configs:
        st.warning("⚠️ No LLM configuration found in simulation config")
        config_source = "⚙️ Custom Configuration"
    
    if config_source == "⚙️ Custom Configuration":
        st.markdown("#### 🔧 Custom LLM Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            base_url = st.text_input(
                "Base URL:", 
                value="https://api.openai.com/v1",
                help="API endpoint URL",
                key="analysis_base_url"
            )
            model = st.text_input(
                "Model:", 
                value="gpt-3.5-turbo",
                help="Model name to use",
                key="analysis_model"
            )
        
        with col2:
            api_key = st.text_input(
                "API Key:", 
                type="password",
                help="Your API key",
                key="analysis_api_key"
            )
            temperature = st.slider(
                "Temperature:", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.7,
                help="Controls randomness in responses",
                key="analysis_temperature"
            )
        
        selected_config = {
            'base_url': base_url,
            'model': model,
            'api_key': api_key,
            'temperature': temperature
        }
    
    # Analysis Options
    st.markdown("### 📊 Analysis Options")
    
    analysis_types = st.multiselect(
        "Select analysis types:",
        [
            "📈 Overall Simulation Summary",
            "🏥 Policy Impact Analysis",
            "👥 Agent Behavior Analysis", 
            "🔧 Subsystem Performance",
            "💡 Policy Recommendations",
            "📋 Comparative Analysis"
        ],
        default=["📈 Overall Simulation Summary", "🏥 Policy Impact Analysis"]
    )
    
    # Analysis depth - changed from slider to selectbox
    analysis_depth = st.selectbox(
        "Analysis Depth:",
        ["Basic", "Detailed", "Comprehensive"],
        index=1,  # Default to "Detailed"
        help="Choose the depth of analysis:\n• Basic: High-level overview\n• Detailed: In-depth analysis with metrics\n• Comprehensive: Complete analysis with recommendations"
    )
    
    # Additional options
    col1, col2 = st.columns(2)
    with col1:
        include_charts = st.checkbox("📊 Include chart descriptions", value=True)
    with col2:
        include_recommendations = st.checkbox("💡 Include actionable recommendations", value=True)
    
    # Generate Analysis Button
    if st.button("🚀 Generate Intelligent Analysis", type="primary"):
        if not analysis_types:
            st.warning("⚠️ Please select at least one analysis type")
        elif not selected_config:
            st.warning("⚠️ Please configure LLM settings")
        else:
            generate_analysis(
                db, config_loader, selected_config, analysis_types, 
                analysis_depth, include_charts, include_recommendations
            )

def generate_analysis(db, config_loader, llm_config: Dict[str, Any], analysis_types: List[str], 
                     depth: str, include_charts: bool = True, include_recommendations: bool = True):
    """Generate intelligent analysis using LLM"""
    
    simulation_id = st.session_state.selected_simulation
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Collect simulation data
        status_text.text("📊 Collecting simulation data...")
        progress_bar.progress(10)
        
        simulation_data = collect_simulation_data(db, config_loader, simulation_id)
        
        # Step 2: Initialize LLM client
        status_text.text("🤖 Initializing LLM client...")
        progress_bar.progress(20)
        
        client = initialize_llm_client(llm_config)
        if not client:
            st.error("❌ Failed to initialize LLM client")
            return
        
        # Step 3: Generate analysis for each type
        results = {}
        total_types = len(analysis_types)
        
        for i, analysis_type in enumerate(analysis_types):
            status_text.text(f"🧠 Generating {analysis_type}...")
            progress_bar.progress(30 + (i * 60 // total_types))
            
            try:
                result = generate_specific_analysis(
                    client, llm_config, simulation_data, analysis_type, depth,
                    include_charts, include_recommendations
                )
                results[analysis_type] = result
                time.sleep(1)  # Rate limiting
            except Exception as e:
                st.error(f"Error generating {analysis_type}: {str(e)}")
                results[analysis_type] = f"Error: {str(e)}"
        
        # Step 4: Display results
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        
        display_analysis_results(results)
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
    finally:
        # Clean up progress indicators
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()

def collect_simulation_data(db, config_loader, simulation_id: str) -> Dict[str, Any]:
    """Collect comprehensive simulation data for analysis"""
    
    data = {
        'simulation_id': simulation_id,
        'basic_info': {
            'agent_count': db.get_agent_count(),
            'max_epoch': db.get_max_epoch(),
            'subsystems': db.get_subsystem_names()
        },
        'configuration': {},
        'subsystem_metrics': {},
        'agent_demographics': db.get_agent_demographics(),
        'sample_agents': []
    }
    
    # Configuration data
    if config_loader:
        data['configuration'] = {
            'simulation_config': config_loader.get_simulation_config(),
            'agent_config': config_loader.get_agent_config(),
            'subsystem_configs': config_loader.get_subsystem_configs()
        }
    
    # Subsystem metrics
    for subsystem in data['basic_info']['subsystems']:
        if subsystem != "token_usage":
            data['subsystem_metrics'][subsystem] = db.get_subsystem_metrics(simulation_id, subsystem)
    
    # Sample agent data (first 5 agents for context)
    agents = db.get_agent_list()[:5]
    for agent in agents:
        agent_data = {
            'id': agent['agent_id'],
            'attributes': agent.get('attributes', {}),
            'history': db.get_agent_history(agent['agent_id'])
        }
        data['sample_agents'].append(agent_data)
    
    return data

def initialize_llm_client(config: Dict[str, Any]) -> OpenAI:
    """Initialize OpenAI client with given configuration"""
    
    try:
        # Handle different API configurations
        base_url = config.get('base_url', 'https://api.openai.com/v1')
        api_key = config.get('api_key', 'sk-dummy')  # Use dummy key for local models
        
        # For local models, we might not need a real API key
        if 'localhost' in base_url or '127.0.0.1' in base_url or base_url.startswith('http://'):
            api_key = 'sk-dummy'
        
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        return client
        
    except Exception as e:
        st.error(f"Failed to initialize LLM client: {e}")
        return None

def generate_specific_analysis(client: OpenAI, config: Dict[str, Any], data: Dict[str, Any], 
                             analysis_type: str, depth: str, include_charts: bool = True, include_recommendations: bool = True) -> str:
    """Generate specific type of analysis"""
    
    # Create analysis prompt based on type
    prompt = create_analysis_prompt(data, analysis_type, depth)
    
    try:
        response = client.chat.completions.create(
            model=config.get('model', 'gpt-3.5-turbo'),
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert data analyst specializing in social simulation and policy analysis. Provide detailed, insightful analysis based on the simulation data provided."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=config.get('temperature', 0.7),
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def create_analysis_prompt(data: Dict[str, Any], analysis_type: str, depth: str) -> str:
    """Create analysis prompt based on type and data"""
    
    base_context = f"""
    Simulation Data Summary:
    - Simulation ID: {data['simulation_id']}
    - Total Agents: {data['basic_info']['agent_count']}
    - Total Epochs: {data['basic_info']['max_epoch'] + 1 if data['basic_info']['max_epoch'] is not None else 0}
    - Active Subsystems: {', '.join(data['basic_info']['subsystems'])}
    
    Agent Demographics:
    {json.dumps(data['agent_demographics'], indent=2)}
    
    Subsystem Metrics:
    {json.dumps(data['subsystem_metrics'], indent=2)}
    """
    
    if analysis_type == "📈 Overall Simulation Summary":
        return f"""
        {base_context}
        
        Please provide a comprehensive overall summary of this medical insurance policy simulation. 
        Analysis depth: {depth}
        
        Focus on:
        1. Key simulation outcomes and trends
        2. Overall policy effectiveness
        3. Agent behavior patterns
        4. System performance metrics
        5. Notable findings and insights
        
        Format your response with clear sections and bullet points for readability.
        """
    
    elif analysis_type == "🏥 Policy Impact Analysis":
        medical_metrics = data['subsystem_metrics'].get('MedicalInsuranceSystem', {})
        return f"""
        {base_context}
        
        Medical Insurance System Specific Data:
        {json.dumps(medical_metrics, indent=2)}
        
        Please analyze the impact of the medical insurance policy changes in this simulation.
        Analysis depth: {depth}
        
        Focus on:
        1. Policy effectiveness in achieving goals
        2. Changes in healthcare utilization patterns
        3. Financial impact on individuals and government
        4. Community clinic vs hospital usage trends
        5. Recommendations for policy improvements
        
        Provide specific metrics and quantitative insights where possible.
        """
    
    elif analysis_type == "👥 Agent Behavior Analysis":
        sample_agents = data['sample_agents'][:3]  # Limit for prompt size
        return f"""
        {base_context}
        
        Sample Agent Data (first 3 agents):
        {json.dumps(sample_agents, indent=2)}
        
        Please analyze agent behavior patterns in this simulation.
        Analysis depth: {depth}
        
        Focus on:
        1. Decision-making patterns across different agent types
        2. Response to policy changes over time
        3. Demographic influences on healthcare choices
        4. Agent adaptation and learning behaviors
        5. Behavioral insights for policy design
        
        Consider both individual and collective behavior patterns.
        """
    
    elif analysis_type == "🔧 Subsystem Performance":
        return f"""
        {base_context}
        
        Please analyze the performance of the simulation subsystems.
        Analysis depth: {depth}
        
        Focus on:
        1. Each subsystem's effectiveness and efficiency
        2. Inter-subsystem interactions and dependencies
        3. Performance bottlenecks or issues
        4. Data quality and completeness
        5. Recommendations for subsystem improvements
        
        Evaluate both technical performance and domain-specific effectiveness.
        """
    
    elif analysis_type == "💡 Policy Recommendations":
        return f"""
        {base_context}
        
        Based on the simulation results, please provide policy recommendations.
        Analysis depth: {depth}
        
        Focus on:
        1. Evidence-based policy recommendations
        2. Potential unintended consequences to consider
        3. Implementation strategies and timelines
        4. Stakeholder impact analysis
        5. Metrics for monitoring policy success
        
        Provide actionable, specific recommendations with clear rationale.
        """
    
    elif analysis_type == "📋 Comparative Analysis":
        return f"""
        {base_context}
        
        Please provide a comparative analysis of different aspects of the simulation.
        Analysis depth: {depth}
        
        Focus on:
        1. Before vs after policy implementation comparisons
        2. Different agent demographic group comparisons
        3. Hospital tier utilization comparisons
        4. Cost-benefit analysis across different scenarios
        5. Performance benchmarking against expected outcomes
        
        Use quantitative comparisons where possible and highlight key differences.
        """
    
    else:
        return f"""
        {base_context}
        
        Please provide a general analysis of this simulation data.
        Analysis depth: {depth}
        
        Provide insights and observations about the simulation results.
        """

def display_analysis_results(results: Dict[str, str]):
    """Display the generated analysis results"""
    
    st.markdown("## 📋 Analysis Results")
    
    # Create tabs for each analysis type
    if len(results) == 1:
        # Single result - display directly
        analysis_type, result = list(results.items())[0]
        st.markdown(f"### {analysis_type}")
        st.markdown(result)
    else:
        # Multiple results - use tabs
        tab_names = list(results.keys())
        tabs = st.tabs(tab_names)
        
        for tab, (analysis_type, result) in zip(tabs, results.items()):
            with tab:
                st.markdown(result)
    
    # Export options
    st.markdown("### 📤 Export Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as Text"):
            export_text = generate_export_text(results)
            st.download_button(
                label="Download Text Report",
                data=export_text,
                file_name=f"analysis_report_{st.session_state.selected_simulation}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("📊 Export as JSON"):
            export_json = json.dumps(results, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON Report",
                data=export_json,
                file_name=f"analysis_report_{st.session_state.selected_simulation}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📋 Copy to Clipboard"):
            export_text = generate_export_text(results)
            st.code(export_text, language="text")
            st.info("💡 Use Ctrl+A, Ctrl+C to copy the text above")

def generate_export_text(results: Dict[str, str]) -> str:
    """Generate formatted text export of analysis results"""
    
    export_lines = [
        f"Medical Insurance Policy Simulation Analysis Report",
        f"Simulation ID: {st.session_state.selected_simulation}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        ""
    ]
    
    for analysis_type, result in results.items():
        export_lines.extend([
            f"{analysis_type}",
            "-" * len(analysis_type),
            result,
            "",
            "=" * 80,
            ""
        ])
    
    return "\n".join(export_lines) 