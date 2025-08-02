import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yaml
from typing import Dict, Any, List

def show_subsystem_viewer():
    """Display subsystem analysis and configuration"""
    
    if not st.session_state.db_manager:
        st.error("❌ Database not loaded")
        return
    
    db = st.session_state.db_manager
    config_loader = st.session_state.config_loader
    
    st.markdown("## 🔧 Subsystem Analysis")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📊 Subsystem Metrics", "📈 Time Series"])
    
    with tab1:
        show_configuration(config_loader)
    
    with tab2:
        show_subsystem_metrics(db)
    
    with tab3:
        show_time_series_analysis(db)

def show_configuration(config_loader):
    """Display simulation configuration"""
    st.markdown("### ⚙️ Simulation Configuration")
    
    if not config_loader:
        st.warning("⚠️ Configuration not loaded")
        return
    
    config = config_loader.get_config()
    
    # Configuration sections
    col1, col2 = st.columns(2)
    
    with col1:
        # LLM Configuration
        st.markdown("#### 🤖 LLM Configuration")
        llm_configs = config_loader.get_llm_configs()
        if llm_configs:
            for i, llm_config in enumerate(llm_configs):
                with st.expander(f"LLM {i+1}: {llm_config.get('model', 'Unknown')}", expanded=False):
                    st.json({
                        'base_url': llm_config.get('base_url', ''),
                        'model': llm_config.get('model', ''),
                        'temperature': llm_config.get('temperature', 0.7),
                        'top_p': llm_config.get('top_p', 0.9),
                        'description': llm_config.get('description', ''),
                        'weight': llm_config.get('weight', 1.0)
                    })
        
        # Agent Configuration
        st.markdown("#### 👥 Agent Configuration")
        agent_config = config_loader.get_agent_config()
        st.json(agent_config)
    
    with col2:
        # Subsystem Configuration
        st.markdown("#### 🔧 Subsystem Configuration")
        subsystem_config = config_loader.get_subsystem_configs()
        
        st.markdown("**Active Subsystems:**")
        for subsystem in subsystem_config.get('active_subsystems', []):
            st.write(f"• {subsystem}")
        
        st.markdown("**Subsystem Directory Group:**")
        st.write(subsystem_config.get('subsystem_directory_group', 'Not specified'))
        
        # Individual subsystem configs
        subsystem_configs = subsystem_config.get('subsystem_configs', {})
        for name, config in subsystem_configs.items():
            with st.expander(f"🔧 {name} Configuration", expanded=False):
                st.json(config)
        
        # Simulation Parameters
        st.markdown("#### 🎮 Simulation Parameters")
        sim_config = config_loader.get_simulation_config()
        st.json(sim_config)
    
    # Full configuration view
    with st.expander("📄 Full Configuration (YAML)", expanded=False):
        st.code(config_loader.get_formatted_config(), language='yaml')

def show_subsystem_metrics(db):
    """Display subsystem metrics and analysis"""
    st.markdown("### 📊 Subsystem Metrics")
    
    simulation_id = st.session_state.selected_simulation
    subsystem_names = db.get_subsystem_names()
    
    if not subsystem_names:
        st.info("No subsystem data available")
        return
    
    # Filter out token usage for main display
    main_subsystems = [s for s in subsystem_names if s != "token_usage"]
    
    if not main_subsystems:
        st.info("No main subsystems found")
        return
    
    # Subsystem selector
    selected_subsystem = st.selectbox(
        "🔧 Select Subsystem:",
        main_subsystems,
        key="subsystem_selector"
    )
    
    if selected_subsystem:
        show_detailed_subsystem_metrics(db, simulation_id, selected_subsystem)
    
    # Overview of all subsystems
    st.markdown("### 🌐 All Subsystems Overview")
    
    for subsystem in main_subsystems:
        with st.expander(f"📊 {subsystem} Summary", expanded=False):
            metrics = db.get_subsystem_metrics(simulation_id, subsystem)
            if metrics:
                # Show key metrics
                if 'final_evaluation' in metrics:
                    st.markdown("**Final Evaluation Results:**")
                    st.json(metrics['final_evaluation'])
                
                if 'simulation_summary' in metrics:
                    st.markdown("**Simulation Summary:**")
                    summary = metrics['simulation_summary']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration (s)", f"{summary.get('duration', 0):.2f}")
                    with col2:
                        st.metric("Agents", summary.get('num_agents', 0))
                    with col3:
                        st.metric("Epochs", summary.get('num_epochs', 0))
            else:
                st.info("No metrics available")

def show_detailed_subsystem_metrics(db, simulation_id: str, subsystem_name: str):
    """Show detailed metrics for a specific subsystem"""
    st.markdown(f"#### 🔍 Detailed Analysis: {subsystem_name}")
    
    metrics = db.get_subsystem_metrics(simulation_id, subsystem_name)
    
    if not metrics:
        st.info("No metrics available for this subsystem")
        return
    
    # Medical Insurance System specific analysis
    if subsystem_name == "MedicalInsuranceSystem":
        show_medical_insurance_analysis(metrics)
    
    # Healthcare Utilization System specific analysis
    elif subsystem_name == "HealthcareUtilizationSystem":
        show_healthcare_utilization_analysis(metrics)
    
    # Generic subsystem analysis
    else:
        show_generic_subsystem_analysis(metrics)

def show_medical_insurance_analysis(metrics: Dict[str, Any]):
    """Show Medical Insurance System specific analysis"""
    st.markdown("#### 🏥 Medical Insurance System Analysis")
    
    # Final evaluation metrics
    if 'final_evaluation' in metrics:
        eval_data = metrics['final_evaluation']
        
        # Visit distribution analysis
        if 'visit_distribution' in eval_data:
            visit_dist = eval_data['visit_distribution']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏥 Visit Distribution by Hospital Tier**")
                if 'total_visits_by_tier' in visit_dist:
                    visits = visit_dist['total_visits_by_tier']
                    fig = px.pie(
                        values=list(visits.values()),
                        names=list(visits.keys()),
                        title="Total Visits by Hospital Tier",
                        color_discrete_sequence=['#667eea', '#f093fb', '#4facfe']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**📈 Policy Impact Metrics**")
                st.metric(
                    "Community Clinic Ratio Change", 
                    f"{visit_dist.get('community_clinic_ratio_change', 0):.3f}",
                    delta=visit_dist.get('community_clinic_ratio_change', 0)
                )
                
                policy_achieved = visit_dist.get('policy_goal_achieved', False)
                st.metric(
                    "Policy Goal Achieved", 
                    "✅ Yes" if policy_achieved else "❌ No"
                )
        
        # Financial impact analysis
        if 'financial_impact' in eval_data:
            financial = eval_data['financial_impact']
            
            st.markdown("**💰 Financial Impact Analysis**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Medical Expenses", 
                    f"${financial.get('total_medical_expenses', 0):,.2f}"
                )
            
            with col2:
                st.metric(
                    "Total Reimbursements", 
                    f"${financial.get('total_reimbursements', 0):,.2f}"
                )
            
            with col3:
                st.metric(
                    "Avg Reimbursement Rate", 
                    f"{financial.get('average_reimbursement_rate', 0):.1%}"
                )
            
            with col4:
                st.metric(
                    "Gov Burden Change", 
                    f"${financial.get('government_burden_change', 0):,.2f}",
                    delta=financial.get('government_burden_change', 0)
                )
        
        # Individual impact analysis
        if 'individual_impact' in eval_data:
            individual = eval_data['individual_impact']
            
            st.markdown("**👤 Individual Impact Analysis**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Avg Out-of-Pocket", 
                    f"${individual.get('average_out_of_pocket', 0):,.2f}"
                )
            
            with col2:
                st.metric(
                    "Median Out-of-Pocket", 
                    f"${individual.get('median_out_of_pocket', 0):,.2f}"
                )
            
            with col3:
                st.metric(
                    "Max Out-of-Pocket", 
                    f"${individual.get('max_out_of_pocket', 0):,.2f}"
                )

def show_healthcare_utilization_analysis(metrics: Dict[str, Any]):
    """Show Healthcare Utilization System specific analysis"""
    st.markdown("#### 🏥 Healthcare Utilization System Analysis")
    
    if 'final_evaluation' in metrics:
        eval_data = metrics['final_evaluation']
        
        # Utilization metrics
        if 'utilization_metrics' in eval_data:
            util_metrics = eval_data['utilization_metrics']
            
            st.markdown("**📊 Utilization Metrics**")
            
            # Average utilization rates
            if 'average_utilization_rates' in util_metrics:
                rates = util_metrics['average_utilization_rates']
                
                fig = px.bar(
                    x=list(rates.keys()),
                    y=list(rates.values()),
                    title="Average Utilization Rates by Hospital Tier",
                    labels={'x': 'Hospital Tier', 'y': 'Utilization Rate'},
                    color=list(rates.values()),
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Overcrowding incidents
            if 'overcrowding_incidents' in util_metrics:
                incidents = util_metrics['overcrowding_incidents']
                
                col1, col2, col3 = st.columns(3)
                for i, (tier, count) in enumerate(incidents.items()):
                    with [col1, col2, col3][i]:
                        st.metric(f"{tier.replace('_', ' ').title()} Overcrowding", count)
        
        # Wait time analysis
        if 'wait_time_analysis' in eval_data:
            wait_times = eval_data['wait_time_analysis']
            
            st.markdown("**⏰ Wait Time Analysis**")
            
            if 'average_wait_times' in wait_times:
                avg_waits = wait_times['average_wait_times']
                max_waits = wait_times.get('max_wait_times', {})
                
                # Create comparison chart
                tiers = list(avg_waits.keys())
                avg_values = list(avg_waits.values())
                max_values = [max_waits.get(tier, 0) for tier in tiers]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Average Wait Time',
                    x=tiers,
                    y=avg_values,
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    name='Maximum Wait Time',
                    x=tiers,
                    y=max_values,
                    marker_color='darkblue'
                ))
                
                fig.update_layout(
                    title='Wait Times by Hospital Tier (minutes)',
                    xaxis_title='Hospital Tier',
                    yaxis_title='Wait Time (minutes)',
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Service quality metrics
        if 'service_quality' in eval_data:
            quality = eval_data['service_quality']
            
            st.markdown("**⭐ Service Quality Analysis**")
            
            if 'quality_trends' in quality:
                trends = quality['quality_trends']
                
                for tier, trend_data in trends.items():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{tier.replace('_', ' ').title()} Start", f"{trend_data.get('start', 0):.2f}")
                    with col2:
                        st.metric(f"{tier.replace('_', ' ').title()} End", f"{trend_data.get('end', 0):.2f}")
                    with col3:
                        st.metric(f"{tier.replace('_', ' ').title()} Average", f"{trend_data.get('average', 0):.2f}")

def show_generic_subsystem_analysis(metrics: Dict[str, Any]):
    """Show generic subsystem analysis for unknown subsystem types"""
    st.markdown("**📊 Generic Subsystem Metrics**")
    
    # Display all available metrics
    for key, value in metrics.items():
        with st.expander(f"📋 {key}", expanded=False):
            if isinstance(value, dict):
                st.json(value)
            elif isinstance(value, list):
                st.json(value)
            else:
                st.write(value)

def show_time_series_analysis(db):
    """Show time series analysis of subsystem metrics"""
    st.markdown("### 📈 Time Series Analysis")
    
    simulation_id = st.session_state.selected_simulation
    subsystem_names = [s for s in db.get_subsystem_names() if s != "token_usage"]
    
    if not subsystem_names:
        st.info("No subsystem data available for time series analysis")
        return
    
    # Collect all time series data from all subsystems
    all_time_series_data = {}
    subsystem_colors = {
        'MedicalInsuranceSystem': '#FF6B6B',      # Red
        'HealthcareUtilizationSystem': '#4ECDC4', # Teal
        'PolicyEvaluationSystem': '#45B7D1',      # Blue
        'EconomicSystem': '#96CEB4',              # Green
        'SocialSystem': '#FFEAA7',                # Yellow
        'EnvironmentSystem': '#DDA0DD',           # Plum
        'AgentSystem': '#FFB347',                 # Orange
        'DecisionSystem': '#98D8C8'               # Mint
    }
    
    # Default colors for unknown subsystems
    default_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF', '#FFD700', '#FF6347']
    
    st.markdown("#### 📊 All Subsystems Time Series Overview")
    st.markdown("*Displaying time series data from all subsystems with different colors for each subsystem*")
    
    # Collect all metrics from all subsystems
    for subsystem in subsystem_names:
        metrics = db.get_subsystem_metrics(simulation_id, subsystem)
        
        if 'final_evaluation' in metrics and 'time_series' in metrics['final_evaluation']:
            time_series_data = metrics['final_evaluation']['time_series']
            
            # Assign color to subsystem
            if subsystem not in subsystem_colors:
                color_index = len([s for s in subsystem_colors.keys() if s in subsystem_names]) % len(default_colors)
                subsystem_colors[subsystem] = default_colors[color_index]
            
            # Process time series data for this subsystem
            for metric_name, metric_data in time_series_data.items():
                if isinstance(metric_data, list) and len(metric_data) > 0:
                    # Handle numeric arrays
                    if all(isinstance(x, (int, float)) for x in metric_data):
                        key = f"{subsystem}_{metric_name}"
                        all_time_series_data[key] = {
                            'data': metric_data,
                            'subsystem': subsystem,
                            'metric': metric_name,
                            'color': subsystem_colors[subsystem],
                            'type': 'numeric'
                        }
                    
                    # Handle dictionary arrays (multi-series)
                    elif all(isinstance(x, dict) for x in metric_data):
                        # Get all unique keys from the dictionaries
                        all_keys = set()
                        for item in metric_data:
                            all_keys.update(item.keys())
                        
                        for sub_key in all_keys:
                            y_values = [item.get(sub_key, 0) for item in metric_data]
                            key = f"{subsystem}_{metric_name}_{sub_key}"
                            all_time_series_data[key] = {
                                'data': y_values,
                                'subsystem': subsystem,
                                'metric': f"{metric_name}_{sub_key}",
                                'color': subsystem_colors[subsystem],
                                'type': 'multi_series'
                            }
    
    if not all_time_series_data:
        st.info("No time series data found in any subsystem")
        return
    
    # Display subsystem color legend
    st.markdown("#### 🎨 Subsystem Color Legend")
    legend_cols = st.columns(min(len(subsystem_colors), 4))
    for i, (subsystem, color) in enumerate(subsystem_colors.items()):
        if subsystem in subsystem_names:
            with legend_cols[i % 4]:
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 20px; height: 20px; background-color: {color}; border-radius: 3px; margin-right: 10px;"></div>
                    <span style="font-size: 14px;">{subsystem}</span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create charts in rows of 4
    chart_keys = list(all_time_series_data.keys())
    charts_per_row = 4
    
    for i in range(0, len(chart_keys), charts_per_row):
        # Create columns for this row
        cols = st.columns(charts_per_row)
        
        # Fill the columns with charts
        for j in range(charts_per_row):
            chart_index = i + j
            if chart_index < len(chart_keys):
                key = chart_keys[chart_index]
                chart_data = all_time_series_data[key]
                
                with cols[j]:
                    create_time_series_chart(
                        chart_data['data'],
                        chart_data['subsystem'],
                        chart_data['metric'],
                        chart_data['color']
                    )
    
    # Summary statistics
    st.markdown("---")
    st.markdown("#### 📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Subsystems", len(subsystem_names))
    
    with col2:
        st.metric("Total Metrics", len(all_time_series_data))
    
    with col3:
        # Calculate average data points across all metrics
        avg_data_points = sum(len(data['data']) for data in all_time_series_data.values()) / len(all_time_series_data)
        st.metric("Avg Data Points", f"{avg_data_points:.1f}")
    
    with col4:
        # Find the metric with most variation
        max_variation = 0
        most_varied_metric = "N/A"
        for key, data in all_time_series_data.items():
            if len(data['data']) > 1:
                variation = max(data['data']) - min(data['data'])
                if variation > max_variation:
                    max_variation = variation
                    most_varied_metric = key.split('_', 1)[1]  # Remove subsystem prefix
        
        st.metric("Most Varied Metric", most_varied_metric[:20] + "..." if len(most_varied_metric) > 20 else most_varied_metric)

def create_time_series_chart(data: List[float], subsystem: str, metric: str, color: str):
    """Create a single time series chart"""
    epochs = list(range(len(data)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=data,
        mode='lines+markers',
        name=f"{subsystem}",
        line=dict(color=color, width=2),
        marker=dict(color=color, size=4),
        hovertemplate=f"<b>{subsystem}</b><br>" +
                     f"Metric: {metric}<br>" +
                     "Epoch: %{x}<br>" +
                     "Value: %{y}<br>" +
                     "<extra></extra>"
    ))
    
    # Clean up metric name for title
    clean_metric = metric.replace('_', ' ').title()
    if len(clean_metric) > 25:
        clean_metric = clean_metric[:22] + "..."
    
    fig.update_layout(
        title=f"{clean_metric}",
        title_font_size=12,
        xaxis_title="Epoch",
        yaxis_title="Value",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
        hovermode='closest'
    )
    
    # Add subsystem label as annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"<b>{subsystem}</b>",
        showarrow=False,
        font=dict(size=10, color=color),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor=color,
        borderwidth=1
    )
    
    st.plotly_chart(fig, use_container_width=True) 