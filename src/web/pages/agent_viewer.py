import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from typing import Dict, Any, List
import random
import openai
from openai import OpenAI

def show_agent_viewer():
    """Display agent information and demographics"""
    
    if not st.session_state.db_manager:
        st.error("❌ Database not loaded")
        return
    
    db = st.session_state.db_manager
    
    st.markdown("## 👥 Agent Information")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🏠 Agent Overview", "👤 Individual Agents", "📊 Demographics"])
    
    with tab1:
        show_agent_overview(db)
    
    with tab2:
        show_individual_agents(db)
    
    with tab3:
        show_demographics(db)

def show_agent_overview(db):
    """Show agent overview statistics"""
    st.markdown("### 📈 Agent Overview")
    
    # Get basic stats
    agent_count = db.get_agent_count()
    max_epoch = db.get_max_epoch()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("👥 Total Agents", agent_count)
    
    with col2:
        st.metric("⏰ Simulation Epochs", (max_epoch + 1) if max_epoch is not None else 0)
    
    with col3:
        # Calculate active agents (agents with decisions)
        agents = db.get_agent_list()
        active_agents = len([a for a in agents if db.get_agent_history(a['agent_id'])])
        st.metric("🎯 Active Agents", active_agents)
    
    # Agent activity heatmap
    st.markdown("### 🔥 Agent Activity Heatmap")
    
    if max_epoch is not None:
        activity_data = []
        agents = db.get_agent_list()[:20]  # Limit to first 20 agents for visualization
        
        for agent in agents:
            history = db.get_agent_history(agent['agent_id'])
            for epoch in range(max_epoch + 1):
                epoch_data = next((h for h in history if h['epoch'] == epoch), None)
                activity_score = 1 if epoch_data else 0
                activity_data.append({
                    'Agent': agent['agent_id'][:8] + "...",  # Truncate for display
                    'Epoch': epoch,
                    'Activity': activity_score
                })
        
        if activity_data:
            df_activity = pd.DataFrame(activity_data)
            fig = px.density_heatmap(
                df_activity, 
                x='Epoch', 
                y='Agent', 
                z='Activity',
                color_continuous_scale='Viridis',
                title="Agent Activity Across Epochs"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_individual_agents(db):
    """Show individual agent details"""
    st.markdown("### 👤 Individual Agent Details")
    
    agents = db.get_agent_list()
    
    if not agents:
        st.info("No agents found in the database")
        return
    
    # Agent grid display
    st.markdown("#### 🎴 Agent Cards")
    
    # Search and filter
    search_term = st.text_input("🔍 Search agents by ID or name:", key="agent_search")
    
    # Filter agents based on search
    filtered_agents = agents
    if search_term:
        filtered_agents = [
            agent for agent in agents 
            if search_term.lower() in agent['agent_id'].lower() or 
               search_term.lower() in str(agent.get('attributes', {})).lower()
        ]
    
    # Pagination settings - 3 rows x 5 columns = 15 agents per page
    agents_per_page = 15
    total_agents = len(filtered_agents)
    total_pages = (total_agents - 1) // agents_per_page + 1 if total_agents > 0 else 1
    
    # Get current page from session state
    if 'current_agent_page' not in st.session_state:
        st.session_state.current_agent_page = 1
    
    # Calculate page bounds
    page = st.session_state.current_agent_page
    start_idx = (page - 1) * agents_per_page
    end_idx = min(start_idx + agents_per_page, total_agents)
    page_agents = filtered_agents[start_idx:end_idx]
    
    # Display agents in a 3x5 grid
    cols_per_row = 5
    for i in range(0, len(page_agents), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(page_agents):
                agent = page_agents[i + j]
                with col:
                    display_agent_card(agent, db)
    
    # Pagination controls after agent cards
    if total_pages > 1:
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⬅️ Previous", disabled=(page <= 1)):
                st.session_state.current_agent_page = max(1, page - 1)
                st.rerun()
        
        with col2:
            if st.button("⏮️ First", disabled=(page <= 1)):
                st.session_state.current_agent_page = 1
                st.rerun()
        
        with col3:
            st.markdown(f"**Page {page} of {total_pages}** (Showing {start_idx + 1}-{end_idx} of {total_agents} agents)")
        
        with col4:
            if st.button("⏭️ Last", disabled=(page >= total_pages)):
                st.session_state.current_agent_page = total_pages
                st.rerun()
        
        with col5:
            if st.button("➡️ Next", disabled=(page >= total_pages)):
                st.session_state.current_agent_page = min(total_pages, page + 1)
                st.rerun()
    else:
        st.markdown(f"**Showing all {total_agents} agents**")
    
    # Check for modal displays - only details modal now
    for agent in page_agents:
        agent_id = agent['agent_id']
        if st.session_state.get(f'show_details_{agent_id}', False):
            show_agent_details_modal(agent, db)

def display_agent_card(agent: Dict[str, Any], db):
    """Display an individual agent card"""
    agent_id = agent['agent_id']
    attributes = agent.get('attributes', {})
    basic_info = attributes.get('basic_info', {})
    
    # Generate softer card colors based on agent ID
    soft_colors = [
        '#8FA7D4',  # Soft blue
        '#A8B5D1',  # Soft lavender
        '#D4A8C7',  # Soft pink
        '#C7A8A8',  # Soft rose
        '#A8C7D4',  # Soft cyan
        '#B5D4A8',  # Soft green
        '#D4C7A8',  # Soft yellow
        '#C7B5A8'   # Soft brown
    ]
    color = soft_colors[hash(agent_id) % len(soft_colors)]
    
    # Extract key info
    name = basic_info.get('name', 'Unknown')
    age = basic_info.get('age', 'Unknown')
    gender = basic_info.get('gender', 'Unknown')
    occupation = attributes.get('economic_attributes', {}).get('occupation', 'Unknown')
    
    # Create card HTML with softer styling
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
        color: #2c3e50;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    ">
        <h4 style="margin: 0; font-size: 1.1em; color: #2c3e50;">👤 {name}</h4>
        <p style="margin: 0.5rem 0; opacity: 0.8; font-size: 0.85em;">ID: {agent_id[:12]}...</p>
        <div style="display: flex; justify-content: space-between; font-size: 0.9em; margin: 0.5rem 0;">
            <span>🎂 {age}</span>
            <span>👤 {gender}</span>
        </div>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8em; opacity: 0.7;">💼 {occupation}</p>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Single button for details (which now includes chat)
    if st.button("🔍 View Details", key=f"view_{agent_id}", use_container_width=True):
        st.session_state[f'show_details_{agent_id}'] = True
        st.rerun()

@st.dialog("🔍 Agent Details", width="large")
def show_agent_details_modal(agent: Dict[str, Any], db):
    """Show detailed agent information in a wider modal with chat included"""
    agent_id = agent['agent_id']
    attributes = agent.get('attributes', {})
    basic_info = attributes.get('basic_info', {})
    name = basic_info.get('name', 'Agent')
    
    st.markdown(f"### 🔍 Agent Details: {name}")
    st.markdown(f"**Agent ID:** {agent_id}")
    
    # Tabs for different aspects (Static Info, History, and Chat)
    detail_tab1, detail_tab2, detail_tab3 = st.tabs(["📋 Static Info", "📈 History", "💬 Chat"])
    
    with detail_tab1:
        st.markdown("#### 📊 Static Attributes")
        
        # Display in organized sections using columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            if 'basic_info' in attributes:
                st.markdown("**👤 Basic Information**")
                st.json(attributes['basic_info'])
            
            if 'psychological_attributes' in attributes:
                st.markdown("**🧠 Psychological Attributes**")
                st.json(attributes['psychological_attributes'])
        
        with col2:
            if 'economic_attributes' in attributes:
                st.markdown("**💰 Economic Attributes**")
                st.json(attributes['economic_attributes'])
            
            if 'social_attributes' in attributes:
                st.markdown("**🤝 Social Attributes**")
                st.json(attributes['social_attributes'])
    
    with detail_tab2:
        st.markdown("#### 📈 Historical Dynamics")
        history = db.get_agent_history(agent_id)
        
        if not history:
            st.info("No historical data available for this agent")
        else:
            # Display history by epoch
            for epoch_data in history:
                epoch = epoch_data['epoch']
                with st.expander(f"📅 Epoch {epoch}", expanded=False):
                    
                    # Create two columns for better layout
                    left_col, right_col = st.columns(2)
                    
                    with left_col:
                        st.markdown("**🧠 Memory System**")
                        st.json(epoch_data.get('memory_system', {}))
                        
                        st.markdown("**😊 Emotion State**")
                        st.write(epoch_data.get('emotion_state', 'Unknown'))
                    
                    with right_col:
                        st.markdown("**🌍 Environment Perception**")
                        st.json(epoch_data.get('environment_perception', {}))
                        
                        st.markdown("**🎯 Decision Output**")
                        st.json(epoch_data.get('decision_output', {}))
                    
                    # Show decision input if available
                    if epoch_data.get('decision_input'):
                        st.markdown("**💭 Decision Input (Prompt)**")
                        st.text_area(
                            "Prompt:", 
                            epoch_data['decision_input'], 
                            height=100, 
                            key=f"prompt_{agent_id}_{epoch}"
                        )
    
    with detail_tab3:
        st.markdown("#### 💬 Chat with Agent")
        show_agent_chat_in_modal(agent)
    
    # Close button
    if st.button("❌ Close", key=f"close_details_{agent_id}"):
        st.session_state[f'show_details_{agent_id}'] = False
        st.rerun()

def show_agent_chat_in_modal(agent: Dict[str, Any]):
    """Show chat interface within the details modal"""
    agent_id = agent['agent_id']
    attributes = agent.get('attributes', {})
    basic_info = attributes.get('basic_info', {})
    name = basic_info.get('name', 'Agent')
    
    # LLM Configuration Section
    with st.expander("🤖 LLM Configuration", expanded=False):
        # Check if we have config from session state
        config_loader = st.session_state.get('config_loader')
        
        # Configuration source selection
        config_source = st.radio(
            "Choose LLM configuration source:",
            ["🔧 Use Simulation Config", "⚙️ Custom Configuration"],
            key=f"config_source_{agent_id}"
        )
        
        selected_config = None
        
        if config_source == "🔧 Use Simulation Config" and config_loader:
            llm_configs = config_loader.get_llm_configs()
            if llm_configs:
                # Let user select from available LLM configs
                config_names = [f"Config {i+1}: {cfg.get('model', 'Unknown')}" for i, cfg in enumerate(llm_configs)]
                selected_idx = st.selectbox(
                    "Select LLM configuration:",
                    range(len(config_names)),
                    format_func=lambda x: config_names[x],
                    key=f"llm_config_select_{agent_id}"
                )
                selected_config = llm_configs[selected_idx]
                
                st.success("✅ Using simulation LLM configuration")
                st.json({
                    'model': selected_config.get('model', 'Unknown'),
                    'base_url': selected_config.get('base_url', ''),
                    'temperature': selected_config.get('temperature', 0.7)
                })
            else:
                st.warning("⚠️ No LLM configuration found in simulation config")
                config_source = "⚙️ Custom Configuration"
        
        if config_source == "⚙️ Custom Configuration":
            st.markdown("#### 🔧 Custom LLM Configuration")
            col1, col2 = st.columns(2)
            with col1:
                base_url = st.text_input("Base URL:", value="https://api.openai.com/v1", key=f"base_url_{agent_id}")
                model = st.text_input("Model:", value="gpt-3.5-turbo", key=f"model_{agent_id}")
            with col2:
                api_key = st.text_input("API Key:", type="password", key=f"api_key_{agent_id}")
                temperature = st.slider("Temperature:", 0.0, 2.0, 0.7, key=f"temperature_{agent_id}")
            
            selected_config = {
                'base_url': base_url,
                'model': model,
                'api_key': api_key,
                'temperature': temperature
            }
    
    # Initialize chat history
    chat_key = f"modal_chat_history_{agent_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    
    # Chat history display
    st.markdown("### 💭 Conversation")
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state[chat_key]:
            st.info(f"👋 Start a conversation with {name}!")
        else:
            for message in st.session_state[chat_key]:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div style="background: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right;">
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #f3e5f5; padding: 10px; border-radius: 10px; margin: 5px 0;">
                        <strong>{name}:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("### ✍️ Your Message")
    user_input = st.text_area("Type your message:", key=f"modal_chat_input_{agent_id}", height=100)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("📤 Send Message", key=f"modal_send_{agent_id}", type="primary"):
            if user_input.strip():
                # Add user message
                st.session_state[chat_key].append({
                    'role': 'user',
                    'content': user_input.strip()
                })
                
                # Generate agent response
                if selected_config:
                    try:
                        response = generate_llm_agent_response(agent, user_input.strip(), selected_config)
                    except Exception as e:
                        response = f"Sorry, I'm having trouble connecting to the LLM service. Error: {str(e)}"
                else:
                    response = generate_agent_response(agent, user_input.strip())
                
                st.session_state[chat_key].append({
                    'role': 'agent',
                    'content': response
                })
                
                st.rerun()
            else:
                st.warning("Please enter a message")
    
    with col2:
        if st.button("🗑️ Clear Chat", key=f"modal_clear_{agent_id}"):
            st.session_state[chat_key] = []
            st.rerun()

def generate_agent_response(agent: Dict[str, Any], user_input: str) -> str:
    """Generate a mock agent response based on agent attributes"""
    attributes = agent.get('attributes', {})
    basic_info = attributes.get('basic_info', {})
    psychological = attributes.get('psychological_attributes', {})
    
    name = basic_info.get('name', 'Agent')
    personality = psychological.get('personality_traits', [])
    values = psychological.get('core_values', '')
    
    # Simple mock responses based on personality
    responses = [
        f"As someone who values {values}, I think about this carefully.",
        f"Given my {', '.join(personality[:2])} nature, I would say...",
        f"From my perspective as a {basic_info.get('occupation', 'person')}, I believe...",
        f"That's an interesting question. Let me think about it from my experience.",
        f"Based on my background and values, I would approach this differently."
    ]
    
    # Add some context-aware responses
    if "health" in user_input.lower() or "medical" in user_input.lower():
        responses.extend([
            "Health is very important to me, especially considering the insurance policies we're dealing with.",
            "I've been thinking about how the medical insurance changes affect people like me.",
            "Healthcare decisions are never easy, but we have to consider both cost and quality."
        ])
    
    return f"Hello! I'm {name}. {random.choice(responses)} What do you think about that?"

def generate_llm_agent_response(agent: Dict[str, Any], user_input: str, llm_config: Dict[str, Any]) -> str:
    """Generate agent response using LLM"""
    try:
        # Initialize OpenAI client
        base_url = llm_config.get('base_url', 'https://api.openai.com/v1')
        api_key = llm_config.get('api_key', 'sk-dummy')
        
        # For local models, use dummy key
        if 'localhost' in base_url or '127.0.0.1' in base_url or base_url.startswith('http://'):
            api_key = 'sk-dummy'
        
        client = OpenAI(base_url=base_url, api_key=api_key)
        
        # Create agent persona prompt
        attributes = agent.get('attributes', {})
        basic_info = attributes.get('basic_info', {})
        psychological = attributes.get('psychological_attributes', {})
        economic = attributes.get('economic_attributes', {})
        social = attributes.get('social_attributes', {})
        
        persona_prompt = f"""
        You are roleplaying as an agent in a medical insurance policy simulation. Here are your characteristics:
        
        Basic Information:
        - Name: {basic_info.get('name', 'Unknown')}
        - Age: {basic_info.get('age', 'Unknown')}
        - Gender: {basic_info.get('gender', 'Unknown')}
        - Education: {basic_info.get('education_level', 'Unknown')}
        - Occupation: {economic.get('occupation', 'Unknown')}
        
        Personality & Values:
        - Personality Traits: {psychological.get('personality_traits', [])}
        - Core Values: {psychological.get('core_values', 'Unknown')}
        - Risk Tolerance: {psychological.get('risk_tolerance', 'Unknown')}
        
        Economic Status:
        - Income Level: {economic.get('income_level', 'Unknown')}
        - Financial Situation: {economic.get('financial_situation', 'Unknown')}
        
        Social Context:
        - Family Status: {social.get('family_status', 'Unknown')}
        - Social Network: {social.get('social_network_size', 'Unknown')}
        
        Please respond to the user's message in character, considering your background, personality, and values. 
        Keep responses conversational and authentic to your character. Focus on how medical insurance policies 
        might affect someone with your background and characteristics.
        """
        
        response = client.chat.completions.create(
            model=llm_config.get('model', 'gpt-3.5-turbo'),
            messages=[
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=300
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"I'm having trouble connecting to the language model right now. Let me give you a simple response instead: {generate_agent_response(agent, user_input)}"

def show_demographics(db):
    """Show demographic analysis of agents"""
    st.markdown("### 📊 Agent Demographics")
    
    demographics = db.get_agent_demographics()
    
    if not demographics:
        st.info("No demographic data available")
        return
    
    # Summary statistics at the top
    st.markdown("#### 📈 Summary Statistics")
    total_agents = sum(demographics.get('age_distribution', {}).values())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Agents", total_agents)
    
    with col2:
        # Most common age group
        if demographics.get('age_distribution'):
            most_common_age = max(demographics['age_distribution'], key=demographics['age_distribution'].get)
            st.metric("🎂 Most Common Age", most_common_age)
    
    with col3:
        # Gender ratio
        if demographics.get('gender_distribution'):
            gender_dist = demographics['gender_distribution']
            if 'male' in gender_dist and 'female' in gender_dist:
                ratio = gender_dist['female'] / gender_dist['male'] if gender_dist['male'] > 0 else 0
                st.metric("⚖️ F/M Ratio", f"{ratio:.2f}")
    
    with col4:
        # Education diversity
        edu_count = len(demographics.get('education_distribution', {}))
        st.metric("🎓 Education Types", edu_count)
    
    # Create visualizations for each demographic category
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        if demographics.get('age_distribution'):
            fig_age = px.pie(
                values=list(demographics['age_distribution'].values()),
                names=list(demographics['age_distribution'].keys()),
                title="🎂 Age Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        # Education distribution - limit to top 10
        if demographics.get('education_distribution'):
            edu_data = demographics['education_distribution']
            # Sort and take top 10
            sorted_edu = dict(sorted(edu_data.items(), key=lambda x: x[1], reverse=True)[:10])
            
            fig_edu = px.bar(
                x=list(sorted_edu.keys()),
                y=list(sorted_edu.values()),
                title="🎓 Education Level Distribution (Top 10)",
                color=list(sorted_edu.values()),
                color_continuous_scale='Viridis'
            )
            fig_edu.update_layout(showlegend=False)
            fig_edu.update_xaxes(tickangle=45)
            st.plotly_chart(fig_edu, use_container_width=True)
    
    with col2:
        # Gender distribution
        if demographics.get('gender_distribution'):
            fig_gender = px.pie(
                values=list(demographics['gender_distribution'].values()),
                names=list(demographics['gender_distribution'].keys()),
                title="👥 Gender Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        
        # Income distribution - limit to top 10
        if demographics.get('income_distribution'):
            income_data = demographics['income_distribution']
            # Sort and take top 10
            sorted_income = dict(sorted(income_data.items(), key=lambda x: x[1], reverse=True)[:10])
            
            fig_income = px.bar(
                x=list(sorted_income.keys()),
                y=list(sorted_income.values()),
                title="💰 Income Level Distribution (Top 10)",
                color=list(sorted_income.values()),
                color_continuous_scale='Blues'
            )
            fig_income.update_layout(showlegend=False)
            fig_income.update_xaxes(tickangle=45)
            st.plotly_chart(fig_income, use_container_width=True)
    
    # Residence type distribution - limit to top 10
    if demographics.get('residence_distribution'):
        st.markdown("#### 🏠 Residence Type Distribution")
        residence_data = demographics['residence_distribution']
        # Sort and take top 10
        sorted_residence = dict(sorted(residence_data.items(), key=lambda x: x[1], reverse=True)[:10])
        
        fig_residence = px.bar(
            x=list(sorted_residence.keys()),
            y=list(sorted_residence.values()),
            title="Residence Type Distribution (Top 10)",
            color=list(sorted_residence.values()),
            color_continuous_scale='Greens'
        )
        fig_residence.update_layout(showlegend=False)
        fig_residence.update_xaxes(tickangle=45)
        st.plotly_chart(fig_residence, use_container_width=True) 