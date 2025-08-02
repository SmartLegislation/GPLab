# 🏥 Medical Insurance Policy Simulation Dashboard

A comprehensive Streamlit-based visualization system for analyzing medical insurance policy simulation results from the GPLab platform.

## 🌟 Features

### 📊 Historical Simulation Selection
- Automatic detection of simulation results from the `results` directory
- Display of available simulations with metadata (date, agents, epochs, subsystems)
- Default selection of the most recent simulation
- Detailed simulation information in expandable sidebar

### 👥 Agent Information Viewer
- **Agent Overview**: Statistics and activity heatmaps
- **Individual Agents**: 
  - Colorful dynamic agent cards with basic information
  - Detailed static attributes (basic info, psychological, economic, social)
  - Historical dynamics across all epochs (memory, emotions, decisions, prompts)
  - Interactive chat interface for role-playing with agents
- **Demographics**: Visual analysis of agent population distribution

### 🔧 Subsystem Analysis
- **Configuration Display**: Complete YAML configuration with organized sections
- **Subsystem Metrics**: Detailed analysis of each subsystem's performance
- **Time Series Analysis**: Interactive charts showing trends over time
- Specialized analysis for Medical Insurance and Healthcare Utilization systems

### 🧠 Intelligent Analysis
- AI-powered analysis using Large Language Models
- Multiple analysis types:
  - Overall Simulation Summary
  - Medical Insurance Policy Impact
  - Agent Behavior Analysis
  - Subsystem Performance
  - Policy Recommendations
  - Comparative Analysis
- Configurable LLM settings (use simulation config or custom)
- Export capabilities (Text, JSON, Copy to clipboard)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Completed simulation results in the `results` directory
- Required Python packages (see `requirements.txt`)

### Installation
1. Install dependencies:
```bash
pip install -r src/web/requirements.txt
```

2. Run the Streamlit application:
```bash
cd src/web
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

### Directory Structure
```
src/web/
├── app.py                 # Main Streamlit application
├── pages/                 # Individual page modules
│   ├── simulation_selector.py
│   ├── agent_viewer.py
│   ├── subsystem_viewer.py
│   └── intelligent_analysis.py
├── utils/                 # Utility modules
│   ├── database_manager.py
│   └── config_loader.py
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 📋 Usage Guide

### 1. Simulation Selection
- The sidebar automatically detects available simulations
- Select from the dropdown to switch between different simulation runs
- View simulation details in the expandable "Simulation Details" section

### 2. Agent Information
- **Overview Tab**: View agent statistics and activity patterns
- **Individual Agents Tab**: 
  - Browse agent cards in a grid layout
  - Use the search box to find specific agents
  - Click "View Details" to see comprehensive agent information
  - Use the chat feature to interact with agents based on their profiles
- **Demographics Tab**: Analyze population distributions with interactive charts

### 3. Subsystem Analysis
- **Configuration Tab**: Review simulation parameters and subsystem settings
- **Subsystem Metrics Tab**: 
  - Select a subsystem for detailed analysis
  - View specialized metrics for Medical Insurance and Healthcare Utilization
  - Explore generic metrics for other subsystems
- **Time Series Tab**: Analyze trends and patterns over simulation epochs

### 4. Intelligent Analysis
- Configure LLM settings (use simulation config or provide custom settings)
- Select analysis types based on your interests
- Choose analysis depth (Basic, Detailed, Comprehensive)
- Generate AI-powered insights and recommendations
- Export results in multiple formats

## 🎨 Design Features

### Visual Styling
- Modern gradient backgrounds and color schemes
- Responsive card-based layouts
- Interactive charts using Plotly
- Emoji icons for better visual appeal
- Consistent color coding across components

### User Experience
- Intuitive navigation with sidebar and tabs
- Progress indicators for long-running operations
- Error handling with helpful messages
- Export and download capabilities
- Responsive design for different screen sizes

## 🔧 Technical Details

### Database Integration
- SQLite database reading for simulation results
- Efficient data retrieval and caching
- Support for multiple simulation formats

### Configuration Management
- YAML configuration file parsing
- Dynamic configuration display
- Support for different subsystem types

### LLM Integration
- OpenAI-compatible API support
- Local model compatibility
- Configurable parameters and prompts
- Rate limiting and error handling

## 🐛 Troubleshooting

### Common Issues

1. **No simulations found**
   - Ensure the `results` directory exists and contains simulation folders
   - Check that simulation folders contain `results.db` files

2. **Database connection errors**
   - Verify database file integrity
   - Check file permissions

3. **LLM analysis failures**
   - Verify API configuration and connectivity
   - Check API key validity for external services
   - Ensure model names are correct

4. **Performance issues**
   - Large simulations may take time to load
   - Consider limiting agent display for better performance
   - Use browser developer tools to monitor memory usage

### Debug Mode
Enable Streamlit debug mode for detailed error information:
```bash
streamlit run app.py --logger.level=debug
```

## 📈 Future Enhancements

- Real-time simulation monitoring
- Comparison between multiple simulations
- Advanced filtering and search capabilities
- Custom visualization builder
- Integration with external analytics tools
- Mobile-responsive design improvements

## 🤝 Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings to new functions
3. Test with different simulation configurations
4. Update this README for new features

## 📄 License

This visualization system is part of the GPLab platform. Please refer to the main project license for usage terms. 