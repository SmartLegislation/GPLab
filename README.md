# GPLab: A Generative Agent-Based Framework for Policy Simulation and Evaluation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-Agent--Based%20Modeling-orange.svg)]()

## 📖 Overview

GPLab is a comprehensive framework that integrates Large Language Models (LLMs) with Agent-Based Modeling (ABM) to simulate and evaluate complex policy scenarios. The framework addresses the limitations of traditional policy evaluation methods by capturing behavioral heterogeneity and systemic transmissions across multi-layered social systems.

### Key Features

- **🤖 LLM-Powered Agents**: Social agents based on bounded rationality principles
- **🏗️ Modular Architecture**: Specialized subsystem modeling for different policy domains
- **🔄 Dynamic Simulation**: Real-time policy transmission analysis
- **📊 Comprehensive Evaluation**: Multi-dimensional policy impact assessment
- **🎯 Cross-Domain Support**: Economics, public health, housing, education, and more
- **📈 Interactive Dashboard**: Streamlit-based visualization system

## 🏛️ Architecture

### Core Components

1. **Social Agents** (`src/agents/`)
   - LLM-driven decision making
   - Memory system with embedding-based retrieval
   - Emotional state modeling
   - Bounded rationality implementation

2. **Subsystem Framework** (`src/subsystems/`)
   - Modular policy domain modeling
   - Cross-subsystem communication via blackboard pattern
   - Specialized evaluation metrics

3. **Simulation Engine** (`src/simulation/`)
   - Asynchronous execution with QPS control
   - Time management and scheduling
   - State persistence and recovery

4. **Visualization Dashboard** (`src/web/`)
   - Real-time simulation monitoring
   - Agent behavior analysis
   - Policy impact visualization
   - AI-powered analysis reports

### Supported Policy Domains

- **🏥 Health Code Policy**: Disease transmission and public health measures
- **🏠 Housing Policy**: Real estate market dynamics and affordability
- **💰 Tax Incentive**: Economic behavior and fiscal policy impact
- **🎓 Education Balance**: Educational resource allocation
- **🩺 Medical Insurance**: Healthcare utilization and coverage
- **🧪 Experimental Scenarios**: Custom policy testing environments

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for local embedding models)
- Access to LLM APIs (OpenAI-compatible endpoints)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/GPLab.git
   cd GPLab
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install web dashboard dependencies**
   ```bash
   pip install -r src/web/requirements.txt
   ```

### Configuration

1. **Configure LLM APIs**
   
   Edit the configuration files in `config/` directory to set up your LLM endpoints:
   ```yaml
   llm_api_configs:
     - base_url: "https://your-llm-endpoint/v1/"
       api_key: "your-api-key"
       model: "your-model-name"
       temperature: 0.7
   ```

2. **Set up embedding models**
   
   Configure local or remote embedding models:
   ```yaml
   use_local_embedding_model: true
   local_embedding_model_path: "path/to/your/embedding/model"
   ```

### Running Simulations

1. **Basic simulation**
   ```bash
   cd src
   python main.py --config ../config/config_health_code_policy.yaml
   ```

2. **Custom configuration**
   ```bash
   python main.py --config ../config/your_custom_config.yaml
   ```

3. **Launch visualization dashboard**
   ```bash
   cd src/web
   streamlit run app.py
   ```

## 📊 Configuration Guide

### Agent Configuration

```yaml
# Agent setup
agent_data_path: "../data/llm_mock_profiles.json"
agent_sample_size: 200
agent_sample_seed: 123
agent_sampling_method: "random"
```

### Simulation Parameters

```yaml
# Simulation control
simulation:
  total_epochs: 30
  agents_per_epoch: 50
  max_concurrent_requests: 50
  
# Time configuration
time_config:
  start_date: "2024-01-01"
  epoch_duration_days: 1
```

### Subsystem Configuration

Each policy domain can be configured independently:

```yaml
subsystems:
  citizen_health:
    module_path: "src.subsystems.health_code_policy.citizen_health"
    class_name: "CitizenHealthSystem"
    config:
      health_params:
        initial_infection_rate: 0.001
        recovery_days_range: [7, 14]
```

## 🔬 Research Applications

### Policy Scenarios Tested

1. **COVID-19 Health Code Policy**
   - Disease transmission modeling
   - Public compliance analysis
   - Economic impact assessment

2. **Housing Affordability Policy**
   - Market dynamics simulation
   - Demographic impact analysis
   - Policy effectiveness evaluation

3. **Tax Incentive Programs**
   - Behavioral response modeling
   - Economic outcome prediction
   - Cross-group impact analysis

### Validation Results

- **Average Rationality Score**: 7.9/10 across five policy scenarios
- **Behavioral Heterogeneity**: Successfully captured group differences
- **Emotional Evolution**: Tracked opinion dynamics over time
- **Policy Intensity Effects**: Demonstrated dose-response relationships

## 📈 Dashboard Features

### Agent Analysis
- Individual agent profiles and decision history
- Population demographics visualization
- Behavioral pattern analysis
- Interactive agent chat interface

### Subsystem Monitoring
- Real-time system metrics
- Configuration display
- Performance analytics
- Time series trend analysis

### AI-Powered Insights
- Automated simulation summaries
- Policy impact analysis
- Behavioral pattern recognition
- Recommendation generation

## 🛠️ Development

### Project Structure

```
GPLab/
├── config/                 # Configuration files
├── data/                   # Agent profiles and datasets
├── src/
│   ├── agents/            # Agent implementation
│   ├── simulation/        # Simulation engine
│   ├── subsystems/        # Policy domain modules
│   ├── utils/             # Utilities and helpers
│   └── web/               # Dashboard application
├── results/               # Simulation outputs
└── requirements.txt       # Dependencies
```

### Adding New Policy Domains

1. **Create subsystem module**
   ```python
   from src.subsystems.base import SocialSystemBase
   
   class YourPolicySystem(SocialSystemBase):
       def init(self, all_agent_data):
           # Initialize your policy system
           pass
           
       def step(self, agent_decisions):
           # Process agent decisions
           pass
           
       def evaluate(self):
           # Evaluate policy outcomes
           pass
   ```

2. **Configure in YAML**
   ```yaml
   subsystems:
     your_policy:
       module_path: "src.subsystems.your_domain.your_policy"
       class_name: "YourPolicySystem"
       config:
         # Your policy parameters
   ```

### Testing

```bash
# Run basic functionality tests
python -m pytest tests/

# Test specific subsystem
python -m pytest tests/test_subsystems.py
```

## 📚 Citation

If you use GPLab in your research, please cite:


## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Wiki](https://github.com/yourusername/GPLab/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/GPLab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/GPLab/discussions)

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for LLM integration
- Visualization powered by [Streamlit](https://streamlit.io/)
- Agent profiles generated using advanced LLM techniques
- Inspired by advances in computational social science

---

**GPLab** - Bridging AI and Policy Science for Better Decision Making