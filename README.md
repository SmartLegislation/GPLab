# GPLab: A Generative Agent-Based Framework for Policy Simulation and Evaluation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Agent--Based%20Modeling-orange.svg)]()

## 📖 Overview

GPLab is a **general-purpose policy simulation framework** that integrates Large Language Models (LLMs) with Agent-Based Modeling (ABM). Unlike domain-specific simulation tools, GPLab provides a **scenario-agnostic platform** where researchers can configure and deploy simulations for virtually any policy domain without modifying the core framework.

The framework addresses the limitations of traditional policy evaluation methods by:
- **Capturing behavioral heterogeneity** through LLM-powered agents with diverse demographic profiles
- **Modeling systemic transmissions** across multi-layered social systems
- **Supporting multi-scenario configurations** through a modular subsystem architecture
- **Enabling rapid policy prototyping** with minimal code changes

### Key Features

- **🤖 LLM-Powered Agents**: Social agents based on bounded rationality principles with memory and emotional modeling
- **🏗️ Scenario-Agnostic Architecture**: General-purpose subsystem framework adaptable to any policy domain
- **🔄 Dynamic Simulation**: Real-time policy transmission analysis with asynchronous execution
- **📊 Comprehensive Evaluation**: Multi-dimensional policy impact assessment with customizable metrics
- **🎯 Cross-Domain Support**: Economics, public health, housing, education, and custom domains
- **📈 Interactive Dashboard**: Streamlit-based visualization with AI-powered analysis
- **⚙️ Flexible Configuration**: YAML-based configuration for rapid scenario deployment

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

- **Behavioral Heterogeneity**: Successfully captured group differences across demographic segments
- **Emotional Evolution**: Tracked opinion dynamics and attitude changes over time
- **Policy Intensity Effects**: Demonstrated dose-response relationships between policy strength and outcomes
- **Cross-Scenario Consistency**: Validated across multiple policy domains

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
├── config/                          # Configuration files
│   ├── config_health_code_policy.yaml    # Health code policy scenario
│   ├── config_housing_policy.yaml        # Housing policy scenario
│   ├── config_tax_incentive.yaml         # Tax incentive scenario
│   └── ...                               # Other scenario configurations
├── data/                            # Agent profiles and datasets
│   ├── cgsc2018_profiles.json           # Sample CGSS data (anonymized)
│   └── llm_mock_profiles.json           # LLM-generated synthetic profiles
├── src/
│   ├── agents/                      # Agent implementation
│   │   ├── social_agent.py              # Core agent class with LLM integration
│   │   ├── memory.py                    # Memory system with embedding retrieval
│   │   └── emotion.py                   # Emotional state modeling
│   ├── simulation/                  # Simulation engine
│   │   ├── simulation_engine.py         # Main simulation orchestrator
│   │   ├── time_manager.py              # Time management and scheduling
│   │   └── state_manager.py             # State persistence and recovery
│   ├── subsystems/                  # Policy domain modules (Scenario-Agnostic)
│   │   ├── base.py                      # Abstract base class for all subsystems
│   │   ├── health_code_policy/          # Health code policy implementation
│   │   ├── housing_policy/              # Housing policy implementation
│   │   ├── tax_incentive/               # Tax incentive implementation
│   │   └── experimental/                # Custom experimental scenarios
│   ├── utils/                       # Utilities and helpers
│   │   ├── llm_client.py                # LLM API client with QPS control
│   │   ├── embedding.py                 # Embedding model utilities
│   │   └── blackboard.py                # Cross-subsystem communication
│   └── web/                         # Dashboard application
│       ├── app.py                       # Main Streamlit application
│       ├── pages/                       # Dashboard pages
│       └── requirements.txt             # Web-specific dependencies
├── results/                         # Simulation outputs
│   └── [scenario_name]/                 # Per-scenario results
└── requirements.txt                 # Python dependencies
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

## 📊 Data

### Data Privacy Notice

This project uses demographic data from the **Chinese General Social Survey (CGSS)**. Due to privacy concerns, we only provide **anonymized sample data** in `data/cgsc2018_profiles.json` with sensitive information redacted.

### Available Datasets

| Dataset | Description | Usage |
|---------|-------------|-------|
| `data/cgsc2018_profiles.json` | Anonymized CGSS 2018 samples (4 profiles) | Format reference only |
| `data/llm_mock_profiles.json` | LLM-generated synthetic profiles | **Recommended for simulation** |

### Obtaining Full CGSS Data

For research requiring authentic survey data, please apply through the official CGSS data portal:

🔗 **http://cgss.ruc.edu.cn/**

The LLM-generated profiles in `data/llm_mock_profiles.json` are created following real demographic distributions and can be used directly for simulation experiments.

## 📚 Citation

If you use GPLab in your research, please cite:

```bibtex
@article{zhang2026,
   title = {GPLab: A Generative Agent-Based Framework for Policy Simulation and Evaluation},
   author = {Zhang, Shuhan and Peng, Zifan and Ren, Yinwang},
   journal = {Journal of Artificial Societies and Social Simulation},
   ISSN = {1460-7425},
   volume = {29},
   number = {1},
   pages = {6},
   year = {2026},
   URL = {http://jasss.soc.surrey.ac.uk/29/1/6.html},
   DOI = {10.18564/jasss.5933},
   keywords = {Large Language Models, Generative Agent-Based Modeling, Policy Simulation, Complex Social Systems, Computational Social Science}
}
```

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for LLM integration
- Visualization powered by [Streamlit](https://streamlit.io/)
- Agent profiles generated using advanced LLM techniques
- Inspired by advances in computational social science

---

**GPLab** - Bridging AI and Policy Science for Better Decision Making