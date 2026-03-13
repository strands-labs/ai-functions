# Automobile Sensor Data Analysis with AI Functions

A comprehensive example demonstrating ML-based anomaly detection on synthetic automobile sensor data using the **ai_functions** framework from AWS Strands Labs.

## 🎯 What This Example Demonstrates

This project showcases how to use **AI Functions** to build sophisticated ML pipelines with natural language prompts. The AI agent autonomously:
- Generates realistic synthetic sensor data
- Implements scikit-learn models
- Detects anomalies using multiple algorithms
- Creates interactive visualizations
- Provides actionable insights

## 📁 Project Structure

### Core Files

1. **generate_data.py** - Generates synthetic sensor data with realistic patterns and anomalies
   - Uses `get_vehicle_specs` tool for vehicle-specific characteristics
   - Creates data for 5 different vehicle types (Sedan, SUV, Sports Car, EV, Pickup)
   - Injects realistic anomalies (~5% of data)

2. **analyze_anomalies.py** - Detects anomalies using Isolation Forest
   - Trains ML model on sensor data
   - Creates 3 interactive visualizations
   - Generates comprehensive analysis report

3. **analyze_anomalies_pca.py** - PCA-based anomaly detection
   - Uses reconstruction error method
   - Extracts eigenvalues and eigenvectors
   - Creates 4 visualizations including scree plot and biplot
   - Compares with Isolation Forest results

## 🚗 Sensors Included (20 total)

### Motion Sensors
- speed_kmh, rpm, throttle_position, brake_pressure, steering_angle

### Temperature Sensors
- engine_temp_celsius, coolant_temp_celsius, transmission_temp_celsius, ambient_temp_celsius

### Pressure Sensors
- tire_pressure_fl/fr/rl/rr (4 tires), oil_pressure_bar

### Power & Fuel
- battery_voltage, fuel_consumption_lph

### GPS
- latitude, longitude

### Metadata
- timestamp, vehicle_id

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have Python 3.9+ installed
python --version

# Install uv (Python package manager)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip:
pip install uv
```

### Installation

```bash
# 1. Navigate to the project directory
cd auto_sensor_data_analysis_agent

# 2. Create a virtual environment with uv
uv venv

# 3. Activate the virtual environment
# macOS/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate

# 4. Install dependencies
uv pip install -r requirements.txt

# 5. Verify installation
python -c "import ai_functions; import pandas; import sklearn; print('✓ All dependencies installed')"
```

### Usage

```bash
# Make sure your virtual environment is activated!

# 1. Generate synthetic sensor data
python generate_data.py

# 2. Analyze with Isolation Forest
python analyze_anomalies.py

# 3. Analyze with PCA (optional)
python analyze_anomalies_pca.py
```

### Alternative: Run Without Activating venv

```bash
# uv can run scripts directly without activating the venv
uv run generate_data.py
uv run analyze_anomalies.py
uv run analyze_anomalies_pca.py
```

## 📊 Output Files

All outputs are saved to `data/`:

### From generate_data.py
- `sensor_data.csv` - Generated sensor data (1000 samples)

### From analyze_anomalies.py
- `sensor_data_with_anomalies.csv` - Data with anomaly labels
- `speed_rpm_plot.html` - Interactive scatter plot
- `temperature_analysis.html` - Time series visualization
- `tire_pressure_heatmap.html` - Tire pressure monitoring
- `analysis_report.txt` - Summary report

### From analyze_anomalies_pca.py
- `sensor_data_with_pca_anomalies.csv` - Data with PCA anomaly labels
- `pca_scree_plot.html` - Eigenvalues visualization
- `pca_biplot.html` - First 2 principal components
- `pca_reconstruction_error_dist.html` - Error distribution
- `pca_3d.html` - 3D PCA space
- `pca_analysis_report.txt` - PCA analysis report

## 🤖 How AI Functions Work

### The Magic Behind the Scenes

AI Functions is a framework that lets you define ML pipelines using natural language prompts. Here's how it works:

#### 1. **Define Your Function**
```python
from ai_functions import ai_function
from ai_functions.types import AIFunctionConfig

@ai_function(config=DATA_CONFIG)
def generate_sensor_data(num_samples: int) -> pd.DataFrame:
    """
    Generate {num_samples} samples of automobile sensor data.
    Include speed, RPM, temperature, and pressure sensors.
    """
```

#### 2. **What Happens When You Call It**
```python
data = generate_sensor_data(num_samples=1000)
```

**Behind the scenes:**
1. **Prompt Building**: Your docstring becomes the prompt
   - `{num_samples}` is replaced with `1000`
   - Additional context about available tools and libraries is added

2. **Agent Creation**: A Strands Agent is created with:
   - The LLM model (e.g., Claude)
   - Python executor tool (sandboxed environment)
   - Your custom tools (e.g., `get_vehicle_specs`)
   - Authorized imports (pandas, numpy, sklearn, etc.)

3. **Code Generation**: The LLM writes Python code:
   ```python
   import pandas as pd
   import numpy as np
   
   # Generate sensor data
   data = pd.DataFrame({
       'timestamp': pd.date_range('2024-01-01', periods=1000),
       'speed_kmh': np.random.uniform(40, 120, 1000),
       # ... more sensors
   })
   
   FinalAnswer(answer=data)  # Returns to you
   ```

4. **Execution**: Code runs in a sandboxed Python environment

5. **Validation**: Post-conditions check the result:
   - Correct data types
   - Required columns present
   - No NaN values
   - Minimum row count

6. **Retry Logic**: If validation fails:
   - Error message added to conversation
   - LLM tries again with feedback
   - Up to `max_attempts` retries

7. **Return**: You get the validated DataFrame!

#### 3. **Key Features**

**Code Execution Mode**
```python
AIFunctionConfig(
    code_execution_mode="local",  # AI writes and runs Python code
    code_executor_additional_imports=["pandas.*", "numpy.*", "sklearn.*"]
)
```

**Post-Conditions (Validation)**
```python
@ai_function(post_conditions=[validate_sensor_data])
def generate_sensor_data(...) -> pd.DataFrame:
    """..."""
```

**Custom Tools**
```python
from strands import tool

@tool
def get_vehicle_specs(vehicle_id: str) -> dict:
    """Get specifications for a vehicle"""
    return {...}

@ai_function(tools=[get_vehicle_specs])
def generate_data(...):
    """Use get_vehicle_specs tool to get realistic specs..."""
```

**Async Support**
```python
# Automatically handles async execution
data = generate_sensor_data(1000)  # Sync call
# or
data = await generate_sensor_data(1000)  # Async call
```

#### 4. **The Complete Flow**

```
User calls function
       ↓
Bind arguments (num_samples=1000)
       ↓
Build prompt from docstring
       ↓
Create Strands Agent
       ↓
LLM generates Python code
       ↓
Execute in sandbox
       ↓
Run post-conditions
       ↓
[If validation fails] → Add error to conversation → Retry
       ↓
[If validation passes] → Return result
```

### Why This Is Powerful

✅ **Natural Language**: Write prompts, not code  
✅ **Type Safe**: Return types enforced via Pydantic  
✅ **Self-Correcting**: Automatic retries with validation feedback  
✅ **Tool Integration**: LLM can call custom tools  
✅ **Library Access**: Use pandas, sklearn, plotly, etc.  
✅ **Async Native**: Built-in async support  

## 🔧 Customization

### Adjust Data Generation
```python
# In generate_data.py main()
data = generate_sensor_data(
    num_samples=2000,      # More samples
    anomaly_rate=0.10      # 10% anomalies
)
```

### Modify Sensor Ranges
Edit the prompt in `generate_sensor_data()` docstring to change normal ranges or add new sensors.

### Add New Vehicle Types
Edit the `get_vehicle_specs` tool to add more vehicle types with different characteristics.

### Change ML Algorithm
Edit the prompt in `detect_anomalies()` to use different algorithms:
- One-Class SVM
- Local Outlier Factor
- Autoencoders

## 📚 Key Concepts Demonstrated

### 1. **AI Function Basics**
- Function decoration with `@ai_function`
- Natural language prompts via docstrings
- Parameter substitution with `{variable}`

### 2. **Code Execution**
- Sandboxed Python environment
- Authorized imports configuration
- Return type validation

### 3. **Post-Conditions**
- Data validation after generation
- Automatic retry on failure
- Error feedback to LLM

### 4. **Custom Tools**
- `@tool` decorator for reusable functions
- Tool integration with AI functions
- Passing simple types (strings, dicts)

### 5. **ML Pipeline**
- Data generation
- Model training (Isolation Forest, PCA)
- Anomaly detection
- Visualization creation
- Report generation

### 6. **Multiple Approaches**
- Isolation Forest (density-based)
- PCA (reconstruction error)
- Comparison of methods

## 🎓 Learning Path

1. **Start with generate_data.py**
   - Understand `@ai_function` decorator
   - See how tools work (`get_vehicle_specs`)
   - Learn about post-conditions

2. **Explore analyze_anomalies.py**
   - See ML model training
   - Understand data loading
   - Learn visualization creation

3. **Try analyze_anomalies_pca.py**
   - Advanced ML technique
   - Eigenvalue/eigenvector analysis
   - Multiple visualization types

4. **Experiment**
   - Modify prompts
   - Add new sensors
   - Try different ML algorithms
   - Create custom tools

## 🔗 Related Resources

- [AI Functions GitHub](https://github.com/strands-labs/ai-functions)
- [Strands SDK Documentation](https://github.com/strands-labs/strands)
- [AWS Strands Labs](https://github.com/strands-labs)


**Built with ❤️ using AWS Strands Labs AI Functions**
