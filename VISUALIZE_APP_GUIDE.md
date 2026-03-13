# 📡 MEC Environment Visualization App - Setup & User Guide

## Overview

This interactive web-based visualization app allows you to:
- **Visualize Network Topology**: See mobile devices, edge servers, and cloud center
- **Monitor Task Processing**: Watch tasks being generated, offloaded, and processed
- **Track Real-time Metrics**: Monitor accuracy, latency, reward, and cache hit rate
- **Interactive Controls**: Execute steps manually or run simulations automatically
- **Analyze Historical Data**: View trends and statistics over time

## Prerequisites

Make sure you have Python 3.8+ installed and are in the project directory:
```bash
cd c:\Users\takhu\OneDrive\Documents\ANDA_Research\Coding\ICTC_MEC_Project
```

## Installation

### Step 1: Install Required Dependencies

```bash
# Install visualization app dependencies
pip install -r requirements_visualize.txt
```

Or individually:
```bash
pip install streamlit==1.28.1 plotly==5.17.0 streamlit-plotly-events==0.0.6
```

### Step 2: Verify Your Environment Setup

Ensure your `env_args.json` exists in the project root with proper configuration:
```bash
# Check if the file exists
dir env_args.json
```

If it doesn't exist, create one or use the app's configuration interface.

## Running the Application

### Method 1: Basic Launch (Recommended)

```bash
streamlit run visualize_app.py
```

The app will open in your default browser at `http://localhost:8501`

### Method 2: With Custom Host/Port

```bash
streamlit run visualize_app.py --server.port 8502
```

### Method 3: Headless Mode (Server)

```bash
streamlit run visualize_app.py --server.headless true --logger.level=info
```

## Using the Application

### 1. Configuration Panel (Left Sidebar)

When you first open the app, configure your environment:

**Key Parameters:**
- **Number of Edge Servers**: 1-10 (default: 3)
- **Number of Mobile Devices**: 1-50 (default: 10)
- **Number of Models**: 1-20 (default: 10)
- **Task Arrival Rate**: 0.1-2.0 (default: 0.7)
  - Controls how many tasks arrive per timestep
- **Zipf Distribution Parameter**: 0.5-2.0 (default: 1.2)
  - Controls model popularity distribution (higher = more skewed)
- **Accuracy Weight**: 0.0-1.0 (default: 0.5)
- **Latency Weight**: 0.0-1.0 (default: 1.0)
- **Random Seed**: For reproducibility

Click **"🔄 Initialize Environment"** to set up the environment with your parameters.

### 2. Environment State Tab (First Tab)

Shows the current status of your MEC system:

**Top Metrics:**
- Current timestep counter
- Number of devices and edges
- Number of available models

**Network Architecture:**
- Visual representation of the network topology
- Cloud center (orange) at the top
- Edge servers (light blue) arranged in a circle
- Mobile devices (green dots) connected to nearest edge
- Gray lines show cloud connectivity

**Edge Server Status:**
- Expandable cards for each edge server
- Shows computing power, bandwidth, cached models
- Displays cache hits and hit rate

**Observation Details:**
- Transmission delays for each device to each edge and cloud
- Number of blocks per device

**Cache Information:**
- Heatmap showing which models are cached at which edges
- Red intensity = cache hit likelihood

### 3. Metrics Tab (Second Tab)

Real-time monitoring and control:

**Control Buttons:**
- **▶️ Execute Single Step**: Run one timestep of the simulation
- **🔄 Reset Environment**: Clear all metrics and restart
- **⚡ Run Multiple Steps**: Run N steps automatically (configure count in input field)

**Real-time Metrics Display:**
Shows the latest values for:
- Reward
- Average Accuracy
- Average Latency (ms)
- Cache Hit Rate

**Metric Graphs:**
Four real-time plots showing trends:
1. **Cumulative Reward**: Total accumulated reward over time
2. **Average Accuracy**: Model accuracy over timesteps
3. **Average Latency**: Task processing latency in milliseconds
4. **Cache Hit Rate**: Percentage of cache hits over time

### 4. History Tab (Third Tab)

Comprehensive performance analysis:

**Data Table:**
- View all recorded metrics in tabular format
- Sortable and scrollable columns

**Download Option:**
- Export metrics as CSV for external analysis

**Summary Statistics:**
- Min, Max, and Mean values for:
  - Reward
  - Accuracy
  - Latency
  - Cache Hit Rate

## Understanding the Metrics

### Reward
- **Higher is better**: Composite metric combining accuracy and latency
- Formula: `reward = accuracy_weight * accuracy - latency_weight * latency`

### Accuracy
- **Range**: 0.0 to 1.0 (0-100%)
- **Higher is better**: Percentage of correct model outputs
- Determined by which model blocks are executed

### Latency
- **Measured in milliseconds**: Time to process and return results
- **Lower is better**: Includes transmission and computation time
- Affected by:
  - Task offloading decision (local/edge/cloud)
  - Edge server workload
  - Cache hits (reduces latency)

### Cache Hit Rate
- **Range**: 0.0 to 1.0 (0-100%)
- **Higher is better**: Percentage of requests served from cache
- Increases with effective caching policies

## Workflow Example

### Scenario: Testing a New Configuration

1. **Initialize Environment**
   - Set: 5 Edge Servers, 20 Mobile Devices, 15 Models
   - Set: Task Arrival Rate = 0.8, Zipf = 1.5
   - Click "Initialize Environment"

2. **Explore Initial State**
   - Check "Environment State" tab
   - View network topology and edge capacities

3. **Run Simulation**
   - Go to "Metrics" tab
   - Click "Run Multiple Steps" with 50 steps
   - Watch the graphs update in real-time

4. **Analyze Results**
   - Switch to "History" tab
   - Review summary statistics
   - Download CSV for further analysis

## Troubleshooting

### Issue: "Module not found" Error

**Solution**: Ensure you're in the correct directory and dependencies are installed:
```bash
cd c:\Users\takhu\OneDrive\Documents\ANDA_Research\Coding\ICTC_MEC_Project
pip install -r requirements_visualize.txt
```

### Issue: Port 8501 Already in Use

**Solution**: Use a different port:
```bash
streamlit run visualize_app.py --server.port 8502
```

### Issue: Slow Performance with Many Devices/Models

**Solution**: 
- Reduce number of devices in configuration
- Run fewer automatic steps at a time
- Close other applications using CPU

### Issue: env_args.json Not Found

**Solution**:
- The app can work without it (uses default values)
- Or copy your existing env_args.json to the project root:
```bash
copy env_args.json .
```

### Issue: Tasks Show Zero Accuracy/Latency

**Solution**:
- Run multiple steps - metrics accumulate over time
- Ensure task arrival rate > 0
- Check if tasks are actually being generated

## Advanced Tips

### 1. Batch Processing
```bash
# Run 10 simulations and save results
for /L %i in (1,1,10) do (
    streamlit run visualize_app.py --server.headless true
    timeout /t 300  # Wait 5 minutes
)
```

### 2. Custom Metrics Export
- Download CSV from History tab
- Use Python/Excel to analyze:
```python
import pandas as pd

df = pd.read_csv('mec_metrics_history.csv')
print(df.describe())
```

### 3. Performance Optimization
- For large networks (>30 edges), consider sampling devices in visualization
- The app caches network visualization - modify `@st.cache_resource` to clear cache

## Key Components Explained

### Network Visualization
- **Cloud (Orange)**: Central computing resource, slowest access but unlimited resources
- **Edges (Blue)**: Regional servers, faster than cloud, limited cache storage
- **Devices (Green)**: Mobile/IoT devices, fastest local compute but weakest

### Cache Mechanism
- Each edge stores frequently-accessed models
- When a task requests a cached model, it's processed locally (faster)
- Uncached models may need to be fetched from cloud (slower)

### Task Flow
1. Task generated at mobile device
2. Device chooses: local processing, edge offload, or cloud
3. If offloaded:
   - Check if model is cached at target edge
   - If cached: process immediately
   - If not cached: fetch from cloud first (additional latency)
4. Return result to device

## Performance Metrics Interpretation

| Metric | Good Value | Poor Value | What It Means |
|--------|-----------|-----------|--------------|
| Reward | > -10 | < -100 | Overall system performance |
| Accuracy | > 0.8 (80%) | < 0.3 (30%) | Model prediction correctness |
| Latency | < 50 ms | > 500 ms | Speed of response |
| Cache Hit Rate | > 0.5 (50%) | < 0.1 (10%) | Cache effectiveness |

## Next Steps

After exploring with the visualization:

1. **Train Agents**: Run your RL training to optimize offloading/caching decisions
2. **Compare Strategies**: Run different configurations and compare results
3. **Export Data**: Use the CSV export for publication-ready analysis
4. **Modify Actions**: Edit the `create_random_actions()` function to implement custom policies

## Support & Customization

### To Add Custom Metrics
Edit the `create_metrics_plot()` function and add new keys to `metrics_history`.

### To Change Visualizations
Modify the Plotly figures in `create_network_visualization()` and `create_cache_visualization()`.

### To Integrate with Agents
Replace `create_random_actions()` with calls to your RL agents:
```python
def create_random_actions(env):
    """Replace with your agent"""
    offload_actions = your_trained_agent.predict(obs)
    exit_actions = your_exit_agent.predict(obs)
    return np.concatenate([offload_actions, exit_actions])
```

---

**Happy Visualizing! 🚀**
