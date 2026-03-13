import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events
import json
import glob
import os
import traceback
from MEC_Environment.gym_environment import Environment
from MEC_Environment.dual_timescale_wrapper import MECDualTimeScaleEnv, SlowEnvWrapper, FastEnvWrapper
from Optimizer.Random_Dual_Optimizer import RandomDualOptimizer
from experiment_utils import load_env_args
from stable_baselines3 import A2C
import time

# ==================== HELPER FUNCTIONS ====================

def create_network_visualization(_env, seed=42):
    """Create a visualization of the network topology.
    
    Note: Uses deterministic seed for device positioning to keep visualization stable.
    """
    fig = go.Figure()
    
    # Add cloud center
    fig.add_trace(go.Scatter(
        x=[0], y=[2],
        mode='markers+text',
        marker=dict(size=30, color='orange'),
        text=['☁️ Cloud'],
        textposition='top center',
        name='Cloud'
    ))
    
    # Add edge servers in a circle
    angles = np.linspace(0, 2*np.pi, _env.num_edges, endpoint=False)
    edge_x = np.cos(angles)
    edge_y = np.sin(angles)
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='markers+text',
        marker=dict(size=20, color='lightblue', line=dict(color='blue', width=2)),
        text=[f'Edge {i}' for i in range(_env.num_edges)],
        textposition='bottom center',
        name='Edge Servers'
    ))
    
    # Add mobile devices (sampled) with deterministic positioning
    if _env.num_devices <= 20:
        device_sample_size = _env.num_devices
    else:
        device_sample_size = 20
    
    # Use deterministic RNG for stable visualization
    rng = np.random.default_rng(seed)
    device_indices = rng.choice(_env.num_devices, min(device_sample_size, _env.num_devices), replace=False)
    
    for dev_idx in device_indices:
        dev = _env.mobile_devices[dev_idx]
        edge_id = dev.edge_id
        
        # Place device near its associated edge with deterministic positioning
        angle = angles[edge_id]
        x = 1.5 * np.cos(angle) + rng.standard_normal() * 0.2
        y = 1.5 * np.sin(angle) + rng.standard_normal() * 0.2
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=8, color='green'),
            name='Mobile Devices' if dev_idx == device_indices[0] else '',
            showlegend=bool(dev_idx == device_indices[0]),
            hovertext=f'Device {dev_idx}',
            hoverinfo='text'
        ))
    
    # Connect edges to cloud
    for i in range(_env.num_edges):
        fig.add_trace(go.Scatter(
            x=[edge_x[i], 0], y=[edge_y[i], 2],
            mode='lines',
            line=dict(color='gray', width=1),
            name='Cloud Link' if i == 0 else '',
            showlegend=bool(i == 0),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title="Network Topology",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    return fig


def create_cache_visualization(_env):
    """Create a visualization of cache states.
    
    Note: Not cached because cache state changes between steps.
    """
    cache_data = []
    
    for edge_id, edge in enumerate(_env.edge_servers):
        for model_id in range(_env.num_models):
            is_cached = 1 if model_id in edge.cached_models else 0
            cache_data.append({
                'Edge': f'Edge {edge_id}',
                'Model': f'Model {model_id}',
                'Cached': is_cached
            })
    
    cache_df = pd.DataFrame(cache_data)
    
    fig = px.density_heatmap(
        cache_df,
        x='Model',
        y='Edge',
        nbinsx=_env.num_models,
        nbinsy=_env.num_edges,
        title='Cache Status Across Edge Servers',
        color_continuous_scale='YlOrRd',
        height=300
    )
    
    return fig


def create_metrics_plot(metrics_history, metric_key, title, xaxis_title, yaxis_title):
    """Create a line plot for a specific metric."""
    if not metrics_history['timestep']:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics_history['timestep'],
        y=metrics_history[metric_key],
        mode='lines+markers',
        name=metric_key,
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_dual_actions(slow_env, fast_env, slow_agent, fast_agent, slow_obs, fast_obs, fast_step_in_slow, slow_agent_type="RL"):
    """Generate actions using pretrained or random agents.
    
    IMPORTANT: Use observations from wrapped environments when models are pretrained!
    
    Caching decisions are only made at the START of a slow episode (fast_step_in_slow == 0).
    For the remaining fast steps, we only make offloading/exit decisions.
    
    Args:
        slow_env: SlowEnvWrapper instance
        fast_env: FastEnvWrapper instance
        slow_agent: Pretrained slow agent or RandomDualOptimizer
        fast_agent: Pretrained fast agent or None
        slow_obs: Current slow environment observation
        fast_obs: Current fast environment observation
        fast_step_in_slow: Current fast step within the slow episode (0 to large_timescale_size-1)
        slow_agent_type: 'RL' for A2C agents, 'Random' for random actions
    
    Returns:
        tuple: (slow_obs, fast_obs, fast_action, made_slow_decision)
            - slow_obs: Updated slow observation
            - fast_obs: Updated fast observation  
            - fast_action: Offloading + exit decisions for this fast step
            - made_slow_decision: Whether a slow (caching) decision was made in this call
    """
    made_slow_decision = False
    
    try:
        # Only make caching decision at the start of slow episode
        if fast_step_in_slow == 0:
            # slow_agent = RandomDualOptimizer(slow_env.dual_env, seed=int(time.time()))
            # slow_action = slow_agent.predict_slow(slow_obs)
            if slow_agent_type == "RL":
                # Use pretrained A2C model for slow decisions
                slow_action, _ = slow_agent.predict(slow_obs, deterministic=True)
            else:
                # Use random optimizer
                slow_action = slow_agent.predict_slow(slow_obs)
            
            # Step the slow environment
            slow_obs, _, _, _, _ = slow_env.step(slow_action)
            made_slow_decision = True
        
        # Always make fast (offloading/exit) decision
        if fast_agent is not None:
            # Use pretrained A2C model for fast decisions
            fast_action, _ = fast_agent.predict(fast_obs, deterministic=True)
        else:
            # Use random actions if no agent specified
            fast_action = fast_env.action_space.sample()
        fast_obs, reward, terminated, truncated, info = st.session_state.fast_env.step(fast_action)
        if terminated or truncated:
            fast_obs, _ = fast_env.reset()
        
        return slow_obs, fast_obs, info, fast_action, made_slow_decision
    
    except Exception as e:
        # Return tuple with error info
        error_msg = str(e)
        
        # Debug info for observation shape mismatches
        debug_info = f"\n**DEBUG INFO:**\n"
        debug_info += f"- fast_obs type: {type(fast_obs)}\n"
        
        if isinstance(fast_obs, dict):
            debug_info += f"- fast_obs is Dict with keys: {list(fast_obs.keys())}\n"
            for key in list(fast_obs.keys())[:3]:
                debug_info += f"  - {key}: shape {fast_obs[key].shape}\n"
        elif isinstance(fast_obs, np.ndarray):
            debug_info += f"- fast_obs is numpy array with shape: {fast_obs.shape}\n"
        
        if fast_agent is not None:
            debug_info += f"- Model expects observation type: {type(fast_agent.observation_space).__name__}\n"
            if hasattr(fast_agent.observation_space, 'spaces'):
                debug_info += f"  Dict with keys: {list(fast_agent.observation_space.spaces.keys())[:3]}...\n"
            elif hasattr(fast_agent.observation_space, 'shape'):
                debug_info += f"  Box with shape: {fast_agent.observation_space.shape}\n"
        
        if "observation shape" in error_msg.lower():
            raise RuntimeError(
                f"❌ **Observation Shape Mismatch Error:**\n\n{error_msg}\n{debug_info}\n\n"
                f"**What this means:**\n"
                f"The model was trained on ONE type of observation,\n"
                f"but you're feeding it a DIFFERENT type of observation.\n\n"
                f"**Solution:**\n"
                f"1. Make sure your environment settings match: 3 edges, 30 devices, 12 models\n"
                f"2. If still failing, use Random Agents instead\n"
                f"3. Run: python inspect_models.py (to diagnose model requirements)"
            ) from e
        else:
            raise RuntimeError(f"Error in create_dual_actions: {error_msg}\n{debug_info}") from e


def get_available_checkpoints():
    """Get list of available checkpoint files in the checkpoints folder."""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return [], []
    
    # Find all .zip files (A2C model format)
    slow_models = sorted(glob.glob(os.path.join(checkpoint_dir, "slow_agent_*.zip")))
    fast_models = sorted(glob.glob(os.path.join(checkpoint_dir, "fast_agent_*.zip")))
    
    return slow_models, fast_models


def load_pretrained_models(slow_model_path, fast_model_path, slow_env, fast_env):
    """Load pretrained A2C models from checkpoint files.
    
    IMPORTANT: Models must be trained on environments with the same dimensions!
    Models are trained with: num_edges=3, num_devices=30, num_models=12
    """
    slow_agent = None
    fast_agent = None
    
    try:
        if slow_model_path:
            slow_agent = A2C.load(slow_model_path.replace(".zip", ""))
            
            # Get observation space info (handle both Dict and Box spaces)
            model_obs_space = slow_agent.observation_space
            env_obs_space = slow_env.observation_space
            
            # Format observation space info for display
            if hasattr(model_obs_space, 'spaces'):  # Dict space
                model_obs_info = f"Dict with keys: {list(model_obs_space.spaces.keys())[:3]}..."
            else:
                model_obs_info = f"Shape {model_obs_space.shape}"
            
            if hasattr(env_obs_space, 'spaces'):  # Dict space
                env_obs_info = f"Dict with keys: {list(env_obs_space.spaces.keys())[:3]}..."
            else:
                env_obs_info = f"Shape {env_obs_space.shape}"
            
            st.success(f"✅ Slow Agent Loaded")
            
    except Exception as e:
        st.error(f"❌ Could not load slow agent: {str(e)}")
        with st.expander("📋 Error Details"):
            st.code(str(e))
        return None, None
    
    try:
        if fast_model_path:
            fast_agent = A2C.load(fast_model_path.replace(".zip", ""))
            # Get observation space info (handle both Dict and Box spaces)
            model_obs_space = fast_agent.observation_space
            env_obs_space = fast_env.observation_space
            model_obs_info = f"Shape {model_obs_space.shape}"
            
            if hasattr(env_obs_space, 'spaces'):  # Dict space
                env_obs_info = f"Dict with keys: {list(env_obs_space.spaces.keys())[:3]}..."
            else:
                env_obs_info = f"Shape {env_obs_space.shape}"
            
            st.success(f"✅ Fast Agent Loaded")
            
    except Exception as e:
        st.error(f"❌ Could not load fast agent: {str(e)}")
        with st.expander("📋 Error Details"):
            st.code(str(e))
        return None, None
    
    return slow_agent, fast_agent


def calculate_global_cache_hit_rate(env):
    """Calculate the global cache hit rate across all edges."""
    total_hits = sum(edge.cache_hits for edge in env.edge_servers)
    total_requests = sum(edge.total_requests for edge in env.edge_servers)
    
    if total_requests == 0:
        return 0.0
    
    return total_hits / total_requests


def extract_metrics_from_info(info):
    """Extract actual metrics from the info dict returned by environment.step()
    
    The environment provides the ground truth metrics in the info dict:
    - accuracy: average accuracy of completed tasks
    - latency: average latency (transmission + computation) of completed tasks
    - compute_latency: average computation latency (waiting + processing)
    - transmit_latency: average transmission latency
    - waiting_time: average waiting time in queue
    - reward: the actual reward computed by the environment
    
    This is more reliable than trying to extract from observation or tasks.
    """
    accuracy = info.get('accuracy', 0.0)
    latency = info.get('latency', 0.0)
    compute_latency = info.get('compute_latency', 0.0)
    transmit_latency = info.get('transmit_latency', 0.0)
    waiting_time = info.get('waiting_time', 0.0)
    
    # Calculate processing time (Compute Latency - Waiting Time)
    processing_time = max(0.0, compute_latency - waiting_time)
    
    return float(accuracy), float(latency), float(waiting_time), float(processing_time), float(transmit_latency)


def init_metrics_history():
    """Return a fresh metrics history dict that tracks per-step values and accumulators."""
    return {
        'timestep': [],
        'total_reward': [],
        'avg_accuracy': [],
        'avg_latency': [],
        'avg_waiting_time': [],
        'avg_processing_time': [],
        'avg_transmit_latency': [],
        'cache_hit_rate': [],
        'num_tasks': [],
        'total_latency': 0.0,
        'total_accuracy_weighted': 0.0,
        'total_waiting_time': 0.0,
        'total_processing_time': 0.0,
        'total_transmit_latency': 0.0,
        'total_tasks_completed': 0,
    }

# ==================== PAGE CONFIGURATION ====================

# Page configuration
st.set_page_config(
    page_title="MEC Environment Visualization",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📡 Mobile Edge Computing (MEC) Environment Visualizer")

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = None
    st.session_state.dual_env = None
    st.session_state.slow_agent = None
    st.session_state.fast_agent = None
    st.session_state.slow_env = None
    st.session_state.fast_env = None
    st.session_state.slow_obs = None
    st.session_state.fast_obs = None
    st.session_state.env_initialized = False
    st.session_state.current_obs = None
    st.session_state.current_tasks = []
    st.session_state.metrics_history = init_metrics_history()
    st.session_state.step_count = 0
    st.session_state.fast_step_in_slow = 0  # Track which fast step we're on within the slow episode
    st.session_state.prev_cache_hits = 0
    st.session_state.prev_total_requests = 0

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

with st.sidebar:
    st.subheader("Environment Setup")
    
    # Load default env args
    try:
        default_env_args = load_env_args()
    except:
        default_env_args = {
            'num_edges': 3,
            'num_devices': 30,
            'num_models': 12,
            'local_computing_range': [1, 2],
            'cloud_computing': 5,
            'edge_computing': 30,
            'edge_storage': 15,
            'bandwidth': 20,
            'i2i_tranmit_rate': 5,
            'cloud_download_rate': 250,
            'models_size_range': [2, 4],
            'task_size_input': [0.1, 0.2],
            'seed': 42,
            'acc_weight': 0.5,
            'latency_weight': 0.5,
            'task_arrival_rate': 0.6,
            'zipf_a': 1.2,
            'large_timescale_size': 20
        }
    
    # Configuration parameters
    num_edges = st.number_input("Number of Edge Servers", 1, 10, default_env_args.get('num_edges', 3))
    num_devices = st.number_input("Number of Mobile Devices", 1, 50, default_env_args.get('num_devices', 30))
    num_models = st.number_input("Number of Models", 1, 20, default_env_args.get('num_models', 12))
    edge_computing = st.slider("Edge Computing Power (GHz)", 1, 100, default_env_args.get('edge_computing', 30))
    edge_storage = st.slider("Edge Storage (GB)", 1, 100, default_env_args.get('edge_storage', 15))
    task_arrival_rate = st.slider("Task Arrival Rate", 0.1, 2.0, default_env_args.get('task_arrival_rate', 0.6), step=0.1)
    zipf_a = st.slider("Zipf Distribution Parameter", 0.5, 2.0, default_env_args.get('zipf_a', 1.2), step=0.1)
    
    seed = st.number_input("Random Seed", 0, 1000, default_env_args.get('seed', 42))
    
    st.divider()
    st.subheader("🤖 Model Selection")
    
    # Get available checkpoints
    slow_models, fast_models = get_available_checkpoints()
    
    use_pretrained = st.checkbox("Use Pretrained Models", value=False)
    
    if use_pretrained:
        st.info("ℹ️ Make sure the pretrained models match the environment configuration!")
        
    if use_pretrained and (slow_models or fast_models):
        slow_model_path = None
        fast_model_path = None
        
        if slow_models:
            slow_model_names = [os.path.basename(m) for m in slow_models]
            selected_slow = st.selectbox("Select Slow Agent (Caching)", slow_model_names, key="slow_model_select")
            slow_model_path = os.path.join("checkpoints", selected_slow)
        
        if fast_models:
            fast_model_names = [os.path.basename(m) for m in fast_models]
            selected_fast = st.selectbox("Select Fast Agent (Offloading/Exit)", fast_model_names, key="fast_model_select")
            fast_model_path = os.path.join("checkpoints", selected_fast)
        
        if not slow_models:
            st.warning("⚠️ No slow agent models found in checkpoints/")
        if not fast_models:
            st.warning("⚠️ No fast agent models found in checkpoints/")
    else:
        if use_pretrained:
            st.info("ℹ️ No pretrained models found. Make sure you have checkpoints in the checkpoints/ folder.")
        slow_model_path = None
        fast_model_path = None
    
    if st.button("🔄 Initialize Environment", use_container_width=True, key="init_btn"):
        try:
            # Validate configuration
            if num_devices < num_edges:
                st.warning(f"⚠️ Number of devices ({num_devices}) should be >= number of edges ({num_edges}). Adjusting...")
                num_devices = max(num_devices, num_edges)
            
            agent_type = "Pretrained" if use_pretrained else "Random"
            st.info(f"ℹ️ Initializing with: {num_edges} edges, {num_devices} devices, {num_models} models ({agent_type} agents)")
            

            env_args = default_env_args
            env_args['num_edges'] = num_edges
            env_args['num_devices'] = num_devices
            env_args['num_models'] = num_models
            env_args['edge_computing'] = edge_computing
            env_args['edge_storage'] = edge_storage
            env_args['task_arrival_rate'] = task_arrival_rate
            env_args['zipf_a'] = zipf_a
            # env_args = {
            #     'num_edges': num_edges,
            #     'num_devices': num_devices,
            #     'num_models': num_models,
            #     'local_computing_range': default_env_args['local_computing_range'],
            #     'cloud_computing': default_env_args['cloud_computing'],
            #     'edge_computing': edge_computing,
            #     'edge_storage': edge_storage,
            #     'bandwidth': default_env_args['bandwidth'],
            #     'i2i_tranmit_rate': default_env_args['i2i_tranmit_rate'],
            #     'cloud_download_rate': default_env_args['cloud_download_rate'],
            #     'models_size_range': default_env_args['models_size_range'],
            #     'task_size_input': default_env_args['task_size_input'],
            #     'cloud_propagation_delay': default_env_args.get('cloud_propagation_delay', 0.2),
            #     'compute_size_range': default_env_args.get('compute_size_range', [1e8, 2e8]),
            #     'large_timescale_size': default_env_args.get('large_timescale_size', 200),
            #     'task_arrival_rate': task_arrival_rate,
            #     'zipf_a': zipf_a,
            #     'seed': seed,
            # }
            # Create dual-timescale environment
            st.session_state.dual_env = MECDualTimeScaleEnv(**env_args)
            st.session_state.env = st.session_state.dual_env.base_env  # Keep reference to base_env for visualization
            
            # Initialize wrapped environments
            st.session_state.slow_env = SlowEnvWrapper(st.session_state.dual_env)
            st.session_state.fast_env = FastEnvWrapper(st.session_state.dual_env)
            
            # Load agents
            if use_pretrained:
                st.info("Attempting to load pretrained models...")
                st.session_state.slow_agent, st.session_state.fast_agent = load_pretrained_models(
                    slow_model_path, fast_model_path, 
                    st.session_state.slow_env, st.session_state.fast_env
                )
                
                if st.session_state.slow_agent is None:
                    st.warning("⚠️ Slow agent not found or failed to load. Falling back to Random caching.")
                    st.session_state.slow_agent = RandomDualOptimizer(st.session_state.dual_env, seed=seed)
                
                if st.session_state.fast_agent is None:
                    st.warning("⚠️ Fast agent not found or failed to load. Falling back to Random actions.")
            else:
                st.info("Using random agents.")
                st.session_state.slow_agent = RandomDualOptimizer(st.session_state.dual_env, seed=seed)
                st.session_state.fast_agent = None
            
            # Initialize wrapped environment observations
            st.session_state.slow_obs, _ = st.session_state.slow_env.reset()
            st.session_state.fast_obs, _ = st.session_state.fast_env.reset()
            
            # Also keep base environment obs for visualization
            st.session_state.current_obs = st.session_state.dual_env.reset()
            st.session_state.env_initialized = True
            st.session_state.step_count = 0
            st.session_state.prev_cache_hits = 0
            st.session_state.prev_total_requests = 0
            st.session_state.metrics_history = init_metrics_history()
            st.success("✅ Environment initialized successfully!")
        except Exception as e:
            import traceback
            st.error(f"❌ Error initializing environment: {str(e)}")
            with st.expander("📋 Detailed Error Information"):
                st.code(traceback.format_exc())

# Main content area
if not st.session_state.env_initialized:
    st.info("👈 Please configure and initialize the environment from the sidebar.")
else:
    # ==================== CONTROL PANEL (Always Visible) ====================
    st.subheader("⚙️ Simulation Controls")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("▶️ Execute Single Step", use_container_width=True, key="step_btn"):
            try:
                large_timescale_size = st.session_state.dual_env.base_env.large_timescale_size
                # Determine if using pretrained models
                slow_agent_type = "RL" if isinstance(st.session_state.slow_agent, A2C) else "Random"
                
                # Verify session state is ready
                if st.session_state.slow_obs is None or st.session_state.fast_obs is None:
                    st.error("❌ Session state not properly initialized. Please reset the environment.")
                    st.stop()
                
                print(f"Step {st.session_state.step_count}: Executing single step...")
                print(f"  Slow obs type: {type(st.session_state.slow_obs)}"
                      f", Fast obs type: {type(st.session_state.fast_obs)}")
                
                # Get dual-timescale actions using wrapped environment observations
                slow_obs, fast_obs, info, fast_action, made_slow_decision = create_dual_actions(
                    st.session_state.slow_env,
                    st.session_state.fast_env,
                    st.session_state.slow_agent,
                    st.session_state.fast_agent,
                    st.session_state.slow_obs,
                    st.session_state.fast_obs,
                    st.session_state.fast_step_in_slow,
                    slow_agent_type=slow_agent_type
                )
                
                # Update wrapped observations
                st.session_state.slow_obs = slow_obs
                st.session_state.fast_obs = fast_obs
                
                # Execute fast timestep
                st.session_state.current_obs = fast_obs
                st.session_state.step_count += 1
                st.session_state.fast_step_in_slow = (st.session_state.fast_step_in_slow + 1) % large_timescale_size
                
                # Extract metrics from the info dict (ground truth from environment)
                # Only collect metrics when there are actual tasks (matching experiment_utils)
                task_count = info.get('num_tasks', 0)
                if task_count > 0:
                    avg_accuracy, avg_latency, avg_waiting, avg_processing, avg_transmit = extract_metrics_from_info(info)
                    cache_hit_rate = calculate_global_cache_hit_rate(st.session_state.env)
                    reward = info.get('rewards', 0.0)
                    st.session_state.metrics_history['timestep'].append(st.session_state.step_count)
                    st.session_state.metrics_history['total_reward'].append(float(reward))
                    st.session_state.metrics_history['avg_accuracy'].append(avg_accuracy)
                    st.session_state.metrics_history['avg_latency'].append(avg_latency)
                    st.session_state.metrics_history['avg_waiting_time'].append(avg_waiting)
                    st.session_state.metrics_history['avg_processing_time'].append(avg_processing)
                    st.session_state.metrics_history['avg_transmit_latency'].append(avg_transmit)
                    st.session_state.metrics_history['cache_hit_rate'].append(cache_hit_rate)
                    st.session_state.metrics_history['num_tasks'].append(task_count)
                    st.session_state.metrics_history['total_latency'] += avg_latency * task_count
                    st.session_state.metrics_history['total_accuracy_weighted'] += avg_accuracy * task_count
                    st.session_state.metrics_history['total_waiting_time'] += avg_waiting * task_count
                    st.session_state.metrics_history['total_processing_time'] += avg_processing * task_count
                    st.session_state.metrics_history['total_transmit_latency'] += avg_transmit * task_count
                    st.session_state.metrics_history['total_tasks_completed'] += task_count
                
                if made_slow_decision:
                    st.success("✅ Step executed + New caching decision made!")
                else:
                    st.success("✅ Step executed (caching unchanged)!")
            except Exception as e:
                st.error(f"❌ Error executing step: {str(e)}")
                with st.expander("📋 Debug Information"):
                    import traceback
                    st.code(traceback.format_exc())
    
    with col_btn2:
        if st.button("🔄 Reset Environment", use_container_width=True, key="reset_btn"):
            try:
                # Reset wrapped environments
                st.session_state.slow_obs, _ = st.session_state.slow_env.reset()
                st.session_state.fast_obs, _ = st.session_state.fast_env.reset()
                st.session_state.current_obs = st.session_state.dual_env.reset()
                st.session_state.step_count = 0
                st.session_state.fast_step_in_slow = 0
                st.session_state.prev_cache_hits = 0
                st.session_state.prev_total_requests = 0
                st.session_state.metrics_history = init_metrics_history()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error resetting environment: {str(e)}")
    
    with col_btn3:
        num_auto_steps = st.number_input("Steps to auto-run", 1, 10000, 10, key="auto_steps")
        if st.button("⚡ Run Multiple Steps", use_container_width=True, key="auto_btn"):
            try:
                progress_bar = st.progress(0)
                large_timescale_size = st.session_state.dual_env.base_env.large_timescale_size
                
                # Verify session state is ready
                if st.session_state.slow_obs is None or st.session_state.fast_obs is None:
                    st.error("❌ Session state not properly initialized. Please reset the environment.")
                    st.stop()
                
                # Determine if using pretrained models
                slow_agent_type = "RL" if isinstance(st.session_state.slow_agent, A2C) else "Random"
                
                for i in range(num_auto_steps):
                    # Get dual-timescale actions using wrapped environment observations
                    slow_obs, fast_obs, info, fast_action, made_slow_decision = create_dual_actions(
                        st.session_state.slow_env,
                        st.session_state.fast_env,
                        st.session_state.slow_agent,
                        st.session_state.fast_agent,
                        st.session_state.slow_obs,
                        st.session_state.fast_obs,
                        st.session_state.fast_step_in_slow,
                        slow_agent_type=slow_agent_type
                    )
                    
                    # Update wrapped observations
                    st.session_state.slow_obs = slow_obs
                    st.session_state.fast_obs = fast_obs
                    
                    # Execute fast timestep
                    st.session_state.current_obs = fast_obs
                    st.session_state.step_count += 1
                    st.session_state.fast_step_in_slow = (st.session_state.fast_step_in_slow + 1) % large_timescale_size
                    
                    # Extract metrics from the info dict (ground truth from environment)
                    # Only collect metrics when there are actual tasks (matching experiment_utils)
                    task_count = info.get('num_tasks', 0)
                    if task_count > 0:
                        avg_accuracy, avg_latency, avg_waiting, avg_processing, avg_transmit = extract_metrics_from_info(info)
                        cache_hit_rate = calculate_global_cache_hit_rate(st.session_state.env)
                        reward = info.get('rewards', 0.0)
                        st.session_state.metrics_history['timestep'].append(st.session_state.step_count)
                        st.session_state.metrics_history['total_reward'].append(float(reward))
                        st.session_state.metrics_history['avg_accuracy'].append(avg_accuracy)
                        st.session_state.metrics_history['avg_latency'].append(avg_latency)
                        st.session_state.metrics_history['avg_waiting_time'].append(avg_waiting)
                        st.session_state.metrics_history['avg_processing_time'].append(avg_processing)
                        st.session_state.metrics_history['avg_transmit_latency'].append(avg_transmit)
                        st.session_state.metrics_history['cache_hit_rate'].append(cache_hit_rate)
                        st.session_state.metrics_history['num_tasks'].append(task_count)
                        st.session_state.metrics_history['total_latency'] += avg_latency * task_count
                        st.session_state.metrics_history['total_accuracy_weighted'] += avg_accuracy * task_count
                        st.session_state.metrics_history['total_waiting_time'] += avg_waiting * task_count
                        st.session_state.metrics_history['total_processing_time'] += avg_processing * task_count
                        st.session_state.metrics_history['total_transmit_latency'] += avg_transmit * task_count
                        st.session_state.metrics_history['total_tasks_completed'] += task_count
                    
                    progress_bar.progress((i + 1) / num_auto_steps)
                st.success(f"✅ Completed {num_auto_steps} steps!")
            except Exception as e:
                st.error(f"❌ Error running steps: {str(e)}")
                with st.expander("📋 Debug Information"):
                    import traceback
                    st.code(traceback.format_exc())
    
    st.divider()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Environment State", "📊 Real-time Metrics", "📈 History", "📊 Average Results"])
    
    with tab1:
        st.subheader("Current Environment Status")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Current Timestep", st.session_state.step_count)
        with col2:
            st.metric("📱 Mobile Devices", st.session_state.env.num_devices)
        with col3:
            st.metric("🖥️ Edge Servers", st.session_state.env.num_edges)
        with col4:
            st.metric("🎬 Models Available", st.session_state.env.num_models)
        
        st.divider()
        
        # Network Architecture Visualization
        st.subheader("🌐 Network Architecture")
        
        col_vis1, col_vis2 = st.columns([2, 1])
        
        with col_vis1:
            # Create network visualization with stable positioning
            fig_network = create_network_visualization(st.session_state.env, seed=42)
            st.plotly_chart(fig_network, use_container_width=True, key="network_topology")
        
        with col_vis2:
            st.subheader("🖥️ Edge Server Status")
            for edge_id, edge in enumerate(st.session_state.env.edge_servers):
                with st.expander(f"Edge Server {edge_id} ({len(edge.cached_models)} models cached)", expanded=False):
                    col_edge1, col_edge2 = st.columns(2)
                    with col_edge1:
                        st.metric("Computing Power", f"{edge.computing_power/1e9:.2f} GHz")
                        st.metric("Bandwidth", f"{edge.bandwidth/1e6:.2f} MHz")
                    with col_edge2:
                        st.metric("Cached Models", len(edge.cached_models))
                        st.metric("Cache Hits", edge.cache_hits)
                    
                    st.metric("Total Requests", edge.total_requests)
                    if edge.total_requests > 0:
                        hit_rate = (edge.cache_hits / edge.total_requests) * 100
                        st.metric("Cache Hit Rate", f"{hit_rate:.2f}%")
                    else:
                        st.info("No requests yet")
                    
                    if len(edge.cached_models) > 0:
                        st.write(f"**Cached Model IDs:** {sorted(edge.cached_models)}")

        
        st.divider()
        
        # Task and Processing Information
        col_task1, col_task2 = st.columns(2)
        
        with col_task1:
            st.subheader("📋 Observation Details")
            if st.session_state.current_obs:
                obs_data = st.session_state.current_obs
                
                # Display available observation keys
                st.write("**Available Observation Keys:**")
                st.write(", ".join(obs_data.keys()) if isinstance(obs_data, dict) else "Observation is not a dictionary")
                
                # Display each key safely
                if isinstance(obs_data, dict):
                    for key, value in obs_data.items():
                        try:
                            if isinstance(value, np.ndarray):
                                st.write(f"**{key}:** shape {value.shape}, dtype {value.dtype}")
                                if value.size <= 20:  # Only display small arrays
                                    st.dataframe(pd.DataFrame(value))
                            elif isinstance(value, list):
                                st.write(f"**{key}:** {value}")
                            else:
                                st.write(f"**{key}:** {value}")
                        except Exception as e:
                            st.write(f"**{key}:** (could not display - {type(value).__name__})")
        
        with col_task2:
            st.subheader("💾 Cache Information")
            
            # Create cache status visualization
            fig_cache = create_cache_visualization(st.session_state.env)
            st.plotly_chart(fig_cache, use_container_width=True, key="cache_status")
    
    with tab2:
        st.subheader("📊 Real-time Metrics")
        st.caption("Current timestep values (latest from each step)")
        
        # Display current metrics
        if len(st.session_state.metrics_history['timestep']) > 0:
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            
            with col_m1:
                st.metric("🎯 Latest Accuracy", 
                         f"{st.session_state.metrics_history['avg_accuracy'][-1]:.4f}")
            with col_m2:
                st.metric("⏱️ Latest Latency (ms)", 
                         f"{st.session_state.metrics_history['avg_latency'][-1]:.4f}")
            with col_m3:
                st.metric("⌛ Latest Waiting (ms)", 
                         f"{st.session_state.metrics_history['avg_waiting_time'][-1]:.4f}")
            with col_m4:
                st.metric("⚙️ Latest Processing (ms)", 
                         f"{st.session_state.metrics_history['avg_processing_time'][-1]:.4f}")
            with col_m5:
                st.metric("📡 Latest Transmit (ms)", 
                         f"{st.session_state.metrics_history['avg_transmit_latency'][-1]:.4f}")
            
            col_m6, col_m7 = st.columns(2)
            with col_m6:
                st.metric("🎯 Latest Reward", 
                         f"{st.session_state.metrics_history['total_reward'][-1]:.4f}")
            with col_m7:
                st.metric("💾 Latest Cache Hit Rate", 
                         f"{st.session_state.metrics_history['cache_hit_rate'][-1]:.2%}")
        
        st.divider()
        
        # Metrics visualization
        col_met1, col_met2 = st.columns(2)
        
        with col_met1:
            fig_reward = create_metrics_plot(
                st.session_state.metrics_history,
                'total_reward',
                'Cumulative Reward Over Time',
                'Timestep',
                'Reward'
            )
            st.plotly_chart(fig_reward, use_container_width=True, key="reward_chart")
        
        with col_met2:
            fig_accuracy = create_metrics_plot(
                st.session_state.metrics_history,
                'avg_accuracy',
                'Average Accuracy Over Time',
                'Timestep',
                'Accuracy'
            )
            st.plotly_chart(fig_accuracy, use_container_width=True, key="accuracy_chart")
        
        col_met3, col_met4 = st.columns(2)
        
        with col_met3:
            fig_latency = create_metrics_plot(
                st.session_state.metrics_history,
                'avg_latency',
                'Average Latency Over Time',
                'Timestep',
                'Latency (ms)'
            )
            st.plotly_chart(fig_latency, use_container_width=True, key="latency_chart")
        
        with col_met4:
            fig_cache = create_metrics_plot(
                st.session_state.metrics_history,
                'cache_hit_rate',
                'Cache Hit Rate Over Time',
                'Timestep',
                'Cache Hit Rate'
            )
            st.plotly_chart(fig_cache, use_container_width=True, key="cache_hit_rate_chart")
    
    with tab3:
        st.subheader("📈 Performance History")
        
        if len(st.session_state.metrics_history['timestep']) > 0:
            # Create comprehensive history dataframe
            history_df = pd.DataFrame({
                'Timestep': st.session_state.metrics_history['timestep'],
                'Total Reward': st.session_state.metrics_history['total_reward'],
                'Avg Accuracy': st.session_state.metrics_history['avg_accuracy'],
                'Avg Latency (ms)': st.session_state.metrics_history['avg_latency'],
                'Waiting Time (ms)': st.session_state.metrics_history['avg_waiting_time'],
                'Processing Time (ms)': st.session_state.metrics_history['avg_processing_time'],
                'Transmit Time (ms)': st.session_state.metrics_history['avg_transmit_latency'],
                'Cache Hit Rate': st.session_state.metrics_history['cache_hit_rate'],
                'Num Tasks': st.session_state.metrics_history['num_tasks']
            })
            
            st.dataframe(history_df, use_container_width=True, height=400)
            
            # Download history as CSV
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History as CSV",
                data=csv,
                file_name="mec_metrics_history.csv",
                mime="text/csv"
            )
            
            # Summary statistics
            st.subheader("Summary Statistics")
            
            # Calculate weighted averages for consistency
            total_tasks = st.session_state.metrics_history['total_tasks_completed']
            w_avg_accuracy = st.session_state.metrics_history['total_accuracy_weighted'] / total_tasks if total_tasks > 0 else 0.0
            w_avg_latency = st.session_state.metrics_history['total_latency'] / total_tasks if total_tasks > 0 else 0.0
            w_avg_waiting = st.session_state.metrics_history['total_waiting_time'] / total_tasks if total_tasks > 0 else 0.0
            w_avg_processing = st.session_state.metrics_history['total_processing_time'] / total_tasks if total_tasks > 0 else 0.0
            w_avg_transmit = st.session_state.metrics_history['total_transmit_latency'] / total_tasks if total_tasks > 0 else 0.0
            
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.write("**Reward Statistics**")
                st.write(f"Min: {min(st.session_state.metrics_history['total_reward']):.4f}")
                st.write(f"Max: {max(st.session_state.metrics_history['total_reward']):.4f}")
                st.write(f"Mean: {np.mean(st.session_state.metrics_history['total_reward']):.4f}")
            
            with col_stat2:
                st.write("**Accuracy Statistics**")
                st.write(f"Min: {min(st.session_state.metrics_history['avg_accuracy']):.4f}")
                st.write(f"Max: {max(st.session_state.metrics_history['avg_accuracy']):.4f}")
                st.write(f"Weighted Mean: {w_avg_accuracy:.4f}")

            st.write("---")
            col_stat3, col_stat4, col_stat5, col_stat6 = st.columns(4)
            
            with col_stat3:
                st.write("**Total Latency (ms)**")
                st.write(f"Min: {min(st.session_state.metrics_history['avg_latency']):.4f}")
                st.write(f"Max: {max(st.session_state.metrics_history['avg_latency']):.4f}")
                st.write(f"Weighted Mean: {w_avg_latency:.4f}")
            
            with col_stat4:
                st.write("**Waiting Time (ms)**")
                st.write(f"Min: {min(st.session_state.metrics_history['avg_waiting_time']):.4f}")
                st.write(f"Max: {max(st.session_state.metrics_history['avg_waiting_time']):.4f}")
                st.write(f"Weighted Mean: {w_avg_waiting:.4f}")

            with col_stat5:
                st.write("**Processing Time (ms)**")
                st.write(f"Min: {min(st.session_state.metrics_history['avg_processing_time']):.4f}")
                st.write(f"Max: {max(st.session_state.metrics_history['avg_processing_time']):.4f}")
                st.write(f"Weighted Mean: {w_avg_processing:.4f}")

            with col_stat6:
                st.write("**Transmit Time (ms)**")
                st.write(f"Min: {min(st.session_state.metrics_history['avg_transmit_latency']):.4f}")
                st.write(f"Max: {max(st.session_state.metrics_history['avg_transmit_latency']):.4f}")
                st.write(f"Weighted Mean: {w_avg_transmit:.4f}")

            st.write("---")
            with st.container():
                st.write("**Cache Hit Rate Statistics**")
                st.write(f"Min: {min(st.session_state.metrics_history['cache_hit_rate']):.2%}")
                st.write(f"Max: {max(st.session_state.metrics_history['cache_hit_rate']):.2%}")
                st.write(f"Mean: {np.mean(st.session_state.metrics_history['cache_hit_rate']):.2%}")
        else:
            st.info("⚠️ No history yet. Execute some steps to see history data.")
    
    with tab4:
        st.subheader("📊 Average Results (Cumulative Averages)")
        st.caption("Computed across all timesteps in the current session")
        
        if len(st.session_state.metrics_history['timestep']) > 0:
            # Calculate averages (matching experiment_utils approach)
            avg_reward = np.mean(st.session_state.metrics_history['total_reward'])
            total_tasks = st.session_state.metrics_history['total_tasks_completed']
            if total_tasks > 0:
                avg_accuracy = st.session_state.metrics_history['total_accuracy_weighted'] / total_tasks
                avg_latency = st.session_state.metrics_history['total_latency'] / total_tasks
                avg_waiting = st.session_state.metrics_history['total_waiting_time'] / total_tasks
                avg_processing = st.session_state.metrics_history['total_processing_time'] / total_tasks
                avg_transmit = st.session_state.metrics_history['total_transmit_latency'] / total_tasks
            else:
                avg_accuracy = 0.0
                avg_latency = 0.0
                avg_waiting = 0.0
                avg_processing = 0.0
                avg_transmit = 0.0
            avg_cache_hit_rate = np.mean(st.session_state.metrics_history['cache_hit_rate'])
            
            # Display as key metrics
            col_avg1, col_avg2, col_avg3, col_avg4 = st.columns(4)
            
            with col_avg1:
                st.metric("📈 Average Reward", f"{avg_reward:.4f}")
            with col_avg2:
                st.metric("📈 Average Accuracy", f"{avg_accuracy:.4f}")
            with col_avg3:
                st.metric("📈 Average Latency (ms)", f"{avg_latency:.4f}")
            with col_avg4:
                st.metric("📈 Average Cache Hit Rate", f"{avg_cache_hit_rate:.2%}")
            
            st.divider()
            
            # Latency Breakdown Visualization
            st.subheader("⏱️ Latency Breakdown")
            breakdown_fig = go.Figure(data=[
                go.Pie(labels=['Waiting Time', 'Processing Time', 'Transmission Time'], 
                       values=[avg_waiting, avg_processing, avg_transmit],
                       hole=.3,
                       marker_colors=['#EF553B', '#00CC96', '#636EFA'])
            ])
            breakdown_fig.update_layout(height=400, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(breakdown_fig, use_container_width=True)

            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("⌛ Avg Waiting", f"{avg_waiting:.4f} ms")
            col_b2.metric("⚙️ Avg Processing", f"{avg_processing:.4f} ms")
            col_b3.metric("📡 Avg Transmit", f"{avg_transmit:.4f} ms")

            st.divider()
            
            # Detailed breakdown table
            st.subheader("Detailed Breakdown")
            breakdown_data = {
                'Metric': ['Reward', 'Accuracy', 'Total Latency (ms)', 'Waiting Time (ms)', 'Processing Time (ms)', 'Transmit Time (ms)', 'Cache Hit Rate'],
                'Mean': [avg_reward, avg_accuracy, avg_latency, avg_waiting, avg_processing, avg_transmit, avg_cache_hit_rate],
                'Std Dev': [
                    np.std(st.session_state.metrics_history['total_reward']),
                    np.std(st.session_state.metrics_history['avg_accuracy']),
                    np.std(st.session_state.metrics_history['avg_latency']),
                    np.std(st.session_state.metrics_history['avg_waiting_time']),
                    np.std(st.session_state.metrics_history['avg_processing_time']),
                    np.std(st.session_state.metrics_history['avg_transmit_latency']),
                    np.std(st.session_state.metrics_history['cache_hit_rate'])
                ],
                'Min': [
                    min(st.session_state.metrics_history['total_reward']),
                    min(st.session_state.metrics_history['avg_accuracy']),
                    min(st.session_state.metrics_history['avg_latency']),
                    min(st.session_state.metrics_history['avg_waiting_time']),
                    min(st.session_state.metrics_history['avg_processing_time']),
                    min(st.session_state.metrics_history['avg_transmit_latency']),
                    min(st.session_state.metrics_history['cache_hit_rate'])
                ],
                'Max': [
                    max(st.session_state.metrics_history['total_reward']),
                    max(st.session_state.metrics_history['avg_accuracy']),
                    max(st.session_state.metrics_history['avg_latency']),
                    max(st.session_state.metrics_history['avg_waiting_time']),
                    max(st.session_state.metrics_history['avg_processing_time']),
                    max(st.session_state.metrics_history['avg_transmit_latency']),
                    max(st.session_state.metrics_history['cache_hit_rate'])
                ]
            }
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True)
            
            st.divider()
            
            # Show distribution of metrics
            st.subheader("Metric Distributions")
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                # Reward distribution
                fig_reward_dist = go.Figure()
                fig_reward_dist.add_trace(go.Histogram(
                    x=st.session_state.metrics_history['total_reward'],
                    nbinsx=20,
                    name='Reward'
                ))
                fig_reward_dist.update_layout(
                    title='Reward Distribution',
                    xaxis_title='Reward',
                    yaxis_title='Frequency',
                    height=400
                )
                st.plotly_chart(fig_reward_dist, use_container_width=True, key="reward_dist")
            
            with col_dist2:
                # Latency distribution
                fig_latency_dist = go.Figure()
                fig_latency_dist.add_trace(go.Histogram(
                    x=st.session_state.metrics_history['avg_latency'],
                    nbinsx=20,
                    name='Latency (ms)'
                ))
                fig_latency_dist.update_layout(
                    title='Latency Distribution',
                    xaxis_title='Latency (ms)',
                    yaxis_title='Frequency',
                    height=400
                )
                st.plotly_chart(fig_latency_dist, use_container_width=True, key="latency_dist")
        else:
            st.info("No data yet. Run some steps to see average results.")
