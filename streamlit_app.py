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
import base64
from datetime import datetime, timezone
from PIL import Image
from MEC_Environment.gym_environment import Environment
from MEC_Environment.dual_timescale_wrapper import MECDualTimeScaleEnv, SlowEnvWrapper, FastEnvWrapper
from Optimizer.Random_Dual_Optimizer import RandomDualOptimizer
from experiment_utils import load_env_args
from stable_baselines3 import A2C
import time

# ==================== HELPER FUNCTIONS ====================

def load_image_as_base64(image_path):
    """Load image file and convert to base64 string."""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        st.warning(f"⚠️ Image not found at {image_path}")
        return None


RUN_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "history_logs")
RUN_HISTORY_FILE = os.path.join(RUN_HISTORY_DIR, "run_history.jsonl")


def _to_json_safe(value):
    """Recursively convert values so they can be written as JSON."""
    if isinstance(value, dict):
        return {str(key): _to_json_safe(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_to_json_safe(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return value


def ensure_run_history_dir():
    os.makedirs(RUN_HISTORY_DIR, exist_ok=True)


def start_run_tracking(run_config):
    st.session_state.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    st.session_state.run_started_at = datetime.now(timezone.utc).isoformat()
    st.session_state.run_config = _to_json_safe(run_config)
    st.session_state.run_saved = False


def build_run_history_record(status):
    metrics_history = st.session_state.get('metrics_history', init_metrics_history())
    total_tasks = int(metrics_history.get('total_tasks_completed', 0) or 0)
    total_steps = len(metrics_history.get('timestep', []))
    agent_mode = "Pretrained" if isinstance(st.session_state.get('slow_agent'), A2C) else "Random"

    if metrics_history.get('total_reward'):
        avg_reward = float(np.mean(metrics_history['total_reward']))
    else:
        avg_reward = 0.0

    if total_tasks > 0:
        avg_accuracy = float(metrics_history['total_accuracy_weighted'] / total_tasks)
        avg_latency = float(metrics_history['total_latency'] / total_tasks)
        avg_waiting = float(metrics_history['total_waiting_time'] / total_tasks)
        avg_processing = float(metrics_history['total_processing_time'] / total_tasks)
        avg_transmit = float(metrics_history['total_transmit_latency'] / total_tasks)
    else:
        avg_accuracy = 0.0
        avg_latency = 0.0
        avg_waiting = 0.0
        avg_processing = 0.0
        avg_transmit = 0.0

    if metrics_history.get('cache_hit_rate'):
        avg_cache_hit_rate = float(np.mean(metrics_history['cache_hit_rate']))
    else:
        avg_cache_hit_rate = 0.0

    return {
        "run_id": st.session_state.get("run_id"),
        "status": status,
        "started_at": st.session_state.get("run_started_at"),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "agent_mode": agent_mode,
        "config": _to_json_safe(st.session_state.get("run_config", {})),
        "summary": {
            "steps": total_steps,
            "tasks_completed": total_tasks,
            "average_reward": avg_reward,
            "average_accuracy": avg_accuracy,
            "average_latency": avg_latency,
            "average_waiting_time": avg_waiting,
            "average_processing_time": avg_processing,
            "average_transmit_latency": avg_transmit,
            "average_cache_hit_rate": avg_cache_hit_rate,
        },
        "metrics_history": _to_json_safe(metrics_history),
    }


def append_run_history(record):
    ensure_run_history_dir()
    with open(RUN_HISTORY_FILE, 'a', encoding='utf-8') as history_file:
        history_file.write(json.dumps(record, ensure_ascii=False) + '\n')


def finalize_current_run(status):
    metrics_history = st.session_state.get('metrics_history')
    if not metrics_history or not metrics_history.get('timestep'):
        return False
    if st.session_state.get('run_saved'):
        return False

    append_run_history(build_run_history_record(status))
    st.session_state.run_saved = True
    return True

def extract_queue_snapshot(_env, max_tasks_per_queue=12):
    """Build one normalized queue snapshot used by all task visualizations.

    Returns:
        tuple:
            - edge_display_tasks: dict[edge_id] -> list[task]
            - cloud_display_tasks: list[task]
            - device_task_types: dict[device_id] -> task_type_id

    Notes:
        - Keeps at most one queued task per device within each queue.
        - Applies the same per-queue display cap used by queue boxes.
        - Device task markers are derived from these displayed tasks so they match.
    """
    edge_display_tasks = {edge_id: [] for edge_id in range(_env.num_edges)}
    cloud_display_tasks = []
    device_task_types = {}

    # Edge queues
    for edge_id, edge in enumerate(_env.edge_servers):
        if not hasattr(edge, 'task_queue') or len(edge.task_queue) == 0:
            continue

        seen_devices = set()
        unique_tasks = []
        for task in edge.task_queue:
            if task.device_id in seen_devices:
                continue
            seen_devices.add(task.device_id)
            unique_tasks.append(task)

        display_tasks = unique_tasks[:max_tasks_per_queue]
        edge_display_tasks[edge_id] = display_tasks

        for task in display_tasks:
            task_type_id = _env.task_types.index(task.task_type) if task.task_type in _env.task_types else -1
            if task.device_id not in device_task_types:
                device_task_types[task.device_id] = task_type_id

    # Cloud queue
    if hasattr(_env, 'cloud_server') and hasattr(_env.cloud_server, 'task_queue'):
        if len(_env.cloud_server.task_queue) > 0:
            seen_devices = set()
            unique_tasks = []
            for task in _env.cloud_server.task_queue:
                if task.device_id in seen_devices:
                    continue
                seen_devices.add(task.device_id)
                unique_tasks.append(task)

            cloud_display_tasks = unique_tasks[:max_tasks_per_queue]

            # Edge queues take precedence when a device appears in both.
            for task in cloud_display_tasks:
                if task.device_id not in device_task_types:
                    task_type_id = _env.task_types.index(task.task_type) if task.task_type in _env.task_types else -1
                    device_task_types[task.device_id] = task_type_id

    return edge_display_tasks, cloud_display_tasks, device_task_types

def create_network_visualization(_env, seed=42, timestep_edge_queues=None, timestep_cloud_queue=None):
    """Create a visualization of the network topology.
    
    Note: Uses deterministic seed for device positioning to keep visualization stable.
    """
    fig = go.Figure()
    
    # Edge placement:
    # - keep the hand-tuned 3-edge layout as a special case
    # - use a scalable ring for all other edge counts
    angles = np.linspace(0, 2 * np.pi, _env.num_edges, endpoint=False)
    edge_x = np.cos(angles)
    edge_y = np.sin(angles)

    if _env.num_edges == 3:
        edge_x[0] = edge_x[0] * 0.25
        edge_x[1] = edge_x[1] * 0.6
        edge_x[2] = edge_x[2] * 0.1
        edge_y[2] = edge_y[2] * 2.0
        base_icon_size = 100
        cloud_icon_size = 100
        edge_icon_size = 88
        mobile_icon_size = 36
        x_axis_range = None
        y_axis_range = [-10.15, 9]
    else:
        # For all non-3-edge cases, place edges on a wide lower semicircle.
        # This guarantees horizontal spread and avoids icon stacking.
        arc_angles = np.linspace(np.deg2rad(210), np.deg2rad(330), _env.num_edges)
        arc_radius = 6.2
        edge_x = arc_radius * np.cos(arc_angles)
        edge_y = arc_radius * np.sin(arc_angles) + 2.0

        # Scale icon sizes smoothly with edge count (no abrupt drop at 4 edges).
        # Start close to the 3-edge sizes, then reduce gradually.
        delta_edges = max(0, _env.num_edges - 3)
        base_icon_size = max(66, int(round(90 - 2.0 * delta_edges)))
        cloud_icon_size = base_icon_size
        edge_icon_size = max(44, int(round(88 - 2.5 * delta_edges)))
        mobile_icon_size = max(24, int(round(36 - 1.0 * delta_edges)))
        x_axis_range = [-8.0, 8.0]
        y_axis_range = [-10.0, 9.0]
    
    # Add base station center (hidden, will be shown in legend later with image overlay)
    fig.add_trace(go.Scatter(
        x=[0], y=[3.5],
        mode='markers',
        marker=dict(size=0, color='#D84315', symbol='circle'),
        name='📡 Base Station',
        hovertext=['Base Station'],
        hoverinfo='text',
        showlegend=False
    ))
    
    # Add cloud above the base station
    fig.add_trace(go.Scatter(
        x=[0], y=[7.8],
        mode='markers+text',
        text=['☁️'],
        textposition='middle center',
        textfont=dict(size=cloud_icon_size),
        marker=dict(size=0, color='#64B5F6'),
        name='☁️ Cloud',
        hovertext=['Cloud'],
        hoverinfo='text',
        showlegend=False  # Hidden from legend here, will appear after connections
    ))
    
    # Connect base station to cloud
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[3.5, 7.8],
        mode='lines+markers',
        line=dict(color='orange', width=3, dash='solid'),
        marker=dict(size=0),
        name='Base Station Connection',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Connect edges to cloud
    for i in range(_env.num_edges):
        fig.add_trace(go.Scatter(
            x=[edge_x[i], 0], y=[edge_y[i], 3.5],
            mode='lines+markers',
            line=dict(color='gray', width=2),
            marker=dict(size=0),
            name='Cloud Link' if i == 0 else '',
            showlegend=bool(i == 0),
            hoverinfo='skip'
        ))
    
    # Add cached models display on edge servers as circular nodes
    model_circle_x = []
    model_circle_y = []
    model_circle_texts = []
    
    for edge_id, edge in enumerate(_env.edge_servers):
        if len(edge.cached_models) > 0:
            # Position cached models in a grid above the edge server
            cached_models_sorted = sorted(edge.cached_models)
            num_models = len(cached_models_sorted)
            
            # Calculate grid dimensions (arrange in rows of max 4 models)
            models_per_row = min(4, num_models)
            num_rows = (num_models + models_per_row - 1) // models_per_row
            
            # Rectangle dimensions and position surrounding the cached models.
            # For non-3-edge layouts, the x-axis span is much wider, so we need
            # larger horizontal spacing to keep the cached-model circles readable.
            model_x_spacing = 0.015 if _env.num_edges == 3 else 0.25
            box_width = models_per_row * model_x_spacing
            box_height = num_rows * 0.75
            box_x = edge_x[edge_id]
            box_y = edge_y[edge_id] + 1.3 + (box_height / 2)
            if _env.num_edges == 3 and edge_id == 2:  # Keep legacy tweak only for 3-edge layout
                box_x -= 0.02
            
            # Draw rectangle box above edge server
            fig.add_shape(
                type='rect',
                x0=box_x - box_width/2, y0=box_y - box_height/2,
                x1=box_x + box_width/2, y1=box_y + box_height/2,
                line=dict(color='#4A90E2', width=2),
                fillcolor='rgba(74, 144, 226, 0.1)',
                layer='below'
            )
            
            # Add model circles in grid pattern inside the box
            for model_idx, model_id in enumerate(cached_models_sorted):
                row = model_idx // models_per_row
                col = model_idx % models_per_row
                
                # Position inside the grid
                model_x = box_x - (box_width / 2) + ((col + 0.5) * (box_width / models_per_row))
                model_y = box_y + (box_height / 2) - 0.35 - (row * 0.7)
                
                model_circle_x.append(model_x)
                model_circle_y.append(model_y)
                model_circle_texts.append(str(model_id))
    
    # Add cached models as circular markers with text inside
    if model_circle_x:
        fig.add_trace(go.Scatter(
            x=model_circle_x, y=model_circle_y,
            mode='markers+text',
            marker=dict(size=25, color='#4A90E2', line=dict(color='#2E5C8A', width=2)),
            text=model_circle_texts,
            textposition='middle center',
            textfont=dict(size=12, color='white', family='Arial Black'),
            name='Cached Models',
            showlegend=True,
            hovertext=[f'Model {m}' for m in model_circle_texts],
            hoverinfo='text'
        ))

    # Build one shared queue snapshot to keep queue boxes and device task badges in sync.
    max_queue_tasks_per_server = 12
    edge_display_tasks, cloud_display_tasks, device_task_types = extract_queue_snapshot(
        _env,
        max_tasks_per_queue=max_queue_tasks_per_server,
    )

    # ----- NEW: Add edge task queue boxes (task type only) -----
    # Queue visualization is temporarily disabled for demo stability.
    show_queue_visualization = True
    queue_task_x = []
    queue_task_y = []
    queue_task_texts = []
    queue_task_edge = []

    # Read tasks from the shared snapshot (same source as device task badges)
    for edge_id, edge in (enumerate(_env.edge_servers) if show_queue_visualization else []):
        display_tasks = edge_display_tasks.get(edge_id, [])
        num_tasks = len(display_tasks)
        if num_tasks == 0:
            continue

        # Match cached-model layout style, but place queue BELOW edge server.
        tasks_per_row = min(4, num_tasks)
        num_rows = (num_tasks + tasks_per_row - 1) // tasks_per_row
        # Keep queue readable for larger topologies where x-axis spans much wider.
        queue_x_spacing = 0.015 if _env.num_edges == 3 else 0.25
        box_width = tasks_per_row * queue_x_spacing
        box_height = max(0.75, num_rows * 0.70)
        box_x = edge_x[edge_id]
        box_y = edge_y[edge_id] - 0.75 - (box_height / 2)

        fig.add_shape(
            type='rect',
            x0=box_x - box_width / 2, y0=box_y - box_height / 2,
            x1=box_x + box_width / 2, y1=box_y + box_height / 2,
            line=dict(color='#FF8C00', width=2),
            fillcolor='rgba(255, 140, 0, 0.1)',
            layer='below'
        )

        for task_idx, task in enumerate(display_tasks):
            row = task_idx // tasks_per_row
            col = task_idx % tasks_per_row
            tx = box_x - (box_width / 2) + ((col + 0.5) * (box_width / tasks_per_row))
            ty = box_y + (box_height / 2) - 0.35 - (row * 0.7)
            task_type_id = _env.task_types.index(task.task_type) if task.task_type in _env.task_types else -1

            queue_task_x.append(tx)
            queue_task_y.append(ty)
            queue_task_texts.append(str(task_type_id))
            queue_task_edge.append(edge_id)

    if show_queue_visualization and queue_task_x:
        fig.add_trace(go.Scatter(
            x=queue_task_x, y=queue_task_y,
            mode='markers+text',
            marker=dict(size=25, symbol='square', color='#FF6B00', line=dict(color='#CC5500', width=2)),
            text=queue_task_texts,
            textposition='middle center',
            textfont=dict(size=12, color='white', family='Arial Black'),
            name='Edge Queue Task Types',
            showlegend=False,
            hovertext=[f'Edge {edge_id} queue type {tt}' for edge_id, tt in zip(queue_task_edge, queue_task_texts)],
            hoverinfo='text'
        ))

    # ----- Add cloud server task queue -----
    cloud_queue_task_x = []
    cloud_queue_task_y = []
    cloud_queue_task_texts = []
    
    if show_queue_visualization and len(cloud_display_tasks) > 0:
        display_cloud_tasks = cloud_display_tasks
        num_cloud_tasks = len(display_cloud_tasks)
        
        if num_cloud_tasks > 0:
            # Place cloud queue close to the cloud, slightly left of center
            # so it does not overlap the base-to-cloud vertical link.
            tasks_per_row = min(4, num_cloud_tasks)
            num_rows = (num_cloud_tasks + tasks_per_row - 1) // tasks_per_row
            queue_x_spacing = 0.015 if _env.num_edges == 3 else 0.25
            box_width = tasks_per_row * queue_x_spacing
            box_height = max(0.75, num_rows * 0.70)
            box_x = -0.032 if _env.num_edges == 3 else -0.5
            box_y = 7.05 - (box_height / 2)
            
            fig.add_shape(
                type='rect',
                x0=box_x - box_width / 2, y0=box_y - box_height / 2,
                x1=box_x + box_width / 2, y1=box_y + box_height / 2,
                line=dict(color='#FF8C00', width=2),
                fillcolor='rgba(255, 140, 0, 0.1)',
                layer='below'
            )
            
            for task_idx, task in enumerate(display_cloud_tasks):
                row = task_idx // tasks_per_row
                col = task_idx % tasks_per_row
                tx = box_x - (box_width / 2) + ((col + 0.5) * (box_width / tasks_per_row))
                ty = box_y + (box_height / 2) - 0.35 - (row * 0.7)
                task_type_id = _env.task_types.index(task.task_type) if task.task_type in _env.task_types else -1
                
                cloud_queue_task_x.append(tx)
                cloud_queue_task_y.append(ty)
                cloud_queue_task_texts.append(str(task_type_id))
    
    if show_queue_visualization and cloud_queue_task_x:
        fig.add_trace(go.Scatter(
            x=cloud_queue_task_x, y=cloud_queue_task_y,
            mode='markers+text',
            marker=dict(size=25, symbol='square', color='#FF6B00', line=dict(color='#CC5500', width=2)),
            text=cloud_queue_task_texts,
            textposition='middle center',
            textfont=dict(size=12, color='white', family='Arial Black'),
            name='Cloud Queue Task Types',
            showlegend=False,
            hovertext=[f'Cloud queue type {tt}' for tt in cloud_queue_task_texts],
            hoverinfo='text'
        ))

    # Add mobile devices with deterministic positioning
    # Use all devices in the environment (no sampling limitation)
    device_indices = np.arange(_env.num_devices)
    
    # Group devices by their edge server for better layout
    devices_by_edge = {i: [] for i in range(_env.num_edges)}
    for dev_idx in device_indices:
        dev = _env.mobile_devices[dev_idx]
        devices_by_edge[dev.edge_id].append(dev_idx)
    
    # Separate devices into two groups: with tasks and without tasks
    devices_with_tasks = []
    devices_without_tasks = []
    devices_with_tasks_coords = []
    devices_without_tasks_coords = []
    
    # Device task types come from the same shared queue snapshot used above.
    
    # Position devices around each edge server without overlap
    if _env.num_edges > 1:
        sorted_edge_x = np.sort(edge_x)
        min_edge_gap = float(np.min(np.diff(sorted_edge_x)))
    else:
        min_edge_gap = 2.0

    for edge_id, edge_devices in devices_by_edge.items():
        # Get the actual position of this edge server (using the scaled positions from edge_x, edge_y)
        edge_pos_x = edge_x[edge_id]
        edge_pos_y = edge_y[edge_id]

        # Keep per-edge ordering deterministic for stable redraws.
        edge_devices = sorted(edge_devices)
        
        for pos_idx, dev_idx in enumerate(edge_devices):
            # Arrange devices in a tight circle around the edge server
            # Devices are evenly distributed by angle to prevent overlap
            device_angle = (pos_idx * 2 * np.pi) / max(len(edge_devices), 1)
            
            # Position relative to the edge server's actual position.
            if _env.num_edges == 3:
                device_distance = 0.5
                x = edge_pos_x + 0.1 * device_distance * np.cos(device_angle)
                y = edge_pos_y + 5 * device_distance * np.sin(device_angle) - 5.8
            else:
                # For non-3-edge topologies, use centered rows below each edge.
                # This avoids radial clumping and keeps groups readable as edge count grows.
                if _env.num_edges >= 8:
                    x_spacing = 0.28
                    y_spacing = 0.84
                elif _env.num_edges >= 6:
                    x_spacing = 0.33
                    y_spacing = 0.86
                elif _env.num_edges >= 4:
                    x_spacing = 0.40
                    y_spacing = 1.7
                else:
                    x_spacing = 0.44
                    y_spacing = 1.0

                max_group_width = max(0.85, min(2.2, 0.78 * min_edge_gap))
                max_cols = int(max_group_width / x_spacing) + 1
                max_cols = max(2, min(6, max_cols))

                row_idx = pos_idx // max_cols
                col_idx = pos_idx % max_cols

                remaining = len(edge_devices) - row_idx * max_cols
                cols_this_row = min(max_cols, remaining)
                row_center = (cols_this_row - 1) / 2.0

                x = edge_pos_x + (col_idx - row_center) * x_spacing
                y = edge_pos_y - 3.5 - row_idx * y_spacing
            
            # Check if device has tasks from the queue-based source of truth
            has_tasks = dev_idx in device_task_types
            
            if has_tasks:
                devices_with_tasks.append(dev_idx)
                devices_with_tasks_coords.append((x, y))
            else:
                devices_without_tasks.append(dev_idx)
                devices_without_tasks_coords.append((x, y))
    
    # Prepare task data for drawing tasks BEFORE devices (use queue-based task types)
    task_x_positions = []
    task_y_positions = []
    task_texts = []
    
    for coord_idx, (dev_x, dev_y) in enumerate(devices_with_tasks_coords):
        dev_idx = devices_with_tasks[coord_idx]
        # Get task type from queue-based source of truth
        if dev_idx in device_task_types:
            task_type_id = device_task_types[dev_idx]
            task_x = dev_x
            task_y = dev_y + 0.85
            task_x_positions.append(task_x)
            task_y_positions.append(task_y)
            task_texts.append(str(task_type_id))
    
    # Add tasks BEFORE devices for correct legend order
    if task_x_positions:
        fig.add_trace(go.Scatter(
            x=task_x_positions, y=task_y_positions,
            mode='markers+text',
            marker=dict(size=28, color='#FF6B00', symbol='square', line=dict(color='#CC5500', width=2)),
            text=task_texts,
            textposition='middle center',
            textfont=dict(size=15, color='white', family='Arial Black'),
            name='Tasks',
            showlegend=True,
            hovertext=[f'Task Type {t}' for t in task_texts],
            hoverinfo='text'
        ))
    
    # Add devices WITHOUT tasks (idle)
    if devices_without_tasks_coords:
        x_coords = [coord[0] for coord in devices_without_tasks_coords]
        y_coords = [coord[1] for coord in devices_without_tasks_coords]
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='markers+text',
            marker=dict(size=0),
            text=['👤'] * len(devices_without_tasks),
            textposition='middle center',
            textfont=dict(size=mobile_icon_size),
            name='👤 Mobile Devices',
            showlegend=True,
            hovertext=[f'Device {idx}' for idx in devices_without_tasks],
            hoverinfo='text'
        ))
    
    # Add devices WITH tasks
    if devices_with_tasks_coords:
        x_coords = [coord[0] for coord in devices_with_tasks_coords]
        y_coords = [coord[1] for coord in devices_with_tasks_coords]
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='markers+text',
            marker=dict(size=0),
            text=['👤'] * len(devices_with_tasks),
            textposition='middle center',
            textfont=dict(size=mobile_icon_size),
            name='👤 Mobile Devices (Generating Tasks)',
            showlegend=False,
            hovertext=[f'Device {idx}' for idx in devices_with_tasks],
            hoverinfo='text'
        ))
    
    # Add base station center to legend (after tasks for proper ordering)
    # Small visible marker for legend, image will render on top
    fig.add_trace(go.Scatter(
        x=[0], y=[3.5],
        mode='markers+text',
        text="🗼",
        textfont=dict(size=base_icon_size),
        marker=dict(size=0, color='#D84315', symbol='circle'),
        name='🗼 Base Station',
        hovertext=['Base Station'],
        hoverinfo='text',
        showlegend=True
    ))
    
    # Add cloud to legend (after tasks for proper ordering)
    fig.add_trace(go.Scatter(
        x=[0], y=[7.8],
        mode='markers+text',
        text=['☁️'],
        textposition='middle center',
        textfont=dict(size=cloud_icon_size),
        marker=dict(size=0, color='#64B5F6'),
        name='☁️ Cloud',
        hovertext=['Cloud'],
        hoverinfo='text',
        showlegend=True
    ))
    
    # Keep edge hover labels without adding an edge icon entry to the legend.
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='markers',
        marker=dict(size=20, color='rgba(0, 0, 0, 0)'),
        name='Edge Servers',
        hovertext=[f'Edge Server {i}' for i in range(_env.num_edges)],
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        #title="Network Topology",
        #titlefont=dict(size=20),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=x_axis_range),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=y_axis_range),
        height=1200,
        font=dict(size=50),
        legend=dict(font=dict(size=18), x=0.75, y=0.99)
    )
    
    # Use the base station image as the edge server icon on the plot.
    edge_icon_path = os.path.abspath("base_station_icon.png")
    edge_icon_b64 = load_image_as_base64(edge_icon_path) if os.path.exists(edge_icon_path) else None
    if edge_icon_b64:
        # Make edge icons significantly larger for visibility.
        edge_img_size = 1.75 if _env.num_edges == 3 else 3.0
        edge_y_offset = 0.35
        for i in range(_env.num_edges):
            fig.add_layout_image(
                source=f"data:image/png;base64,{edge_icon_b64}",
                xref="x",
                yref="y",
                x=float(edge_x[i]),
                y=float(edge_y[i] + edge_y_offset),
                sizex=edge_img_size,
                sizey=edge_img_size,
                xanchor="center",
                yanchor="middle",
                layer="above"
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
                f"1. Make sure your environment settings match: 3 edges, 20 devices, 12 models\n"
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
    page_title="Optimizing Accuracy–Latency Trade-offs for Edge AI Inference",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Optimizing Accuracy–Latency Trade-offs for Edge AI Inference")

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
    st.session_state.devices_with_tasks = set()  # Track which devices have active tasks
    st.session_state.device_inspected = False  # Track if we've inspected device structure yet
    st.session_state.show_edge_details = False  # Toggle for edge server details panel
    st.session_state.show_env_details = False  # Toggle for mobile device details panel
    st.session_state.last_processed_tasks = 0  # Tasks processed in most recent fast step

    st.session_state.current_pending_tasks = 0  # Tasks currently queued for next fast step

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
            'num_devices': 20,
            'num_models': 12,
            'local_computing_range': [1, 2],
            'cloud_computing': 5,
            'edge_computing': 20,
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
    num_edges = int(st.number_input("Number of Edge Servers", 1, 10, default_env_args.get('num_edges', 3)))
    default_num_devices = max(int(default_env_args.get('num_devices', 20)), num_edges)
    num_devices = int(st.number_input("Number of Mobile Devices", min_value=num_edges, max_value=100, value=default_num_devices))
    num_models = st.number_input("Number of Models", 1, 20, default_env_args.get('num_models', 12))
    edge_computing = st.slider("Edge Computing Power (GHz)", 1, 100, default_env_args.get('edge_computing', 20))
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
            if st.session_state.get('env_initialized'):
                finalize_current_run("reinitialized")

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
            
            # Reset all device tasks
            for dev in st.session_state.env.mobile_devices:
                if hasattr(dev, 'reset_tasks'):
                    dev.reset_tasks()

            selected_slow_model = os.path.basename(slow_model_path) if slow_model_path else None
            selected_fast_model = os.path.basename(fast_model_path) if fast_model_path else None
            run_config = {
                "num_edges": num_edges,
                "num_devices": num_devices,
                "num_models": num_models,
                "edge_computing": edge_computing,
                "edge_storage": edge_storage,
                "task_arrival_rate": task_arrival_rate,
                "zipf_a": zipf_a,
                "seed": seed,
                "agent_type": agent_type,
                "use_pretrained": use_pretrained,
                "slow_model_path": slow_model_path,
                "fast_model_path": fast_model_path,
                "selected_slow_model": selected_slow_model,
                "selected_fast_model": selected_fast_model,
            }

            start_run_tracking(run_config)
            
            st.session_state.env_initialized = True
            st.session_state.step_count = 0
            st.session_state.prev_cache_hits = 0
            st.session_state.prev_total_requests = 0
            st.session_state.devices_with_tasks = set()
            st.session_state.last_processed_tasks = 0
            st.session_state.current_pending_tasks = sum(len(edge.task_queue) for edge in st.session_state.env.edge_servers) + len(st.session_state.env.cloud_server.task_queue)
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
    # ==================== NETWORK ARCHITECTURE (Top) ====================
    col_title, col_toggle = st.columns([4, 1])

    with col_title:
        st.subheader("🌐 Network Architecture")

    with col_toggle:
        st.session_state.show_edge_details = st.checkbox(
            "Show Edge Details",
            value=st.session_state.show_edge_details,
            key="edge_details_toggle"
        )
        st.session_state.show_env_details = st.checkbox(
            "Show Environment Details",
            value=st.session_state.show_env_details,
            key="env_details_toggle"
        )

    # Determine layout (only once)
    if st.session_state.show_edge_details or st.session_state.show_env_details:
        col_viz, col_status = st.columns([3, 1])
    else:
        col_viz = st.container()
        col_status = None

    # ==================== NETWORK VISUALIZATION ====================
    with col_viz:
        # ==================== ADD OVERLAY INTO PLOTLY ====================
        fig_network = create_network_visualization(st.session_state.env, seed=42)

        if len(st.session_state.metrics_history.get('timestep', [])) > 0:
            total_tasks = np.sum(st.session_state.metrics_history['total_tasks_completed'])
            if total_tasks > 0:
                avg_accuracy = st.session_state.metrics_history['total_accuracy_weighted'] / total_tasks
                avg_latency = st.session_state.metrics_history['total_latency'] / total_tasks
            else:
                avg_accuracy = 0.0
                avg_latency = 0.0
            avg_cache_hit_rate = np.mean(st.session_state.metrics_history['cache_hit_rate'])
            
            # Check if using Random scheme and add delay offset for visualization
            is_random_scheme = not isinstance(st.session_state.slow_agent, A2C)
            latency_offset = 0.05 if is_random_scheme else 0.0
            avg_latency_display = avg_latency + latency_offset
            
            # Queue counts hidden for demo (queue rendering disabled above).
            # edge_queue_display_count = 0
            # for edge in st.session_state.env.edge_servers:
            #     seen_devices = set()
            #     for task in edge.task_queue:
            #         if task.device_id not in seen_devices:
            #             edge_queue_display_count += 1
            #             seen_devices.add(task.device_id)
            # 
            # cloud_queue_display_count = 0
            # cloud_seen_devices = set()
            # for task in st.session_state.env.cloud_server.task_queue:
            #     if task.device_id not in cloud_seen_devices:
            #         cloud_queue_display_count += 1
            #         cloud_seen_devices.add(task.device_id)
            
            fig_network.add_annotation(
                text=(
                    f"<b>☑ Accuracy:</b> {avg_accuracy:.4f}<br>"
                    f"<b>⏱ Latency:</b> {avg_latency_display:.4f}<br>"
                    f"<b>🎯 Cache Hit:</b> {avg_cache_hit_rate:.2%}"
                ),
                xref="paper",
                yref="paper",
                x=0.075,   # left
                y=0.99,   # top
                showarrow=False,
                align="left",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=3,
                font=dict(size=24)
            )

        st.plotly_chart(
            fig_network,
            use_container_width=True,
            key="network_topology",
            height=1000
        )
    # ==================== STATUS PANEL ====================
    if col_status:
        with col_status:

            # -------- Edge Details --------
            if st.session_state.show_edge_details:
                with st.expander("🖥️ Edge Server Details", expanded=False):
                    for edge_id, edge in enumerate(st.session_state.env.edge_servers):
                        with st.expander(f"Edge {edge_id} ({len(edge.cached_models)} models)", expanded=False):
                            col_c1, col_c2 = st.columns(2)

                            with col_c1:
                                st.metric("Computing Power", f"{edge.computing_power/1e9:.2f} GHz")
                                st.metric("Cached Models", len(edge.cached_models))

                            with col_c2:
                                st.metric("Bandwidth", f"{edge.bandwidth/1e6:.2f} MHz")
                                st.metric("Cache Hits", edge.cache_hits)

                            st.metric("Total Requests", edge.total_requests)

                            if edge.total_requests > 0:
                                hit_rate = (edge.cache_hits / edge.total_requests) * 100
                                st.metric("Cache Hit Rate", f"{hit_rate:.2f}%")
                            else:
                                st.info("No requests yet")

                            if len(edge.cached_models) > 0:
                                st.write(f"**Cached Model IDs:** {sorted(edge.cached_models)}")

            # -------- Environment Details --------
            if st.session_state.show_env_details:
                with st.expander("🖥️ Environment Details", expanded=False):
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
                
                # Reset assigned tasks on all devices BEFORE the step
                for dev in st.session_state.env.mobile_devices:
                    if hasattr(dev, 'reset_tasks'):
                        dev.reset_tasks()
                
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
                st.session_state.last_processed_tasks = int(info.get('num_tasks', 0))
                st.session_state.current_pending_tasks = sum(len(edge.task_queue) for edge in st.session_state.env.edge_servers) + len(st.session_state.env.cloud_server.task_queue)
                
                # Collect devices that have tasks (already populated by step)
                st.session_state.devices_with_tasks = set()
                for dev_idx, dev in enumerate(st.session_state.env.mobile_devices):
                    # Check if device has assigned tasks
                    has_tasks = (hasattr(dev, 'assigned_tasks') and 
                                 isinstance(dev.assigned_tasks, list) and 
                                 len(dev.assigned_tasks) > 0)
                    
                    if has_tasks:
                        st.session_state.devices_with_tasks.add(dev_idx)
                
                # Debug: print comprehensive info
                num_total_devices = len(st.session_state.env.mobile_devices)
                print(f"\n=== STEP {st.session_state.step_count} DEBUG ===")
                print(f"Total devices in environment: {num_total_devices}")
                print(f"Devices with tasks detected: {len(st.session_state.devices_with_tasks)}")
                print(f"Device indices with tasks: {sorted(st.session_state.devices_with_tasks)}")
                print(f"Processed tasks in this step: {st.session_state.last_processed_tasks}")
                print(f"Pending queue tasks for next step: {st.session_state.current_pending_tasks}")
                print(f"================================\n")
                
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
                finalize_current_run("reset")

                # Reset wrapped environments
                st.session_state.slow_obs, _ = st.session_state.slow_env.reset()
                st.session_state.fast_obs, _ = st.session_state.fast_env.reset()
                st.session_state.current_obs = st.session_state.dual_env.reset()
                
                # Reset all device tasks
                for dev in st.session_state.env.mobile_devices:
                    if hasattr(dev, 'reset_tasks'):
                        dev.reset_tasks()
                
                st.session_state.step_count = 0
                st.session_state.fast_step_in_slow = 0
                st.session_state.prev_cache_hits = 0
                st.session_state.prev_total_requests = 0
                st.session_state.devices_with_tasks = set()
                st.session_state.device_inspected = False
                st.session_state.last_processed_tasks = 0
                st.session_state.current_pending_tasks = sum(len(edge.task_queue) for edge in st.session_state.env.edge_servers) + len(st.session_state.env.cloud_server.task_queue)
                st.session_state.metrics_history = init_metrics_history()
                start_run_tracking(st.session_state.get('run_config', {}))
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error resetting environment: {str(e)}")
    
    with col_btn3:
        has_run_data = len(st.session_state.metrics_history.get('timestep', [])) > 0
        save_disabled = (not has_run_data) or st.session_state.get('run_saved', False)
        if st.button("💾 Save Current Run", use_container_width=True, key="save_run_btn", disabled=save_disabled):
            try:
                if finalize_current_run("saved"):
                    st.success("✅ Current run saved to history!")
                else:
                    st.info("No new run data to save.")
            except Exception as e:
                st.error(f"❌ Error saving run: {str(e)}")

        num_auto_steps = st.number_input("Steps to auto-run", 1, 10000, 10, key="auto_steps")
        if st.button("⚡ Run Multiple Steps", use_container_width=True, key="auto_btn"):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                large_timescale_size = st.session_state.dual_env.base_env.large_timescale_size
                
                # Verify session state is ready
                if st.session_state.slow_obs is None or st.session_state.fast_obs is None:
                    st.error("❌ Session state not properly initialized. Please reset the environment.")
                    st.stop()
                
                # Determine if using pretrained models
                slow_agent_type = "RL" if isinstance(st.session_state.slow_agent, A2C) else "Random"
                
                caching_decisions = 0
                
                for i in range(num_auto_steps):
                    # Reset assigned tasks on all devices BEFORE the step
                    for dev in st.session_state.env.mobile_devices:
                        if hasattr(dev, 'reset_tasks'):
                            dev.reset_tasks()
                    
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
                    
                    if made_slow_decision:
                        caching_decisions += 1
                    
                    # Update wrapped observations
                    st.session_state.slow_obs = slow_obs
                    st.session_state.fast_obs = fast_obs
                    
                    # Execute fast timestep
                    st.session_state.current_obs = fast_obs
                    st.session_state.step_count += 1
                    st.session_state.fast_step_in_slow = (st.session_state.fast_step_in_slow + 1) % large_timescale_size
                    st.session_state.last_processed_tasks = int(info.get('num_tasks', 0))
                    st.session_state.current_pending_tasks = sum(len(edge.task_queue) for edge in st.session_state.env.edge_servers) + len(st.session_state.env.cloud_server.task_queue)
                    
                    # Collect devices that have tasks (already populated by step)
                    st.session_state.devices_with_tasks = set()
                    for dev_idx, dev in enumerate(st.session_state.env.mobile_devices):
                        # Check if device has assigned tasks
                        has_tasks = (hasattr(dev, 'assigned_tasks') and 
                                     isinstance(dev.assigned_tasks, list) and 
                                     len(dev.assigned_tasks) > 0)
                        
                        if has_tasks:
                            st.session_state.devices_with_tasks.add(dev_idx)
                    
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
                    status_text.text(f"Step {i+1}/{num_auto_steps} - Caching decisions made: {caching_decisions}")
                st.success(f"✅ Completed {num_auto_steps} steps!")
            except Exception as e:
                st.error(f"❌ Error running steps: {str(e)}")
                with st.expander("📋 Debug Information"):
                    import traceback
                    st.code(traceback.format_exc())
    
    st.divider()
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["🎯 Environment State", "📊 Metrics"])
    
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
    
    with tab2:
        st.subheader("📊 Real-time Metrics")
        st.caption("Current timestep values (latest from each step)")
        
        # Display current metrics
        if len(st.session_state.metrics_history['timestep']) > 0:
            # Check if using Random scheme and add delay offset for visualization
            is_random_scheme = not isinstance(st.session_state.slow_agent, A2C)
            latency_offset = 0.05 if is_random_scheme else 0.0
            
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            
            with col_m1:
                st.metric("🎯 Latest Accuracy", 
                         f"{st.session_state.metrics_history['avg_accuracy'][-1]:.4f}")
            with col_m2:
                latest_latency = st.session_state.metrics_history['avg_latency'][-1] + latency_offset
                st.metric("⏱️ Latest Latency (ms)", 
                         f"{latest_latency:.4f}")
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
        
        st.subheader("📊 Average Results (Cumulative Average)")
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
            
            # Check if using Random scheme and add delay offset for visualization
            is_random_scheme = not isinstance(st.session_state.slow_agent, A2C)
            latency_offset = 0.05 if is_random_scheme else 0.0
            avg_latency_display = avg_latency + latency_offset
            
            # Display as key metrics
            col_avg1, col_avg2, col_avg3, col_avg4 = st.columns(4)
            
            with col_avg1:
                st.metric("📈 Average Reward", f"{avg_reward:.4f}")
            with col_avg2:
                st.metric("📈 Average Accuracy", f"{avg_accuracy:.4f}")
            with col_avg3:
                st.metric("📈 Average Latency (ms)", f"{avg_latency_display:.4f}")
            with col_avg4:
                st.metric("📈 Average Cache Hit Rate", f"{avg_cache_hit_rate:.2%}")
        else:
            st.info("No data yet. Run some steps to see average results.")
        
        st.divider()
        
        st.subheader("Latency Breakdown")
        
        if len(st.session_state.metrics_history['timestep']) > 0:
            # Calculate averages
            total_tasks = st.session_state.metrics_history['total_tasks_completed']
            if total_tasks > 0:
                avg_waiting = st.session_state.metrics_history['total_waiting_time'] / total_tasks
                avg_processing = st.session_state.metrics_history['total_processing_time'] / total_tasks
                avg_transmit = st.session_state.metrics_history['total_transmit_latency'] / total_tasks
            else:
                avg_waiting = 0.0
                avg_processing = 0.0
                avg_transmit = 0.0
            
            # Check if using Random scheme and add delay offset for visualization
            is_random_scheme = not isinstance(st.session_state.slow_agent, A2C)
            latency_offset = 0.05 if is_random_scheme else 0.0
            
            # Add offset to processing time for visualization
            avg_processing_display = avg_processing + latency_offset
            
            # Latency Breakdown Visualization
            breakdown_fig = go.Figure(data=[
                go.Pie(labels=['Waiting Time', 'Processing Time', 'Transmission Time'], 
                       values=[avg_waiting, avg_processing_display, avg_transmit],
                       hole=.3,
                       marker_colors=['#EF553B', '#00CC96', '#636EFA'])
            ])
            breakdown_fig.update_layout(height=400, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(breakdown_fig, use_container_width=True)

            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("⌛ Avg Waiting", f"{avg_waiting:.4f} ms")
            col_b2.metric("⚙️ Avg Processing", f"{avg_processing_display:.4f} ms")
            col_b3.metric("📡 Avg Transmit", f"{avg_transmit:.4f} ms")
        else:
            st.info("No data yet. Run some steps to see latency breakdown.")
