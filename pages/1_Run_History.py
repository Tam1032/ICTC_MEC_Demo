import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
HISTORY_DIR = os.path.join(ROOT_DIR, "history_logs")
HISTORY_FILE = os.path.join(HISTORY_DIR, "run_history.jsonl")


st.set_page_config(page_title="Run History", page_icon="🧾", layout="wide")
st.title("Run History")


def load_history_records():
    if not os.path.exists(HISTORY_FILE):
        return []

    records = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as history_file:
        for line in history_file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def save_history_records(records):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as history_file:
        for record in records:
            history_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def records_to_dataframe(records):
    rows = []
    for record in records:
        config = record.get("config", {})
        summary = record.get("summary", {})
        agent_mode = str(record.get("agent_mode", ""))
        timestamp = record.get("started_at") or record.get("ended_at") or ""

        method_display = {
            "Pretrained Models": "Pretrained Models",
            "Pretrained": "Pretrained Models",
            "Cloud Only": "Cloud Only",
            "Local Offload": "Local Offload",
            "Local Edge Only": "Local Offload",
            "Random": "Random",
            "": "Random",
        }.get(agent_mode, agent_mode)

        try:
            timestamp_dt = pd.to_datetime(timestamp, utc=True)
            timestamp_display = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            timestamp_dt = pd.NaT
            timestamp_display = str(timestamp)

        rows.append({
            "run_id": record.get("run_id", ""),
            "timestamp": timestamp_display,
            "timestamp_sort": timestamp_dt,
            "method": method_display,
            "seed": config.get("seed", ""),
            "num_edges": config.get("num_edges", ""),
            "num_devices": config.get("num_devices", ""),
            "num_models": config.get("num_models", ""),
            "edge_computing": config.get("edge_computing", ""),
            "edge_storage": config.get("edge_storage", ""),
            "task_arrival_rate": config.get("task_arrival_rate", ""),
            "zipf_a": config.get("zipf_a", ""),
            "avg_reward": summary.get("average_reward", 0.0),
            "avg_accuracy": summary.get("average_accuracy", 0.0),
            "avg_latency": summary.get("average_latency", 0.0),
        })

    return pd.DataFrame(rows)


records = load_history_records()

if not records:
    st.info("No saved runs found yet. Click 'Save Current Run' in the main page after running the simulation.")
    st.stop()

# Initialize session state for checkboxes
if 'selected_runs' not in st.session_state:
    st.session_state.selected_runs = {}

df = records_to_dataframe(records)
if "timestamp_sort" in df.columns:
    df = df.sort_values("timestamp_sort", ascending=False, na_position="last").reset_index(drop=True)

# Initialize checkboxes for all runs
for idx, row in df.iterrows():
    run_id = row['run_id']
    if run_id not in st.session_state.selected_runs:
        st.session_state.selected_runs[run_id] = False

avg_columns = ["Avg Reward", "Avg Accuracy", "Avg Latency"]
config_columns = [
    "seed",
    "num_edges",
    "num_devices",
    "num_models",
    "edge_computing",
    "edge_storage",
    "task_arrival_rate",
    "zipf_a",
]


def highlight_avg_columns(dataframe):
    styles = pd.DataFrame("", index=dataframe.index, columns=dataframe.columns)
    for column in avg_columns:
        if column in styles.columns:
            styles[column] = "background-color: #E8F8F0; font-weight: 700; color: #0B5D3B;"
    if "Optimizer" in styles.columns:
        styles["Optimizer"] = "font-weight: 700; background-color: #EEF4FF;"
    return styles


hide_config_columns = st.checkbox("Hide configuration columns", value=True)
st.caption("Key results are highlighted for demo readability. Check the runs you want to compare.")

with st.expander("Manage Records (optional)", expanded=False):
    run_ids = [str(record.get("run_id", "")) for record in records]
    selected_run_id = st.selectbox("Run ID to remove", options=run_ids, key="delete_run_id")
    col_manage_1, col_manage_2 = st.columns(2)

    with col_manage_1:
        if st.button("Remove Selected Run", type="secondary", key="remove_selected_run"):
            updated_records = [record for record in records if str(record.get("run_id", "")) != selected_run_id]
            if len(updated_records) == len(records):
                st.warning("Selected run was not found in history.")
            else:
                save_history_records(updated_records)
                st.success("Selected run removed.")
                st.rerun()

    with col_manage_2:
        confirm_clear_all = st.checkbox("Confirm clear all runs", value=False, key="confirm_clear_all_runs")
        if st.button("Clear All Runs", type="secondary", disabled=not confirm_clear_all, key="clear_all_runs"):
            save_history_records([])
            st.success("All runs removed.")
            st.rerun()

if hide_config_columns:
    table_df = df.drop(columns=[col for col in config_columns if col in df.columns])
else:
    table_df = df

if "timestamp_sort" in table_df.columns:
    table_df = table_df.drop(columns=["timestamp_sort"])

display_names = {
    "run_id": "Run ID",
    "timestamp": "Timestamp",
    "method": "Optimizer",
    "seed": "Seed",
    "num_edges": "Edges",
    "num_devices": "Devices",
    "num_models": "Models",
    "edge_computing": "Edge Computing",
    "edge_storage": "Edge Storage",
    "task_arrival_rate": "Task Arrival Rate",
    "zipf_a": "Zipf a",
    "avg_reward": "Avg Reward",
    "avg_accuracy": "Avg Accuracy",
    "avg_latency": "Avg Latency",
}
table_df = table_df.rename(columns=display_names)

# Create table with embedded compare tick as the first column
st.subheader("📊 Run History")
table_display_df = table_df.copy()
table_display_df.insert(0, "Compare", False)

edited_table_df = st.data_editor(
    table_display_df,
    use_container_width=True,
    hide_index=True,
    key="run_history_editor",
    disabled=[col for col in table_display_df.columns if col != "Compare"],
    column_config={
        "Compare": st.column_config.CheckboxColumn("Compare"),
    },
)

# ==================== COMPARISON SECTION ====================
st.divider()
st.subheader("📊 Compare Selected Runs")

# Get selected runs from the Compare column in the table
selected_run_ids = []
if "Run ID" in edited_table_df.columns:
    selected_run_ids = edited_table_df.loc[edited_table_df["Compare"], "Run ID"].astype(str).tolist()
    selected_run_ids = [run_id for run_id in selected_run_ids if run_id]

selected_records = [record for record in records if str(record.get("run_id", "")) in selected_run_ids]

# Generate comparison plots if runs are selected
if selected_records:
    # Prepare data for comparison
    comparison_data = []
    for record in selected_records:
        method = str(record.get("agent_mode", "")).strip()
        
        # Normalize method display name
        method_display = {
            "Pretrained Models": "Pretrained Models",
            "Pretrained": "Pretrained Models",
            "Cloud Only": "Cloud Only",
            "Local Offload": "Local Offload",
            "Local Edge Only": "Local Offload",
            "Random": "Random",
            "": "Random",
        }.get(method, method)
        
        summary = record.get("summary", {})
        config = record.get("config", {})
        run_id = record.get("run_id", "")
        
        comparison_data.append({
            "method": method_display,
            "run_id": run_id[:8],  # Short run ID
            "avg_accuracy": summary.get("average_accuracy", 0.0),
            "avg_latency": summary.get("average_latency", 0.0),
            "avg_reward": summary.get("average_reward", 0.0),
            "avg_cache_hit_rate": summary.get("average_cache_hit_rate", 0.0),
            "num_edges": config.get("num_edges", ""),
            "num_devices": config.get("num_devices", ""),
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create comparison plots
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.subheader("📈 Accuracy Comparison")
        
        fig_accuracy = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for color_idx, row in enumerate(comparison_df.to_dict("records")):
            fig_accuracy.add_trace(go.Bar(
                x=[row["method"]],
                y=[row["avg_accuracy"]],
                name=f"{row['method']} ({row['run_id']})",
                text=f"{row['avg_accuracy']:.4f}",
                textposition="outside",
                textfont=dict(size=18, color='black', family='Arial'),
                marker=dict(
                    color=colors[color_idx % len(colors)],
                    line=dict(color='black', width=1)
                ),
                hovertemplate=f"<b>{row['method']}</b><br>Accuracy: {row['avg_accuracy']:.4f}<extra></extra>"
            ))
        
        accuracy_max = comparison_df['avg_accuracy'].max() if len(comparison_df) > 0 else 1.0
        accuracy_top = max(accuracy_max + 0.05, 1.0)
        fig_accuracy.update_layout(
            xaxis_title="Method",
            yaxis_title="Average Accuracy",
            yaxis=dict(range=[0.5, accuracy_top]),
            height=600,
            showlegend=False,
            hovermode='x unified',
            margin=dict(b=50, l=50, r=50, t=50)
        )
        
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with col_plot2:
        st.subheader("⏱️ Latency Comparison")
        
        fig_latency = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for color_idx, row in enumerate(comparison_df.to_dict("records")):
            fig_latency.add_trace(go.Bar(
                x=[row["method"]],
                y=[row["avg_latency"]],
                name=f"{row['method']} ({row['run_id']})",
                text=f"{row['avg_latency']:.4f}",
                textposition="outside",
                textfont=dict(size=18, color='black', family='Arial'),
                marker=dict(
                    color=colors[color_idx % len(colors)],
                    line=dict(color='black', width=1)
                ),
                hovertemplate=f"<b>{row['method']}</b><br>Latency: {row['avg_latency']:.4f}<extra></extra>"
            ))
        
        latency_max = comparison_df['avg_latency'].max() if len(comparison_df) > 0 else 1.0
        latency_top = max(latency_max + 0.05, 0.3)
        fig_latency.update_layout(
            xaxis_title="Method",
            yaxis_title="Average Latency (ms)",
            yaxis=dict(range=[0.3, latency_top]),
            height=600,
            showlegend=False,
            hovermode='x unified',
            margin=dict(b=50, l=50, r=50, t=50)
        )
        
        st.plotly_chart(fig_latency, use_container_width=True)
else:
    st.info("Select at least one run from the table above to view comparison plots.")