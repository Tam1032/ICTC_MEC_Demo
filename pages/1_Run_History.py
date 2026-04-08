import json
import os

import pandas as pd
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
        agent_mode = record.get("agent_mode", "")
        timestamp = record.get("started_at") or record.get("ended_at") or ""

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
            "reinforcement_learning": "Enabled" if agent_mode == "Pretrained" else "Disabled",
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


df = records_to_dataframe(records)
if "timestamp_sort" in df.columns:
    df = df.sort_values("timestamp_sort", ascending=False, na_position="last").reset_index(drop=True)

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
    if "Reinforcement Learning" in styles.columns:
        styles["Reinforcement Learning"] = "font-weight: 700; background-color: #EEF4FF;"
    return styles


hide_config_columns = st.checkbox("Hide configuration columns", value=True)
st.caption("Key results are highlighted for demo readability.")

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
    "reinforcement_learning": "Reinforcement Learning",
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


st.dataframe(
    table_df.style.apply(highlight_avg_columns, axis=None),
    use_container_width=True,
    hide_index=True,
)