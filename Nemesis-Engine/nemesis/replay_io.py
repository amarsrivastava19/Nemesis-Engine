import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

def save_replay_parquet(filename, states, values, policies_per_agent):
    """
    Args:
        states: np.ndarray [N, state_dim]
        values: np.ndarray [N]
        policies_per_agent: list of length 5, each entry is list of length-N arrays (ragged)
    """
    N = states.shape[0]
    data = {
        'state': [s.tolist() for s in states],
        'value': values.tolist(),
    }

    for i, agent_policies in enumerate(policies_per_agent):
        data[f'policy_agent_{i}'] = [p for p in agent_policies]

    table = pa.table(data)
    pq.write_table(table, filename)
    print(f"✅ Saved replay: {filename}")


def load_replay_parquet(filename):
    table = pq.read_table(filename)
    df = table.to_pandas()

    states = np.stack(df['state'].to_list()).astype(np.float32)
    values = df['value'].values.astype(np.float32)

    policies_padded = []
    policies_mask = []

    for col in [c for c in df.columns if c.startswith('policy_agent_')]:
        lists = df[col].to_list()
        max_len = max(len(x) for x in lists)
        padded = np.zeros((len(lists), max_len), dtype=np.float32)
        mask = np.zeros((len(lists), max_len), dtype=np.float32)

        for i, row in enumerate(lists):
            padded[i, :len(row)] = row
            mask[i, :len(row)] = 1.0

        policies_padded.append(padded)
        policies_mask.append(mask)

    return states, values, policies_padded, policies_mask