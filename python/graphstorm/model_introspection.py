import logging
from typing import Optional, Dict, Tuple
import torch.nn as nn

def _count_params(m: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def _fmt(n: int) -> str:
    return f"{n:,}"

def _safe_getattr(obj, attr):
    return getattr(obj, attr, None)

def _node_id(label: str) -> str:
    # Mermaid-safe simple IDs
    return label.replace(" ", "_").replace("-", "_").replace(".", "_")

def _abbr_params(n: int) -> str:
    """Return compact param counts like 22.5m, 2.6m, .55m, 123k, or raw int."""
    def _trim(x: float) -> str:
        s = f"{x:.2f}".rstrip("0").rstrip(".")
        return s[1:] if s.startswith("0.") else s
    if n >= 1_000_000:
        return f"{_trim(n / 1_000_000)}m"
    if n >= 1_000:
        return f"{_trim(n / 1_000)}k"
    return str(n)

def mermaid_component_diagram(model: nn.Module, tasks: Optional[Dict[str, object]] = None) -> str:
    """
    Build a component-level Mermaid flowchart:
    - Model container subgraph with params
    - Node input encoder subgraph with params (if present)
    - GNN encoder subgraph with params (if present)
    - Decoder subgraph with params, with one node per task if multi-task
    Note: Edge encoder is intentionally omitted from the diagram.
    """
    NBSP = "\u00A0"

    # Components
    node_enc = _safe_getattr(model, "node_input_encoder")
    gnn_enc  = _safe_getattr(model, "gnn_encoder")
    task_decoders = _safe_getattr(model, "task_decoders")
    single_decoder = _safe_getattr(model, "decoder")

    # Params per component
    model_total, model_trainable = _count_params(model)
    node_total = _count_params(node_enc)[0] if node_enc is not None else 0
    gnn_total  = _count_params(gnn_enc)[0]  if gnn_enc  is not None else 0

    # Decoder params (sum across tasks if multi-task)
    dec_total = 0
    multi_task = isinstance(task_decoders, dict) and len(task_decoders) > 0
    if multi_task:
        for _, dec in task_decoders.items():
            dec_total += _count_params(dec)[0]
    elif single_decoder is not None:
        dec_total = _count_params(single_decoder)[0]

    # Mermaid content
    lines = []
    lines.append("```mermaid")
    lines.append("graph TD")
    lines.append("  %% Styling")
    lines.append("  classDef meta fill:#eef,stroke:#99f,stroke-width:1px;")
    lines.append("")
    lines.append("  %% Model container")
    lines.append(f"  subgraph Model[{model.__class__.__name__}{NBSP}params={_abbr_params(model_total)}]")

    # Node encoder subgraph
    if node_enc is not None:
        lines.append(f"    %% Node input encoder")
        lines.append(f"    subgraph NodeIn[{node_enc.__class__.__name__}{NBSP}params={_abbr_params(node_total)}]")
        # Minimal placeholder to stabilize layout
        lines.append(f"      NodeEnc[ENCODER]")
        lines.append(f"    end")

    # GNN encoder subgraph
    if gnn_enc is not None:
        lines.append(f"    %% Relational GNN encoder")
        lines.append(f"    subgraph GNN[{gnn_enc.__class__.__name__}{NBSP}params={_abbr_params(gnn_total)}]")
        # Minimal placeholder to stabilize layout
        lines.append(f"      GNNCore[ENCODER]")
        lines.append(f"    end")

    # Decoder subgraph
    if multi_task or single_decoder is not None:
        lines.append(f"    %% Decoders")
        lines.append(f"    subgraph Dec[Decoder{NBSP}params={_abbr_params(dec_total)}]")
        if multi_task:
            for task_id, dec in task_decoders.items():
                node_id = _node_id(f"Dec_{task_id}")
                # Prefer TASK Classifier style where possible
                label = f"{str(task_id).upper()} Classifier"
                lines.append(f"      {node_id}[{label}]")
        else:
            lines.append(f"      DecMain[{single_decoder.__class__.__name__}]")
        lines.append(f"    end")

    lines.append("  end")
    lines.append("")
    lines.append("  class Model meta")
    lines.append("")
    # Data flow among subgraphs
    if node_enc is not None:
        lines.append("  Model --> NodeIn")
    if gnn_enc is not None:
        lines.append("  Model --> GNN")
    if multi_task or single_decoder is not None:
        lines.append("  Model --> Dec")
    if node_enc is not None and gnn_enc is not None:
        lines.append("  NodeIn --> GNN")
    if gnn_enc is not None and (multi_task or single_decoder is not None):
        lines.append("  GNN --> Dec")
    lines.append("")
    lines.append(f"  %% Params total trainable {_fmt(model_total)} / {_fmt(model_trainable)}")
    lines.append("```")

    return "\n".join(lines)

def save_mermaid_diagram(model: nn.Module, out_path: str, tasks: Optional[Dict[str, object]] = None):
    content = mermaid_component_diagram(model, tasks=tasks)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)