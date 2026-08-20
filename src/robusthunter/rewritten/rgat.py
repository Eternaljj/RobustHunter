from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from robusthunter.method_contract import RELATIONS

from .contracts import NODE_TYPES, canonical_sha256


NODE_TYPE_VOCAB = tuple(sorted(NODE_TYPES))
RELATION_VOCAB = tuple(sorted(RELATIONS))
NODE_TYPE_TO_ID = {value: index for index, value in enumerate(NODE_TYPE_VOCAB)}
RELATION_TO_ID = {value: index for index, value in enumerate(RELATION_VOCAB)}

SYSCALL_TO_RELATION = {
    "1": "fork",
    "2": "fork",
    "3": "exec",
    "4": "exit",
    "6": "unlink",
    "7": "write",
    "8": "recv",
    "9": "send",
    "10": "write",
    "11": "unlink",
    "12": "read",
    "13": "mmap",
    "14": "read",
    "15": "write",
    "16": "connect",
}
_CANDIDATE_NODE_TYPES = {"0": "File", "1": "Process", "2": "Socket", "3": "Registry"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/-]+")


@dataclass(frozen=True)
class GraphFeatures:
    node_ids: tuple[str, ...]
    node_types: tuple[str, ...]
    attributes: torch.Tensor
    type_ids: torch.Tensor
    edge_index: torch.Tensor
    relation_ids: torch.Tensor
    relation_names: tuple[str, ...]
    times: torch.Tensor
    feature_sha256: str


def canonical_node_type(value: Any) -> str:
    text = str(value or "").replace(".0", "").strip()
    if text in _CANDIDATE_NODE_TYPES:
        return _CANDIDATE_NODE_TYPES[text]
    title = text.title()
    if title not in NODE_TYPES:
        raise ValueError(f"unknown node type: {value!r}")
    return title


def canonical_relation(value: Any) -> str:
    text = str(value or "").replace(".0", "").strip().lower()
    relation = SYSCALL_TO_RELATION.get(text, text)
    if relation not in RELATION_TO_ID:
        raise ValueError(f"unknown relation: {value!r}")
    return relation


def node_identifier(node: dict[str, Any], fallback: int) -> str:
    for key in ("id", "graph_node_id", "node_id", "Unnamed: 0", "n_id"):
        if node.get(key) not in (None, ""):
            return str(node[key]).replace(".0", "")
    return str(fallback)


def node_attribute_text(node: dict[str, Any]) -> str:
    values: list[str] = []
    attributes = node.get("attributes")
    if isinstance(attributes, dict):
        values.extend(str(value) for _, value in sorted(attributes.items()))
    for key in ("label", "attribute", "name", "exe", "path", "ip", "port", "pid"):
        if node.get(key) not in (None, ""):
            values.append(str(node[key]))
    return " ".join(values).strip()


def deterministic_attribute_features(
    nodes: list[dict[str, Any]],
    dimension: int = 64,
) -> torch.Tensor:
    """Versioned, replayable attribute features with no external model calls."""
    rows = torch.zeros((len(nodes), dimension), dtype=torch.float32)
    for row, node in enumerate(nodes):
        tokens = _TOKEN_RE.findall(node_attribute_text(node).lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            rows[row, bucket] += sign
    return F.normalize(rows, p=2, dim=1)


def _edge_endpoint(edge: dict[str, Any], source: bool) -> str:
    keys = ("source", "src", "source_graph_node_id") if source else (
        "target",
        "dst",
        "target_graph_node_id",
    )
    for key in keys:
        if edge.get(key) not in (None, ""):
            return str(edge[key]).replace(".0", "")
    return ""


def compile_graph_features(
    graph: dict[str, Any],
    *,
    attribute_dim: int = 64,
) -> GraphFeatures:
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    node_ids = tuple(node_identifier(node, index) for index, node in enumerate(nodes))
    node_types = tuple(
        canonical_node_type(node.get("type") or node.get("node_type"))
        for node in nodes
    )
    aliases: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for key in ("id", "graph_node_id", "node_id", "Unnamed: 0", "n_id", "row_id"):
            if node.get(key) not in (None, ""):
                aliases[str(node[key]).replace(".0", "")] = index
    src: list[int] = []
    dst: list[int] = []
    relations: list[str] = []
    times: list[float] = []
    for edge in edges:
        source = aliases.get(_edge_endpoint(edge, True))
        target = aliases.get(_edge_endpoint(edge, False))
        if source is None or target is None:
            raise ValueError("edge endpoint is absent from node table")
        relation = canonical_relation(
            edge.get("relation") or edge.get("type") or edge.get("label") or edge.get("syscall")
        )
        src.append(source)
        dst.append(target)
        relations.append(relation)
        times.append(float(edge.get("time") or edge.get("timestamp") or 0.0))
    payload = {
        "node_ids": node_ids,
        "node_types": node_types,
        "attributes": [node_attribute_text(node) for node in nodes],
        "edges": list(zip(src, relations, dst, times)),
        "attribute_dim": attribute_dim,
        "node_vocab": NODE_TYPE_VOCAB,
        "relation_vocab": RELATION_VOCAB,
    }
    return GraphFeatures(
        node_ids=node_ids,
        node_types=node_types,
        attributes=deterministic_attribute_features(nodes, attribute_dim),
        type_ids=torch.tensor([NODE_TYPE_TO_ID[value] for value in node_types], dtype=torch.long),
        edge_index=torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long),
        relation_ids=torch.tensor([RELATION_TO_ID[value] for value in relations], dtype=torch.long),
        relation_names=tuple(relations),
        times=torch.tensor(times, dtype=torch.float32),
        feature_sha256=canonical_sha256(payload),
    )


class RGATLayer(nn.Module):
    def __init__(self, hidden_dim: int, relation_count: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_key = nn.Embedding(relation_count * 2, hidden_dim)
        self.relation_value = nn.Embedding(relation_count * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return nodes
        count = nodes.shape[0]
        aggregate = nodes.new_zeros((count, nodes.shape[1]))
        for direction in (0, 1):
            source = edge_index[direction]
            target = edge_index[1 - direction]
            directed_relation = relation_ids + direction * len(RELATION_VOCAB)
            key = self.key(nodes[source]) + self.relation_key(directed_relation)
            value = self.value(nodes[source]) + self.relation_value(directed_relation)
            logits = (self.query(nodes[target]) * key).sum(dim=1) / (nodes.shape[1] ** 0.5)
            for target_index in torch.unique(target):
                mask = target == target_index
                weights = torch.softmax(logits[mask], dim=0)
                aggregate[target_index] = aggregate[target_index] + (weights[:, None] * value[mask]).sum(dim=0)
        update = self.output(torch.cat([nodes, aggregate], dim=1))
        return self.norm(nodes + F.gelu(update))


class GCNLayer(nn.Module):
    """Relation-agnostic message passing used only by the encoder ablation."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        del relation_ids
        if edge_index.shape[1] == 0:
            return nodes
        count = nodes.shape[0]
        aggregate = nodes.new_zeros((count, nodes.shape[1]))
        degree = nodes.new_zeros((count, 1))
        for direction in (0, 1):
            source = edge_index[direction]
            target = edge_index[1 - direction]
            messages = self.message(nodes[source])
            aggregate.index_add_(0, target, messages)
            degree.index_add_(
                0,
                target,
                nodes.new_ones((target.shape[0], 1)),
            )
        aggregate = aggregate / degree.clamp_min(1.0)
        update = self.output(torch.cat([nodes, aggregate], dim=1))
        return self.norm(nodes + F.gelu(update))


class SharedRGATEncoder(nn.Module):
    """One parameter set encodes both query and candidate graphs."""

    def __init__(
        self,
        *,
        attribute_dim: int = 64,
        type_dim: int = 16,
        hidden_dim: int = 64,
        layers: int = 2,
    ):
        super().__init__()
        self.attribute_dim = attribute_dim
        self.type_embedding = nn.Embedding(len(NODE_TYPE_VOCAB), type_dim)
        self.input_projection = nn.Linear(attribute_dim + type_dim, hidden_dim)
        self.layers = nn.ModuleList(
            RGATLayer(hidden_dim, len(RELATION_VOCAB)) for _ in range(layers)
        )

    def forward(self, features: GraphFeatures) -> torch.Tensor:
        device = self.input_projection.weight.device
        attributes = features.attributes.to(device)
        type_ids = features.type_ids.to(device)
        hidden = self.input_projection(
            torch.cat([attributes, self.type_embedding(type_ids)], dim=1)
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                features.edge_index.to(device),
                features.relation_ids.to(device),
            )
        return hidden


class SharedGCNEncoder(nn.Module):
    """Shared relation-agnostic GCN backbone for ``w/o Relation Encoder``."""

    def __init__(
        self,
        *,
        attribute_dim: int = 64,
        type_dim: int = 16,
        hidden_dim: int = 64,
        layers: int = 2,
    ):
        super().__init__()
        self.attribute_dim = attribute_dim
        self.type_embedding = nn.Embedding(len(NODE_TYPE_VOCAB), type_dim)
        self.input_projection = nn.Linear(attribute_dim + type_dim, hidden_dim)
        self.layers = nn.ModuleList(GCNLayer(hidden_dim) for _ in range(layers))

    def forward(self, features: GraphFeatures) -> torch.Tensor:
        device = self.input_projection.weight.device
        attributes = features.attributes.to(device)
        type_ids = features.type_ids.to(device)
        hidden = self.input_projection(
            torch.cat([attributes, self.type_embedding(type_ids)], dim=1)
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                features.edge_index.to(device),
                features.relation_ids.to(device),
            )
        return hidden
