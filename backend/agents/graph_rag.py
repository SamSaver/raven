"""GraphRAG module using NetworkX for knowledge graph construction and traversal.

Extracts entities and relationships from documents using Ollama, builds a
knowledge graph, and supports relationship-based retrieval as a complement
to vector search.
"""

import json
import pickle
from pathlib import Path

import networkx as nx
import structlog

from backend.config import settings
from backend.generation.llm import generate

logger = structlog.get_logger()

GRAPH_PATH = Path("data/knowledge_graph.pkl")

EXTRACTION_PROMPT = """Extract entities and relationships from the following text.
Return a JSON object with two arrays:
- "entities": list of objects with "name" (string) and "type" (string, e.g. "person", "organization", "concept", "location", "event", "technology")
- "relationships": list of objects with "source" (entity name), "target" (entity name), "relation" (string describing the relationship)

Only extract clear, factual relationships. Keep entity names concise and consistent.

Text:
{text}

Return ONLY valid JSON, no explanation."""

COMMUNITY_SUMMARY_PROMPT = """Summarize the following group of related entities and their relationships
into a concise paragraph. Focus on the key themes and connections.

Entities and relationships:
{community_data}

Summary:"""


class KnowledgeGraph:
    def __init__(self, graph_path: Path | None = None):
        self.graph_path = graph_path or GRAPH_PATH
        self.graph = self._load_or_create()

    def _load_or_create(self) -> nx.DiGraph:
        if self.graph_path.exists():
            with open(self.graph_path, "rb") as f:
                graph = pickle.load(f)
            logger.info("graph_rag.loaded", nodes=graph.number_of_nodes(), edges=graph.number_of_edges())
            return graph
        return nx.DiGraph()

    def save(self) -> None:
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.graph, f)
        logger.info("graph_rag.saved", nodes=self.graph.number_of_nodes(), edges=self.graph.number_of_edges())

    def extract_and_add(self, text: str, doc_id: str, chunk_id: str) -> dict:
        """Extract entities and relationships from text and add to graph.

        Returns dict with counts of entities and relationships added.
        """
        prompt = EXTRACTION_PROMPT.format(text=text[:2000])

        try:
            response = generate(prompt, temperature=0.1, max_tokens=1024)
            # Try to parse JSON from response
            data = self._parse_json_response(response)
        except Exception as e:
            logger.warning("graph_rag.extraction_failed", error=str(e))
            return {"entities": 0, "relationships": 0}

        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        # Add entities as nodes
        for entity in entities:
            name = entity.get("name", "").strip().lower()
            entity_type = entity.get("type", "unknown")
            if not name:
                continue

            if self.graph.has_node(name):
                # Update existing node with additional source
                sources = self.graph.nodes[name].get("sources", [])
                sources.append({"doc_id": doc_id, "chunk_id": chunk_id})
                self.graph.nodes[name]["sources"] = sources
                # Increment mention count
                self.graph.nodes[name]["mentions"] = self.graph.nodes[name].get("mentions", 1) + 1
            else:
                self.graph.add_node(
                    name,
                    type=entity_type,
                    sources=[{"doc_id": doc_id, "chunk_id": chunk_id}],
                    mentions=1,
                )

        # Add relationships as edges
        for rel in relationships:
            source = rel.get("source", "").strip().lower()
            target = rel.get("target", "").strip().lower()
            relation = rel.get("relation", "related_to")

            if not source or not target:
                continue

            # Ensure nodes exist
            if not self.graph.has_node(source):
                self.graph.add_node(source, type="unknown", sources=[], mentions=1)
            if not self.graph.has_node(target):
                self.graph.add_node(target, type="unknown", sources=[], mentions=1)

            if self.graph.has_edge(source, target):
                # Add relation to existing edge
                existing = self.graph.edges[source, target].get("relations", [])
                existing.append(relation)
                self.graph.edges[source, target]["relations"] = existing
            else:
                self.graph.add_edge(
                    source, target,
                    relations=[relation],
                    doc_id=doc_id,
                )

        logger.info(
            "graph_rag.extracted",
            entities=len(entities),
            relationships=len(relationships),
        )

        return {"entities": len(entities), "relationships": len(relationships)}

    def _parse_json_response(self, response: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = response.strip()
        # Try to find JSON block
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        return json.loads(text)

    def query_neighbors(self, entity: str, max_hops: int = 2) -> dict:
        """Get entity and its neighborhood up to max_hops away."""
        entity = entity.strip().lower()

        if not self.graph.has_node(entity):
            # Try fuzzy match
            matches = [n for n in self.graph.nodes if entity in n or n in entity]
            if matches:
                entity = matches[0]
            else:
                return {"entity": entity, "found": False, "neighbors": []}

        # BFS traversal up to max_hops
        neighbors = []
        visited = {entity}
        current_level = [entity]

        for hop in range(max_hops):
            next_level = []
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
                        edge_data = self.graph.edges[node, neighbor]
                        neighbors.append({
                            "entity": neighbor,
                            "type": self.graph.nodes[neighbor].get("type", "unknown"),
                            "relation_from": node,
                            "relations": edge_data.get("relations", []),
                            "hop": hop + 1,
                        })

                # Also check predecessors (incoming edges)
                for predecessor in self.graph.predecessors(node):
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_level.append(predecessor)
                        edge_data = self.graph.edges[predecessor, node]
                        neighbors.append({
                            "entity": predecessor,
                            "type": self.graph.nodes[predecessor].get("type", "unknown"),
                            "relation_to": node,
                            "relations": edge_data.get("relations", []),
                            "hop": hop + 1,
                        })

            current_level = next_level

        return {
            "entity": entity,
            "found": True,
            "type": self.graph.nodes[entity].get("type", "unknown"),
            "mentions": self.graph.nodes[entity].get("mentions", 0),
            "neighbors": neighbors,
        }

    def find_path(self, source: str, target: str) -> list[dict] | None:
        """Find shortest path between two entities."""
        source = source.strip().lower()
        target = target.strip().lower()

        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return None

        try:
            path = nx.shortest_path(self.graph, source, target)
            result = []
            for i in range(len(path) - 1):
                edge = self.graph.edges[path[i], path[i + 1]]
                result.append({
                    "from": path[i],
                    "to": path[i + 1],
                    "relations": edge.get("relations", []),
                })
            return result
        except nx.NetworkXNoPath:
            return None

    def detect_communities(self) -> list[list[str]]:
        """Detect communities using connected components on undirected version."""
        undirected = self.graph.to_undirected()
        communities = list(nx.connected_components(undirected))
        # Sort by size, largest first
        communities = sorted(communities, key=len, reverse=True)
        return [list(c) for c in communities]

    def get_community_summaries(self, max_communities: int = 5) -> list[dict]:
        """Generate LLM summaries for the top communities."""
        communities = self.detect_communities()[:max_communities]
        summaries = []

        for i, community in enumerate(communities):
            if len(community) < 2:
                continue

            # Build community description
            parts = []
            subgraph = self.graph.subgraph(community)
            for u, v, data in subgraph.edges(data=True):
                relations = data.get("relations", ["related to"])
                parts.append(f"- {u} → {', '.join(relations)} → {v}")

            for node in community:
                node_type = self.graph.nodes[node].get("type", "unknown")
                parts.append(f"- Entity: {node} (type: {node_type})")

            community_data = "\n".join(parts[:30])  # Limit for LLM context

            try:
                summary = generate(
                    COMMUNITY_SUMMARY_PROMPT.format(community_data=community_data),
                    temperature=0.3,
                    max_tokens=300,
                )
                summaries.append({
                    "community_id": i,
                    "entities": community[:20],
                    "size": len(community),
                    "edges": subgraph.number_of_edges(),
                    "summary": summary,
                })
            except Exception as e:
                logger.warning("graph_rag.summary_failed", community=i, error=str(e))

        return summaries

    def graph_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the knowledge graph by finding relevant entities and their context.

        Combines entity matching with neighborhood traversal for graph-augmented retrieval.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Score nodes by relevance to query
        scored_nodes = []
        for node in self.graph.nodes:
            node_words = set(node.split())
            overlap = len(query_words & node_words)
            # Partial match: any query word appears in node name
            partial = any(w in node for w in query_words if len(w) > 2)
            if overlap > 0 or partial:
                score = overlap + (0.5 if partial and overlap == 0 else 0)
                scored_nodes.append((node, score))

        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        results = []
        seen_sources = set()
        for node, score in scored_nodes[:top_k]:
            info = self.query_neighbors(node, max_hops=1)
            # Collect source chunk_ids from the entity
            for source in self.graph.nodes[node].get("sources", []):
                chunk_id = source.get("chunk_id")
                if chunk_id and chunk_id not in seen_sources:
                    seen_sources.add(chunk_id)
                    results.append({
                        "entity": node,
                        "doc_id": source.get("doc_id", ""),
                        "chunk_id": chunk_id,
                        "graph_score": score,
                        "neighbor_count": len(info.get("neighbors", [])),
                    })

        return results[:top_k]

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "communities": len(self.detect_communities()),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
        }
