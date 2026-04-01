"""Agent tools for the agentic RAG pipeline.

Each tool is a callable that takes a query/input and returns structured evidence.
Tools are registered with LangGraph via the ToolNode mechanism.
"""

import ast
import operator

import structlog
from langchain_core.tools import tool

from backend.config import settings
from backend.ingestion.embedder import embed_query
from backend.retrieval.hybrid import HybridSearcher
from backend.retrieval.reranker import rerank
from backend.storage.vector import VectorStore

logger = structlog.get_logger()


@tool
def vector_search(query: str) -> str:
    """Search the document knowledge base for information relevant to the query.
    Use this when you need to find facts, definitions, explanations, or data
    from the ingested documents."""
    searcher = HybridSearcher()
    results = searcher.search(
        query=query,
        top_k=5,
        hybrid_weight=settings.default_hybrid_weight,
    )

    if settings.reranker_enabled and results:
        results = rerank(query, results, top_k=3)

    if not results:
        return "No relevant documents found for this query."

    output_parts = []
    for i, r in enumerate(results, 1):
        source = r.get("source", "unknown")
        page = r.get("page_number", "?")
        content = r.get("content", "")[:500]
        score = r.get("rerank_score", r.get("score", r.get("rrf_score", 0.0)))
        output_parts.append(
            f"[Source {i}: {source}, Page {page}, Score: {score:.3f}]\n{content}"
        )

    return "\n\n---\n\n".join(output_parts)


@tool
def web_search(query: str) -> str:
    """Search the web for current information not available in the document store.
    Use this when the document store lacks sufficient information, or when the
    question requires up-to-date knowledge."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No web search results found."

        output_parts = []
        for i, r in enumerate(results, 1):
            output_parts.append(
                f"[Web Result {i}: {r.get('title', 'Untitled')}]\n"
                f"URL: {r.get('href', 'N/A')}\n"
                f"{r.get('body', 'No snippet available.')}"
            )

        return "\n\n---\n\n".join(output_parts)

    except Exception as e:
        logger.warning("tool.web_search_failed", error=str(e))
        return f"Web search failed: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic (+, -, *, /),
    exponentiation (**), and common math functions. Input should be a valid
    Python math expression like '2 + 3 * 4' or '(100 / 5) ** 2'."""
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return allowed_operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            operand = _eval_node(node.operand)
            return allowed_operators[op_type](operand)
        else:
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def summarize_evidence(evidence_texts: str) -> str:
    """Summarize and synthesize multiple pieces of evidence into a coherent answer.
    Input should be the combined evidence text from previous tool calls.
    Use this as a final step to produce a well-structured answer."""
    from backend.generation.llm import generate

    prompt = (
        "Synthesize the following evidence into a clear, coherent summary. "
        "Preserve key facts and cite sources where noted.\n\n"
        f"Evidence:\n{evidence_texts}"
    )
    return generate(prompt, temperature=0.3, max_tokens=1024)


# Registry of all available tools
ALL_TOOLS = [vector_search, web_search, calculator, summarize_evidence]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}
