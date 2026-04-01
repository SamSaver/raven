"""Agentic RAG pipeline using LangGraph.

Implements the Plan-Route-Act-Verify-Stop loop for multi-hop question answering.
The agent decomposes complex queries into sub-goals, selects appropriate tools,
gathers evidence, and synthesizes a final answer.
"""

from typing import Annotated, TypedDict

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from backend.agents.tools import ALL_TOOLS
from backend.config import settings

logger = structlog.get_logger()

MAX_ITERATIONS = 8


class AgentState(TypedDict):
    """State passed between nodes in the agent graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int
    final_answer: str


AGENT_SYSTEM_PROMPT = """You are Raven, an intelligent research assistant with access to tools.

Your task is to answer the user's question thoroughly and accurately by:
1. Breaking down complex questions into sub-questions if needed
2. Using the appropriate tools to gather evidence
3. Synthesizing evidence into a comprehensive answer with citations

Available tools:
- vector_search: Search the document knowledge base for relevant information
- web_search: Search the web for current/external information
- calculator: Evaluate mathematical expressions
- summarize_evidence: Synthesize multiple pieces of evidence into a coherent answer

Strategy:
- For simple factual questions: use vector_search directly
- For questions requiring current info: use web_search
- For multi-hop questions: break into parts, search for each, then synthesize
- For calculations: use the calculator tool
- Always cite your sources

When you have gathered sufficient evidence, provide your final answer directly.
Do NOT call tools if you already have enough information to answer."""


def _build_agent_graph():
    """Build the LangGraph agent with Plan-Route-Act-Verify-Stop loop."""

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_host,
        temperature=0.3,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    tool_node = ToolNode(ALL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        """The reasoning node: decides what to do next."""
        messages = state["messages"]
        iteration = state.get("iteration", 0)

        # Add iteration tracking to prevent infinite loops
        if iteration >= MAX_ITERATIONS:
            return {
                "messages": [AIMessage(content=(
                    "I've reached my research limit. Let me summarize what I've found so far "
                    "based on the evidence gathered."
                ))],
                "iteration": iteration + 1,
            }

        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "iteration": iteration + 1}

    def should_continue(state: AgentState) -> str:
        """Decide whether to continue tool use or stop."""
        messages = state["messages"]
        last_message = messages[-1]
        iteration = state.get("iteration", 0)

        # Stop if max iterations reached
        if iteration >= MAX_ITERATIONS:
            return "end"

        # Continue if there are tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # Otherwise, the agent has produced a final answer
        return "end"

    def format_output(state: AgentState) -> dict:
        """Extract the final answer from the conversation."""
        messages = state["messages"]
        # Find the last AI message that isn't a tool call
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                return {"final_answer": msg.content}
        return {"final_answer": "I was unable to find a satisfactory answer."}

    # Build the graph
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("output", format_output)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": "output"},
    )

    graph.add_edge("tools", "agent")
    graph.add_edge("output", END)

    return graph.compile()


# Lazy-initialized compiled graph
_compiled_graph = None


def get_agent():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_agent_graph()
    return _compiled_graph


def run_agent(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> dict:
    """Run the agentic RAG pipeline on a query.

    Returns:
        dict with keys: answer, messages (full trace), iterations
    """
    agent = get_agent()

    messages: list[BaseMessage] = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]

    if history:
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=query))

    initial_state: AgentState = {
        "messages": messages,
        "iteration": 0,
        "final_answer": "",
    }

    logger.info("agent.start", query=query[:100])

    result = agent.invoke(initial_state)

    final_answer = result.get("final_answer", "")
    iterations = result.get("iteration", 0)

    # Extract tool usage trace
    tool_trace = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_trace.append({
                    "tool": tc["name"],
                    "input": str(tc["args"])[:200],
                })
        elif isinstance(msg, ToolMessage):
            tool_trace.append({
                "tool_response": msg.content[:300],
            })

    logger.info("agent.done", iterations=iterations, tools_used=len(tool_trace))

    return {
        "answer": final_answer,
        "iterations": iterations,
        "tool_trace": tool_trace,
    }


def run_agent_stream(
    query: str,
    history: list[dict[str, str]] | None = None,
):
    """Run the agentic RAG pipeline with streaming updates.

    Yields dicts with step-by-step progress:
      {"type": "thinking", "content": "..."}
      {"type": "tool_call", "tool": "...", "input": "..."}
      {"type": "tool_result", "content": "..."}
      {"type": "answer", "content": "..."}
    """
    agent = get_agent()

    messages: list[BaseMessage] = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]

    if history:
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=query))

    initial_state: AgentState = {
        "messages": messages,
        "iteration": 0,
        "final_answer": "",
    }

    for event in agent.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "agent":
                new_messages = node_output.get("messages", [])
                for msg in new_messages:
                    if isinstance(msg, AIMessage):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield {
                                    "type": "tool_call",
                                    "tool": tc["name"],
                                    "input": str(tc["args"])[:200],
                                }
                        elif msg.content:
                            yield {"type": "thinking", "content": msg.content[:500]}

            elif node_name == "tools":
                new_messages = node_output.get("messages", [])
                for msg in new_messages:
                    if isinstance(msg, ToolMessage):
                        yield {
                            "type": "tool_result",
                            "content": msg.content[:500],
                        }

            elif node_name == "output":
                answer = node_output.get("final_answer", "")
                if answer:
                    yield {"type": "answer", "content": answer}
