"""
Streamlit Application for Reflexion Agent with Streaming

A true Reflexion agent that generates an answer, reflects on it (identifying
what's missing/superfluous), searches for additional info, then revises
its answer — looping until satisfied or max iterations reached.

Flow: Respond → Execute Tools → Revise → (loop or END)
"""

import os
import json
import streamlit as st
from typing import List, Generator
from pydantic import BaseModel, Field

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph imports
from langgraph.graph import END, MessageGraph


# =============================================================================
# Configuration
# =============================================================================

TAVILY_API_KEY = os.getenv("Tavily_api_key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4.1-nano"
DEFAULT_MAX_ITERATIONS = 2


# =============================================================================
# Structured Output Models
# =============================================================================


class Reflection(BaseModel):
    missing: str = Field(description="What information is missing from the answer")
    superfluous: str = Field(description="What information is unnecessary")


class AnswerQuestion(BaseModel):
    answer: str = Field(description="Main response to the question (~250 words)")
    reflection: Reflection = Field(description="Self-critique of the answer")
    search_queries: List[str] = Field(description="1-3 queries for additional research")


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer using new information."""
    references: List[str] = Field(description="Citations supporting your updated answer")


# =============================================================================
# Initialize Components
# =============================================================================


@st.cache_resource
def initialize_components():
    tavily_tool = TavilySearchResults(max_results=3, tavily_api_key=TAVILY_API_KEY)
    llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.7)
    return llm, tavily_tool


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """You are a helpful research assistant that provides thorough, well-researched answers.

Your response must follow these steps:
1. {first_instruction}
2. Present a clear, well-structured answer with evidence and reasoning.
3. Reflect and critique your own answer — identify what's missing and what's unnecessary.
4. After the reflection, list 1-3 search queries for finding additional information to improve your answer.

Focus on accuracy, completeness, and citing sources when available."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Answer the user's question using the required structured format."),
])


# =============================================================================
# Build Chains
# =============================================================================


def get_initial_chain():
    llm, _ = initialize_components()
    first_responder_prompt = prompt_template.partial(
        first_instruction="Provide a detailed ~250 word answer"
    )
    return first_responder_prompt | llm.bind_tools(tools=[AnswerQuestion])


def get_revisor_chain():
    llm, _ = initialize_components()
    revise_instructions = (
        "Revise your previous answer using the new information.\n"
        "- Use the previous critique to add missing information and remove unnecessary content.\n"
        "- Include numerical citations to support your claims.\n"
        "- Add a References section at the bottom with URLs from search results.\n"
        "- Keep your response under 250 words, prioritizing precision over volume."
    )
    revisor_prompt = prompt_template.partial(first_instruction=revise_instructions)
    return revisor_prompt | llm.bind_tools(tools=[ReviseAnswer])


# =============================================================================
# Graph Nodes
# =============================================================================


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    """Execute search queries from the last AI message's tool calls."""
    _, tavily_tool = initialize_components()

    last_ai_message = state[-1]
    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])
            query_results = {}
            for query in search_queries:
                result = tavily_tool.invoke(query)
                query_results[query] = result
            tool_messages.append(
                ToolMessage(content=json.dumps(query_results), tool_call_id=call_id)
            )

    return tool_messages


def event_loop(state: List[BaseMessage]) -> str:
    """Decide whether to continue revising or stop."""
    max_iter = st.session_state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits >= max_iter:
        return END
    return "execute_tools"


# =============================================================================
# Build the Reflexion Graph
# =============================================================================


def create_reflexion_graph():
    """
    Create the Reflexion agent graph.

    Flow: respond → execute_tools → revise → (loop back or END)
    """
    initial_chain = get_initial_chain()
    revisor_chain = get_revisor_chain()

    graph = MessageGraph()

    graph.add_node("respond", initial_chain)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("revise", revisor_chain)

    graph.set_entry_point("respond")
    graph.add_edge("respond", "execute_tools")
    graph.add_edge("execute_tools", "revise")
    graph.add_conditional_edges("revise", event_loop)

    return graph.compile()


# =============================================================================
# Streaming Agent Run
# =============================================================================


def run_reflexion_streaming(query: str) -> Generator[tuple, None, None]:
    """
    Run the Reflexion agent and yield progress steps.

    Yields:
        (step_type, content) tuples for UI display
    """
    app = create_reflexion_graph()

    try:
        yield ("status", "Generating initial answer...")

        events = list(app.stream([HumanMessage(content=query)], stream_mode="updates"))

        revision_count = 0

        for event in events:
            # Initial respond node
            if "respond" in event:
                msg = event["respond"]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tc = msg.tool_calls[0]
                    args = tc.get("args", {})

                    answer = args.get("answer", "")
                    if answer:
                        yield ("initial_answer", answer)

                    reflection = args.get("reflection", {})
                    if reflection:
                        missing = reflection.get("missing", "")
                        superfluous = reflection.get("superfluous", "")
                        yield ("reflection", {"missing": missing, "superfluous": superfluous})

                    queries = args.get("search_queries", [])
                    if queries:
                        yield ("search_queries", queries)

            # Tool execution node
            if "execute_tools" in event:
                msgs = event["execute_tools"]
                if isinstance(msgs, list):
                    for msg in msgs:
                        if isinstance(msg, ToolMessage):
                            yield ("tool_results", msg.content[:500])
                elif isinstance(msgs, ToolMessage):
                    yield ("tool_results", msgs.content[:500])

            # Revise node
            if "revise" in event:
                revision_count += 1
                msg = event["revise"]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tc = msg.tool_calls[0]
                    args = tc.get("args", {})

                    answer = args.get("answer", "")
                    if answer:
                        yield ("revised_answer", {"answer": answer, "revision": revision_count})

                    reflection = args.get("reflection", {})
                    if reflection:
                        missing = reflection.get("missing", "")
                        superfluous = reflection.get("superfluous", "")
                        yield ("reflection", {"missing": missing, "superfluous": superfluous})

                    queries = args.get("search_queries", [])
                    if queries:
                        yield ("search_queries", queries)

                    references = args.get("references", [])
                    if references:
                        yield ("references", references)

        # Get final answer from the full invocation
        final_result = app.invoke([HumanMessage(content=query)])
        final_answer = ""
        final_references = []

        # Walk backwards to find the last revised answer
        for msg in reversed(final_result):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "ReviseAnswer":
                        final_answer = tc["args"].get("answer", "")
                        final_references = tc["args"].get("references", [])
                        break
                    elif tc["name"] == "AnswerQuestion" and not final_answer:
                        final_answer = tc["args"].get("answer", "")
                if final_answer:
                    break

        if final_answer:
            yield ("final_answer", {"answer": final_answer, "references": final_references})

        yield ("done", revision_count)

    except Exception as e:
        yield ("error", f"Error: {str(e)}")


# =============================================================================
# Streamlit UI
# =============================================================================


def main():
    st.set_page_config(page_title="Reflexion Agent", page_icon="🔄", layout="wide")

    st.markdown(
        """
    <style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
    .step-status {
        background-color: rgba(100, 100, 255, 0.15);
        color: #e0e0e0;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #6464ff;
    }
    .step-initial {
        background-color: rgba(78, 205, 196, 0.15);
        color: #e0e0e0;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #4ecdc4;
    }
    .step-reflection {
        background-color: rgba(255, 107, 107, 0.15);
        color: #e0e0e0;
        padding: 12px;
        border-radius: 5px;
        border-left: 3px solid #ff6b6b;
    }
    .step-search {
        background-color: rgba(255, 165, 2, 0.15);
        color: #e0e0e0;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #ffa502;
    }
    .step-revision {
        background-color: rgba(155, 89, 182, 0.15);
        color: #e0e0e0;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #9b59b6;
    }
    .final-answer {
        background-color: rgba(40, 167, 69, 0.15);
        color: #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #28a745;
        font-size: 16px;
        line-height: 1.6;
    }
    .references {
        background-color: rgba(52, 152, 219, 0.15);
        color: #e0e0e0;
        padding: 12px;
        border-radius: 5px;
        border-left: 3px solid #3498db;
        font-size: 14px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🔄 Reflexion Agent")
    st.markdown("""
    A **Reflexion** agent that generates an answer, **critiques itself**, searches for
    better information, then **revises** its answer — repeating until satisfied.
    """)

    st.divider()

    # Input
    st.subheader("📝 Ask a Question")
    user_question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the health benefits and risks of intermittent fasting?",
        key="question_input",
    )

    run_button = st.button("🚀 Run Agent", type="primary", disabled=not user_question)

    st.divider()

    # Results
    if run_button and user_question:
        st.subheader("📊 Agent Progress")

        progress_container = st.container()
        final_answer_text = ""
        final_references = []
        total_revisions = 0

        with progress_container:
            for step_type, content in run_reflexion_streaming(user_question):

                if step_type == "status":
                    st.markdown(
                        f'<div class="step-status">⏳ {content}</div>',
                        unsafe_allow_html=True,
                    )

                elif step_type == "initial_answer":
                    st.markdown("**📝 Initial Draft:**")
                    st.markdown(
                        f'<div class="step-initial">{content}</div>',
                        unsafe_allow_html=True,
                    )

                elif step_type == "reflection":
                    st.markdown("**🪞 Self-Reflection:**")
                    missing = content.get("missing", "N/A")
                    superfluous = content.get("superfluous", "N/A")
                    st.markdown(
                        f'<div class="step-reflection">'
                        f"<b>Missing:</b> {missing}<br>"
                        f"<b>Superfluous:</b> {superfluous}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                elif step_type == "search_queries":
                    queries_str = ", ".join(
                        [f'"{q}"' for q in content]
                    )
                    st.markdown(
                        f'<div class="step-search">🔍 Searching for: {queries_str}</div>',
                        unsafe_allow_html=True,
                    )

                elif step_type == "tool_results":
                    with st.expander("📄 View Search Results"):
                        st.text(content)

                elif step_type == "revised_answer":
                    rev_num = content.get("revision", 1)
                    answer = content.get("answer", "")
                    st.markdown(f"**✏️ Revision {rev_num}:**")
                    st.markdown(
                        f'<div class="step-revision">{answer}</div>',
                        unsafe_allow_html=True,
                    )

                elif step_type == "references":
                    refs_html = "<br>".join(content)
                    st.markdown(
                        f'<div class="references"><b>📚 References:</b><br>{refs_html}</div>',
                        unsafe_allow_html=True,
                    )

                elif step_type == "final_answer":
                    final_answer_text = content.get("answer", "")
                    final_references = content.get("references", [])

                elif step_type == "done":
                    total_revisions = content

                elif step_type == "error":
                    st.error(content)

        st.divider()

        # Final answer
        if final_answer_text:
            st.subheader("✅ Final Answer")
            st.markdown(
                f'<div class="final-answer">{final_answer_text}</div>',
                unsafe_allow_html=True,
            )

            if final_references:
                refs_html = "<br>".join(
                    [f"[{i+1}] {ref}" for i, ref in enumerate(final_references)]
                )
                st.markdown(
                    f'<div class="references"><b>📚 References:</b><br>{refs_html}</div>',
                    unsafe_allow_html=True,
                )

        if total_revisions > 0:
            st.info(f"📈 Total revision cycles: **{total_revisions}**")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Reflexion Agent**

        This agent uses the **Reflexion** pattern:

        1. **Respond**: Generate an initial answer with self-reflection
        2. **Search**: Use search queries from the reflection to find better info
        3. **Revise**: Improve the answer using new evidence + self-critique
        4. **Loop**: Repeat until max iterations reached

        Unlike a simple ReAct agent, this agent **critiques its own output**
        and iteratively improves it.
        """)

        st.header("⚙️ Settings")
        st.slider(
            "Max Revision Cycles",
            min_value=1,
            max_value=5,
            value=DEFAULT_MAX_ITERATIONS,
            key="max_iterations",
            help="How many times the agent will revise its answer",
        )
        st.markdown(f"""
        - **Model**: {MODEL_NAME}
        - **Tavily API**: {"✓ Loaded" if TAVILY_API_KEY else "✗ Missing"}
        - **OpenAI API**: {"✓ Loaded" if OPENAI_API_KEY else "✗ Missing"}
        """)


if __name__ == "__main__":
    main()
