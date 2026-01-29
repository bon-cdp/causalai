"""
CausalAI Streamlit Web Application.

A visual interface for causal AI conversations with DAG visualization.
100% FREE - No auth required. Donation-based support model.
"""

import streamlit as st
import json
from uuid import UUID

from causalai.core.dag import ConversationDAG
from causalai.core.nodes import (
    ConversationNode,
    NodeType,
    create_user_message,
    create_system_prompt,
)
from causalai.core.edges import create_causal_edge
from causalai.generation.qwen import QwenProvider, create_qwen_provider
from causalai.generation.providers import GenerationConfig


# --- Session State Helpers ---
def init_session():
    """Initialize session state."""
    if "dag" not in st.session_state:
        st.session_state.dag = None
    if "provider" not in st.session_state:
        st.session_state.provider = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


def get_api_key() -> str | None:
    """Get API key from Streamlit secrets or session state."""
    # First try Streamlit secrets (for deployment)
    try:
        return st.secrets.get("DASHSCOPE_API_KEY")
    except Exception:
        pass
    # Fall back to session state (for local dev with manual entry)
    return st.session_state.get("api_key")


# --- DAG Visualization ---
def render_dag_html(dag: ConversationDAG) -> str:
    """Render DAG as interactive HTML using vis.js."""
    nodes = []
    edges = []

    # Color scheme for node types
    colors = {
        NodeType.SYSTEM_PROMPT: "#6c757d",
        NodeType.USER_MESSAGE: "#007bff",
        NodeType.ASSISTANT_RESPONSE: "#28a745",
        NodeType.INTERVENTION: "#dc3545",
        NodeType.FORK_POINT: "#ffc107",
        NodeType.TOOL_CALL: "#17a2b8",
    }

    for node in dag.get_all_nodes():
        label = node.content[:50] + "..." if len(node.content) > 50 else node.content
        label = label.replace("\n", " ")
        nodes.append({
            "id": str(node.id),
            "label": label,
            "color": colors.get(node.node_type, "#6c757d"),
            "title": f"{node.node_type.value}\\n{node.content}",
            "shape": "box" if node.node_type == NodeType.USER_MESSAGE else "ellipse",
        })

    for node_id in dag._graph.nodes():
        for child_id in dag._graph.successors(node_id):
            edges.append({
                "from": str(node_id),
                "to": str(child_id),
                "arrows": "to",
            })

    html = f"""
    <html>
    <head>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            #graph {{
                width: 100%;
                height: 400px;
                border: 1px solid #ddd;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <div id="graph"></div>
        <script>
            var nodes = new vis.DataSet({json.dumps(nodes)});
            var edges = new vis.DataSet({json.dumps(edges)});
            var container = document.getElementById('graph');
            var data = {{ nodes: nodes, edges: edges }};
            var options = {{
                layout: {{
                    hierarchical: {{
                        direction: 'UD',
                        sortMethod: 'directed',
                        levelSeparation: 80,
                    }}
                }},
                physics: false,
                nodes: {{
                    font: {{ size: 12 }},
                    margin: 10,
                }},
                edges: {{
                    smooth: {{ type: 'cubicBezier' }},
                    color: '#666',
                }},
            }};
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    return html


# --- Donation Footer ---
def render_donation_footer():
    """Render donation and contact information."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h4 style="margin: 0 0 0.5rem 0;">Support Our Research</h4>
        <p style="margin: 0.5rem 0; font-size: 0.9rem;">
            CausalAI is 100% free. If you find it useful, consider supporting us:
        </p>
        <p style="margin: 0.5rem 0;">
            <strong>PayPal:</strong> leejeonghyeok2012@gmail.com
        </p>
        <p style="margin: 0.5rem 0;">
            <strong>Consulting:</strong> sfj416@gmail.com
        </p>
    </div>
    """, unsafe_allow_html=True)


# --- Main App UI ---
def render_app():
    """Render main application."""
    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")

        # Check if API key is in secrets
        api_key = get_api_key()

        if not api_key:
            # Show API key input only if not in secrets
            api_key_input = st.text_input(
                "DashScope API Key",
                type="password",
                value="",
                help="Your Alibaba Model Studio API key (or set in secrets.toml)",
            )
            if api_key_input:
                st.session_state.api_key = api_key_input
                api_key = api_key_input

        if api_key:
            try:
                st.session_state.provider = create_qwen_provider(api_key)
                st.success("API configured")
            except Exception as e:
                st.error(f"Invalid API key: {e}")
        else:
            st.info("API key configured via secrets")
            if get_api_key():
                try:
                    st.session_state.provider = create_qwen_provider(get_api_key())
                except Exception:
                    pass

        st.markdown("---")

        # Model selection
        model = st.selectbox(
            "Model",
            ["qwen-plus", "qwen-turbo", "qwen-max"],
            index=0,
        )
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

        st.markdown("---")

        if st.button("Clear Conversation"):
            st.session_state.dag = ConversationDAG()
            st.session_state.messages = []
            st.rerun()

        # Support info in sidebar
        st.markdown("---")
        st.markdown("""
        **Support CausalAI**
        - PayPal: `leejeonghyeok2012@gmail.com`
        - Consulting: `sfj416@gmail.com`
        """)

    # Main content
    st.title("CausalAI Chat")
    st.markdown("*Pearl's Causal Inference for AI Conversations - 100% Free*")

    # Initialize DAG if needed
    if st.session_state.dag is None:
        st.session_state.dag = ConversationDAG()
        # Add system prompt
        system_node = create_system_prompt(
            "You are a helpful AI assistant. Be concise and accurate."
        )
        st.session_state.dag.add_node(system_node)

    # DAG Visualization
    with st.expander("Conversation Graph", expanded=True):
        if st.session_state.dag.node_count > 0:
            html = render_dag_html(st.session_state.dag)
            st.components.v1.html(html, height=420)
        else:
            st.info("Start a conversation to see the causal graph")

    # Chat interface
    st.markdown("---")

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Type your message..."):
        # Check API key
        if not st.session_state.provider:
            st.error("Please configure your DashScope API key in the sidebar or secrets.toml")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Create user node
        user_node = create_user_message(prompt)
        st.session_state.dag.add_node(user_node)

        # Connect to previous node
        nodes = list(st.session_state.dag._graph.nodes())
        if len(nodes) > 1:
            prev_node = nodes[-2]
            st.session_state.dag.add_edge(
                create_causal_edge(prev_node, user_node.id)
            )

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    config = GenerationConfig(
                        model=model,
                        temperature=temperature,
                    )

                    # Build messages for API
                    api_messages = [
                        {"role": "system", "content": "You are a helpful AI assistant."}
                    ]
                    for msg in st.session_state.messages:
                        api_messages.append({
                            "role": msg["role"],
                            "content": msg["content"],
                        })

                    result = st.session_state.provider.generate(api_messages, config)
                    response = result.content

                    st.markdown(response)

                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                    })

                    # Create response node
                    response_node = ConversationNode(
                        node_type=NodeType.ASSISTANT_RESPONSE,
                        content=response,
                    )
                    st.session_state.dag.add_node(response_node)
                    st.session_state.dag.add_edge(
                        create_causal_edge(user_node.id, response_node.id)
                    )

                    # Rerun to update graph
                    st.rerun()

                except Exception as e:
                    st.error(f"Error generating response: {e}")

    # Donation footer
    render_donation_footer()


def main():
    """Main entry point."""
    st.set_page_config(
        page_title="CausalAI",
        page_icon="",
        layout="wide",
    )

    init_session()
    render_app()


if __name__ == "__main__":
    main()
