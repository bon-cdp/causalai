# CausalAI

Pearl's Causal Inference Framework for AI Conversations.

A meta-control system that models AI conversations as causal DAGs, applying do-calculus rules to enable interventions, forks, and alignment testing at each node.

**100% FREE** - No auth, no limits. Donation-based support model.

## Quick Start - Web UI

```bash
# Clone and setup
cd causalai
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[ui]"

# Run the web app
streamlit run src/causalai/ui/app.py
```

Then open http://localhost:8501 in your browser.

## Configuration

### Option 1: Streamlit Secrets (Recommended)

Create `.streamlit/secrets.toml`:
```toml
DASHSCOPE_API_KEY = "sk-your-key-here"
```

### Option 2: Environment Variable

```bash
export DASHSCOPE_API_KEY=sk-your-key-here
```

### Option 3: Enter in App

Enter your API key directly in the sidebar when running the app.

Get an API key from [Alibaba Model Studio](https://www.alibabacloud.com/help/en/model-studio/get-api-key).

## Deployment

### Streamlit Community Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `src/causalai/ui/app.py` as the main file
5. Add your `DASHSCOPE_API_KEY` in the Secrets section
6. Deploy!

Your app will be available at `https://your-app.streamlit.app`

### Firebase Hosting (Alternative)

For Firebase deployment with Cloud Run:

1. Create `firebase.json`:
```json
{
  "hosting": {
    "public": "public",
    "rewrites": [{
      "source": "**",
      "run": {
        "serviceId": "causalai",
        "region": "us-central1"
      }
    }]
  }
}
```

2. Create a Dockerfile and deploy to Cloud Run
3. Configure Firebase Hosting to proxy to Cloud Run

## Installation

```bash
pip install causalai
```

For development:

```bash
pip install causalai[dev]
```

For all features:

```bash
pip install causalai[all]
```

## Python API

```python
from causalai.core import ConversationDAG, ConversationNode, NodeType
from causalai.core.edges import create_causal_edge
from causalai.causal import DSeparationAnalyzer, DoCalculusEngine
from causalai.generation import QwenProvider, GenerationConfig

# Create a conversation DAG
dag = ConversationDAG()

# Add nodes
user_msg = ConversationNode(
    node_type=NodeType.USER_MESSAGE,
    content="What is causal inference?"
)
response = ConversationNode(
    node_type=NodeType.ASSISTANT_RESPONSE,
    content="Causal inference is..."
)

dag.add_node(user_msg)
dag.add_node(response)
dag.add_edge(create_causal_edge(user_msg.id, response.id))

# Analyze causal structure
analyzer = DSeparationAnalyzer(dag)
engine = DoCalculusEngine(dag)

# Use Qwen for generation
provider = QwenProvider(api_key="sk-xxx")
result = provider.generate(
    messages=[{"role": "user", "content": "Hello!"}],
    config=GenerationConfig(model="qwen-plus"),
)
print(result.content)
```

## Features

- **Web UI**: Interactive Streamlit app with DAG visualization
- **Conversation as Causal DAG**: Model conversations as directed acyclic graphs
- **Do-Calculus**: Pearl's three rules for causal inference
- **D-Separation**: Test conditional independence in causal graphs
- **Graph Surgery**: Mutilate graphs for intervention analysis
- **Information Theory**: Causal entropy, mutual information, transfer entropy
- **Qwen Integration**: Alibaba's Qwen models via DashScope

## Roadmap

- [ ] Parallel output generation (run N candidates, select best)
- [ ] Node forking and branching
- [ ] Alignment testing at each node
- [ ] Newton physics grounding
- [ ] REST API for programmatic access

## Support Our Research

CausalAI is 100% free and open source. If you find it useful, consider supporting us:

- **PayPal**: leejeonghyeok2012@gmail.com
- **Consulting**: sfj416@gmail.com

## License

MIT
