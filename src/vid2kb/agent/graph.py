from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from vid2kb.agent import tools
from vid2kb.agent.state import AgentState


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node('planner', tools.tool_planner)
    g.add_node('ingest', tools.tool_ingest)
    g.add_node('transcribe', tools.tool_transcribe)
    g.add_node('visual', tools.tool_visual)
    g.add_node('compose', tools.tool_compose)
    g.add_node('render', tools.tool_render)
    g.add_node('ingest_kb', tools.tool_ingest_kb)
    g.add_node('voiceover', tools.tool_voiceover)
    g.add_node('report', tools.tool_report)

    g.add_edge(START, 'planner')

    g.add_conditional_edges(
        'planner',
        lambda s: s.get('next', 'report'),
        {
            'ingest': 'ingest',
            'transcribe': 'transcribe',
            'visual': 'visual',
            'compose': 'compose',
            'render': 'render',
            'ingest_kb': 'ingest_kb',
            'voiceover': 'voiceover',
            'report': 'report',
            'ask_user': END,
        },
    )

    for node in ('ingest', 'transcribe', 'visual', 'compose', 'render', 'ingest_kb', 'voiceover'):
        g.add_edge(node, 'planner')
    g.add_edge('report', END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
