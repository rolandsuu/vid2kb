from __future__ import annotations

"""Retrieval probe questions for the eval harness.

Each entry is a plain question plus a list of expected keywords. The harness
asks every question against the knowledge base, takes the top-3 chunks, and
scores a lexical hit if any keyword appears in the combined chunk text.
Keywords are chosen to be content-grounded for the sample Chinese tutorial
videos (AI-agent style tutorials); adjust them to match your own corpus.
"""

PROBE_QUESTIONS: list[dict] = [
    {
        'question': '什么是智能体？',
        'keywords': ['智能体', 'Agent', '自主'],
    },
    {
        'question': '智能体如何理解用户意图？',
        'keywords': ['意图', '大语言模型', '理解'],
    },
    {
        'question': '智能体如何调用工具执行操作？',
        'keywords': ['工具', '调用', 'API'],
    },
    {
        'question': '智能体有哪些应用场景？',
        'keywords': ['应用', '场景', '个人助理', '客户服务'],
    },
    {
        'question': '这个视频教程的主要内容是什么？',
        'keywords': ['教程', '介绍', '总结'],
    },
]
