from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from vid2kb.agent.state import AgentState


def _audit(state: AgentState, stage: str, status: str, note: str = '', seconds: float = 0.0) -> None:
    try:
        from vid2kb.config import settings
        from vid2kb.llm import drain_usage

        run_id = state.get('run_id', 'unknown')
        path = Path(settings.data_dir) / 'runs' / run_id / 'audit.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'stage': stage,
            'status': status,
            'seconds': seconds,
            'note': note,
        }
        usage = drain_usage()
        if usage:
            record['tokens'] = usage
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        pass


def _state_summary(state: AgentState) -> dict:
    return {
        'video_path': bool(state.get('video_path')),
        'transcript': bool(state.get('transcript')),
        'timeline': bool(state.get('timeline')),
        'document': bool(state.get('document')),
        'markdown': bool(state.get('markdown')),
        'kb_doc_id': bool(state.get('kb_doc_id')),
    }


_ALLOWED_NEXT = {'ingest', 'transcribe', 'visual', 'compose', 'render', 'ingest_kb', 'voiceover', 'clips', 'report', 'ask_user'}


def tool_planner(state: AgentState) -> dict:
    from vid2kb.config import settings
    from vid2kb.llm import deepseek_client, record_usage

    iterations = state.get('iterations', 0)
    if iterations > 12:
        _audit(state, 'planner', 'loop_guard', 'max 12 iterations')
        return {'next': 'report', 'errors': state.get('errors', []) + ['loop guard: max 12 iterations']}

    started = time.time()
    system_prompt = (
        '你是一个视频知识智能体的规划器。根据当前状态选择下一个工具。'
        '可选工具：ingest（下载或拷贝视频）、transcribe（语音转写）、visual（画面分析）、'
        'compose（撰写文档）、render（渲染 markdown/pdf）、ingest_kb（写入知识库）、'
        'report（生成最终报告）、ask_user（询问用户）。'
        '规则：如果没有 video_path 则选 ingest；'
        '如果有 video_path 但没有 transcript 则选 transcribe；'
        '如果有 transcript 但没有 timeline 则选 visual；'
        '如果有 timeline 但没有 document 则选 compose；'
        '如果有 document 但没有 markdown 则选 render；'
        '如果有 markdown 但没有 kb_doc_id 则选 ingest_kb；'
        '如果用户要求配音则选 voiceover；'
        '否则选 report。'
        '如果某个阶段（ingest/transcribe/visual/compose/render/ingest_kb）'
        '在错误记录中已经失败 2 次及以上，则不要重试该阶段，直接选 report。'
        '必须输出一个 json 对象，字段为 next 和 reason。'
    )
    user_message = (
        'You must output a valid json object. Respond with json only.\n\n'
        f'user_prompt: {state.get("user_prompt", "")}\n\n'
        f'state summary: {json.dumps(_state_summary(state), ensure_ascii=False)}\n'
    )
    try:
        client = deepseek_client()
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
            ],
            response_format={'type': 'json_object'},
        )
        record_usage('deepseek', response)
        raw = response.choices[0].message.content
        data = json.loads(raw)
        next_tool = data.get('next', 'report')
        if next_tool not in _ALLOWED_NEXT:
            next_tool = 'report'
        if next_tool not in ('report', 'ask_user'):
            fail_count = sum(1 for e in state.get('errors', []) if next_tool in e and 'failed' in e)
            if fail_count >= 2:
                next_tool = 'report'
        _audit(state, 'planner', 'done', note=f'next={next_tool}', seconds=time.time() - started)
        return {'next': next_tool, 'iterations': iterations + 1}
    except Exception as e:
        _audit(state, 'planner', 'error', str(e), seconds=time.time() - started)
        return {'next': 'report', 'errors': state.get('errors', []) + [f'planner failed: {e}']}


def tool_ingest(state: AgentState) -> dict:
    from vid2kb.media.download import copy_local_file, download_video
    from vid2kb.media.store import ArtifactStore

    started = time.time()
    try:
        store = ArtifactStore(state['run_id'])
        src = state['source']
        if src.startswith('http'):
            path = download_video(src, store.raw)
        else:
            path = copy_local_file(Path(src), store.raw)
        _audit(state, 'ingest', 'done', note=str(path), seconds=time.time() - started)
        return {'video_path': str(path), 'steps': ['ingest: ok']}
    except Exception as e:
        _audit(state, 'ingest', 'error', str(e), seconds=time.time() - started)
        return {'errors': state.get('errors', []) + [f'ingest failed: {e}'], 'steps': [f'ingest: failed: {e}']}


def _transcribe_with_fallback(audio: Path, backend: str):
    order = ['funasr', 'whisper'] if backend != 'whisper' else ['whisper', 'funasr']
    errors: list[str] = []
    for name in order:
        try:
            if name == 'funasr':
                from vid2kb.asr.funasr_engine import FunASREngine

                engine = FunASREngine()
            else:
                from vid2kb.asr.whisper_engine import WhisperEngine

                engine = WhisperEngine()
            return engine.transcribe(audio), errors
        except Exception as e:
            errors.append(f'asr {name} failed: {e}')
    return None, errors


def tool_transcribe(state: AgentState) -> dict:
    from vid2kb.config import settings
    from vid2kb.media.ffmpeg import extract_audio, probe_duration
    from vid2kb.media.store import ArtifactStore

    started = time.time()
    try:
        store = ArtifactStore(state['run_id'])
        video = Path(state['video_path'])
        probe_duration(video)
        audio = extract_audio(video, store.audio / 'audio.wav')
        transcript, asr_errors = _transcribe_with_fallback(audio, settings.asr_backend)
        if transcript is None:
            _audit(state, 'transcribe', 'error', '; '.join(asr_errors), seconds=time.time() - started)
            return {
                'errors': state.get('errors', []) + [f'transcribe failed: {err}' for err in asr_errors],
                'steps': [f'transcribe: failed: {asr_errors[-1]}'],
            }
        _audit(state, 'transcribe', 'done', note=transcript.language, seconds=time.time() - started)
        return {
            'transcript': transcript.text,
            'transcript_language': transcript.language,
            'steps': ['transcribe: ok'],
        }
    except Exception as e:
        _audit(state, 'transcribe', 'error', str(e), seconds=time.time() - started)
        return {
            'errors': state.get('errors', []) + [f'transcribe failed: {e}'],
            'steps': [f'transcribe: failed: {e}'],
        }


def tool_visual(state: AgentState) -> dict:
    from vid2kb.config import settings
    from vid2kb.media.ffmpeg import sample_frames
    from vid2kb.media.store import ArtifactStore
    from vid2kb.vision.qwen_vl import analyze_frames

    started = time.time()
    try:
        store = ArtifactStore(state['run_id'])
        video = Path(state['video_path'])
        frames = sample_frames(video, store.frames, settings.frame_interval_seconds, settings.max_frames)
        if not frames:
            _audit(state, 'visual', 'error', 'no frames sampled', seconds=time.time() - started)
            return {
                'errors': state.get('errors', []) + ['no frames sampled'],
                'steps': ['visual: failed: no frames sampled'],
            }
        timeline = analyze_frames(frames, state.get('user_prompt', ''), (state.get('transcript') or '')[:4000])
        _audit(state, 'visual', 'done', note=str(len(timeline.frames)), seconds=time.time() - started)
        return {'timeline': timeline.model_dump(), 'steps': ['visual: ok']}
    except Exception as e:
        _audit(state, 'visual', 'error', str(e), seconds=time.time() - started)
        return {'errors': state.get('errors', []) + [f'visual failed: {e}'], 'steps': [f'visual: failed: {e}']}


def tool_compose(state: AgentState) -> dict:
    from vid2kb.docgen.composer import compose_document
    from vid2kb.docgen.planner import plan_document
    from vid2kb.docgen.validate import validate_document
    from vid2kb.schemas import VisualTimeline

    started = time.time()
    try:
        timeline = VisualTimeline.model_validate(state['timeline'])
        transcript = state.get('transcript') or ''
        spec = plan_document(state.get('user_prompt', ''), transcript[:6000], timeline.summary)
        doc = compose_document(spec, transcript, timeline, state.get('user_prompt', ''))
        frame_timestamps = {f.timestamp_seconds for f in timeline.frames}
        problems = validate_document(doc, spec, transcript, frame_timestamps)
        if problems:
            doc.warnings.extend(problems)
        _audit(state, 'compose', 'done', note=doc.doc_type, seconds=time.time() - started)
        return {'document': doc.model_dump(), 'doc_spec': spec.model_dump(), 'steps': ['compose: ok']}
    except Exception as e:
        _audit(state, 'compose', 'error', str(e), seconds=time.time() - started)
        return {'errors': state.get('errors', []) + [f'compose failed: {e}'], 'steps': [f'compose: failed: {e}']}


def tool_render(state: AgentState) -> dict:
    from vid2kb.docgen.render import render_markdown, render_pdf
    from vid2kb.media.store import ArtifactStore
    from vid2kb.schemas import KnowledgeDocument

    started = time.time()
    try:
        doc = KnowledgeDocument.model_validate(state['document'])
        store = ArtifactStore(state['run_id'])
        markdown = render_markdown(doc)
        (store.out / 'document.md').write_text(markdown, encoding='utf-8')
        pdf = render_pdf(markdown, store.out / 'document.pdf')
        _audit(state, 'render', 'done', seconds=time.time() - started)
        return {'markdown': markdown, 'pdf_path': str(pdf), 'steps': ['render: ok']}
    except Exception as e:
        _audit(state, 'render', 'error', str(e), seconds=time.time() - started)
        return {'errors': state.get('errors', []) + [f'render failed: {e}'], 'steps': [f'render: failed: {e}']}


def build_narration_script(document: dict) -> list[str]:
    title = (document.get('title') or '').strip()
    summary = (document.get('summary') or '').strip()
    key_points = document.get('key_points') or []
    sections = document.get('sections') or []
    if not title and not summary and not key_points and not sections:
        return []

    lines: list[str] = []
    intro = f'视频标题：{title or "未命名视频"}。'
    if summary:
        intro += f'内容概述：{summary}。'
    if key_points:
        intro += '核心要点：' + '；'.join(str(p) for p in key_points) + '。'
    lines.append(intro)

    for sec in sections:
        heading = (sec.get('heading') or '').strip()
        body = (sec.get('body_md') or '').strip()
        if not heading and not body:
            continue
        script = f'{heading}。' if heading else ''
        script += body
        lines.append(script)

    clean: list[str] = []
    for line in lines:
        line = line.replace('#', '').replace('*', '').replace('`', '').strip()
        if line:
            clean.append(line)
    return clean


def tool_voiceover(state: AgentState) -> dict:
    from vid2kb.config import settings
    from vid2kb.media.ffmpeg import probe_duration
    from vid2kb.media.store import ArtifactStore
    from vid2kb.tts.cosyvoice import synthesize_speech

    started = time.time()
    try:
        document = state.get('document') or {}
        scripts = build_narration_script(document)
        if not scripts:
            _audit(state, 'voiceover', 'error', 'empty narration script', seconds=time.time() - started)
            return {
                'errors': state.get('errors', []) + ['voiceover failed: empty document'],
                'steps': ['voiceover: failed: empty document'],
            }
        store = ArtifactStore(state['run_id'])
        out_dir = store.out / 'voiceover'
        out_dir.mkdir(parents=True, exist_ok=True)
        files: list[dict] = []
        total_seconds = 0.0
        for i, script in enumerate(scripts, start=1):
            path = synthesize_speech(
                script,
                voice=settings.tts_voice,
                out_path=out_dir / f'section_{i:02d}.mp3',
                model=settings.tts_model,
                sample_rate=settings.tts_sample_rate,
            )
            seconds = probe_duration(path)
            files.append({'index': i, 'path': str(path), 'seconds': seconds})
            total_seconds += seconds
        _audit(state, 'voiceover', 'done', note=f'{len(files)} files', seconds=time.time() - started)
        return {
            'voiceover': {'files': files, 'total_seconds': total_seconds},
            'steps': ['voiceover: ok'],
        }
    except Exception as e:
        _audit(state, 'voiceover', 'error', str(e), seconds=time.time() - started)
        return {
            'errors': state.get('errors', []) + [f'voiceover failed: {e}'],
            'steps': [f'voiceover: failed: {e}'],
        }


def tool_ingest_kb(state: AgentState) -> dict:
    from vid2kb.kb.store import ingest_document

    started = time.time()
    try:
        document = state.get('document') or {}
        metadata = {
            'run_id': state.get('run_id', ''),
            'doc_title': document.get('title', ''),
            'doc_type': document.get('doc_type', ''),
            'source_url': state.get('source', ''),
            'timestamp_marks': [],
        }
        run_id = state['run_id']
        n = ingest_document(state['markdown'], metadata, doc_id=run_id)
        _audit(state, 'ingest_kb', 'done', note=str(n), seconds=time.time() - started)
        return {'kb_doc_id': run_id, 'kb_node_count': n, 'steps': ['ingest_kb: ok']}
    except Exception as e:
        _audit(state, 'ingest_kb', 'error', str(e), seconds=time.time() - started)
        return {'errors': state.get('errors', []) + [f'ingest_kb failed: {e}'], 'steps': [f'ingest_kb: failed: {e}']}


def tool_report(state: AgentState) -> dict:
    from vid2kb.media.store import ArtifactStore

    started = time.time()
    try:
        document = state.get('document') or {}
        markdown = state.get('markdown')
        store = ArtifactStore(state.get('run_id', 'unknown'))
        markdown_path = str(store.out / 'document.md') if markdown else None
        report = {
            'run_id': state.get('run_id'),
            'title': document.get('title'),
            'doc_type': document.get('doc_type'),
            'markdown_path': markdown_path,
            'pdf_path': state.get('pdf_path'),
            'kb_doc_id': state.get('kb_doc_id'),
            'kb_node_count': state.get('kb_node_count'),
            'transcript_language': state.get('transcript_language'),
            'errors': state.get('errors', []),
            'warnings': document.get('warnings', []) if document else [],
            'steps': [getattr(s, 'content', s) for s in (state.get('steps') or [])],
        }
        _audit(state, 'report', 'done', seconds=time.time() - started)
        return {'final_report': report, 'steps': ['report: ok']}
    except Exception as e:
        _audit(state, 'report', 'error', str(e), seconds=time.time() - started)
        return {'errors': state.get('errors', []) + [f'report failed: {e}'], 'steps': [f'report: failed: {e}']}
