import json

from conftest import write_task
from enginesmith import cli, workload
from enginesmith.agents import argv


def output(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_validate_reports_the_task_and_its_cases(tmp_path, model_dir, capsys):
    assert cli.main(['validate', str(write_task(tmp_path, model_dir))]) == 0
    result = output(capsys)
    assert result['valid'] and [c['id'] for c in result['cases']] == ['p64', 'p256']
    (tmp_path / 'bad.toml').write_text('name = "x"\n[model]\npath = "nowhere"\n')
    assert cli.main(['validate', str(tmp_path / 'bad.toml')]) == 2
    assert not output(capsys)['valid']


def test_dataset_writes_the_public_prompts(tmp_path, capsys):
    out = tmp_path / 'dev.jsonl'
    assert cli.main(['dataset', '--vocab', '5000', '--lengths', '16,256', '--out', str(out)]) == 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines[0] == {'id': 'p16', 'prompt': workload.prompt(16, 5000), 'output_tokens': 65, 'weight': 1.0}
    assert lines[1]['prompt'][:16] == lines[0]['prompt'] and lines[-1]['weight'] == 0.0


def test_probe_reads_the_config_and_estimates_the_kv_cache(tmp_path, capsys):
    model = tmp_path / 'm'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({
        'model_type': 'x', 'vocab_size': 32000, 'num_hidden_layers': 4, 'hidden_size': 512,
        'num_attention_heads': 8, 'num_key_value_heads': 2, 'max_position_embeddings': 32768,
        'layer_types': ['full_attention', 'sliding_attention', 'full_attention', 'linear_attention']}))
    (model / 'model.safetensors').write_bytes(b'\0' * 1000)
    assert cli.main(['probe', str(model)]) == 0
    facts = output(capsys)
    assert facts['kv_bytes_per_token_estimate'] == 2 * 2 * 2 * 64 * 2 and facts['weights_bytes'] == 1000
    assert facts['lengths']['16384'] == 'fits (estimate)'
    assert facts['lengths']['32768'].startswith('beyond the model context')


def test_the_agent_launchers():
    workspace = '/w'
    claude = argv('claude', None, 'opus', workspace)
    assert claude[:2] == ['claude', '-p'] and claude[-2:] == ['--model', 'opus'] and '/w' in claude[2]
    codex = argv('codex', None, None, workspace)
    assert codex[:2] == ['codex', 'exec'] and codex[codex.index('--cd') + 1] == '/w' and 'brief/README.md' in codex[-1]
    kimi = argv('kimi', None, 'kimi-for-coding', workspace)
    assert kimi[:2] == ['kimi', '-p'] and kimi[-2:] == ['--model', 'kimi-for-coding'] and '/w' in kimi[2]
    pi = argv('pi', None, 'deepseek-flash', workspace)
    assert pi[:5] == ['pi', '--print', '--mode', 'json', '--no-session'] and pi[5:7] == ['--model', 'deepseek-flash']
    dsh = argv('dsh', None, None, workspace)
    assert dsh[:6] == ['env', 'DSH_PERMISSION_MODE=danger-full-access', 'dsh', '--profile', 'headless', '--json']
    assert '/w' in dsh[-1]
    assert argv('command', ['run', '{workspace}', '{prompt}'], None, workspace)[:2] == ['run', '/w']


def test_the_current_agent_comes_from_its_markers(monkeypatch):
    from enginesmith.agents import current
    assert current() is None
    monkeypatch.setenv('CLAUDECODE', '1')
    assert current() == 'claude'
    monkeypatch.setenv('DSH_SHELL', '1')   # DeepSeek Harness started from a Claude Code terminal
    assert current() == 'dsh'


def test_each_agent_s_usage_format_is_read(tmp_path, monkeypatch):
    from enginesmith.agents import usage
    # Codex and Claude Code record the model or the reasoning effort in their own session files.
    monkeypatch.setenv('CODEX_HOME', str(tmp_path / 'codex'))
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(tmp_path / 'claude'))
    rollout = tmp_path / 'codex' / 'sessions' / '2026' / '10' / '31' / 'rollout-2026-10-31-0a1b2c3d-thread.jsonl'
    rollout.parent.mkdir(parents=True)
    rollout.write_text(''.join(json.dumps(r) + '\n' for r in (
        {'type': 'session_meta', 'payload': {}},
        {'type': 'turn_context', 'payload': {'model': 'gpt-a', 'effort': 'low'}},
        {'type': 'turn_context', 'payload': {'model': 'gpt-b', 'effort': 'high'}})))
    transcript = tmp_path / 'claude' / 'projects' / '-work' / 'claude-session-1.jsonl'
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({'type': 'assistant', 'effort': 'xhigh'}) + '\n')
    formats = {
        'pi': [{'type': 'message_end', 'message': {'role': 'assistant', 'model': 'deepseek-flash',
                                                   'thinkingLevel': 'high', 'usage': {'input': 9, 'output': 1}}}],
        'dsh': [{'type': 'status', 'phase': 'step_end', 'usage': {'inputTokens': 10, 'outputTokens': 5,
                                                                'cacheReadTokens': 100, 'cacheWriteTokens': 0}}] * 2,
        'claude': [{'type': 'system', 'subtype': 'init', 'model': 'claude-x', 'session_id': 'claude-session-1'},
                   {'type': 'result', 'total_cost_usd': 1.5, 'usage': {
                       'input_tokens': 7, 'output_tokens': 3, 'cache_read_input_tokens': 40,
                       'cache_creation_input_tokens': 2}}],
        'codex': [{'type': 'thread.started', 'thread_id': '0a1b2c3d-thread'},
                  {'type': 'turn.completed', 'usage': {'input_tokens': 50, 'cached_input_tokens': 30,
                                                       'output_tokens': 4}}],
    }
    expected = {'pi': ('deepseek-flash', 'high', 9, 1, None, None, None),
                'dsh': (None, None, 20, 10, 200, 0, None), 'claude': ('claude-x', 'xhigh', 7, 3, 40, 2, 1.5),
                'codex': ('gpt-b', 'high', 20, 4, 30, None, None)}
    for kind, events in formats.items():
        log = tmp_path / f'{kind}.log'
        log.write_text(''.join(json.dumps(e) + '\n' for e in events) + 'a plain line\n')
        got = usage(log)
        assert (got['model'], got['effort'], got['input_tokens'], got['output_tokens'], got['cache_read_tokens'],
                got['cache_write_tokens'], got['cost_usd']) == expected[kind], kind
    assert set(usage(tmp_path / 'missing.log').values()) == {None}


def test_sessions_do_not_inherit_the_parent_agent_s_session(monkeypatch, tmp_path):
    from enginesmith.agents import environment
    monkeypatch.setenv('CLAUDECODE', '1')
    monkeypatch.setenv('PI_SESSION_ID', 'parent')
    monkeypatch.setenv('DSH_SESSION_ID', 'parent')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'kept')
    env = environment(tmp_path / 'session.json')
    assert 'CLAUDECODE' not in env and 'PI_SESSION_ID' not in env and 'DSH_SESSION_ID' not in env
    assert env['ANTHROPIC_API_KEY'] == 'kept'
    assert env['ENGINESMITH_SESSION'] == str(tmp_path / 'session.json')


def test_explain_prints_the_plan_from_the_task(tmp_path, model_dir, capsys):
    path = write_task(tmp_path, model_dir, objective={'prefill': 0.25, 'decode': 0.75, 'length_weights': {'256': 3}},
                      acceptance={'case_improvement': 0})
    (tmp_path / 'prepare.json').write_text(json.dumps({'observed': {'load_ms': 500, 'lengths': [
        {'length': 64, 'prefill_ms': 10, 'decode_ms': 70, 'decode_ms_per_token': 10, 'peak_memory_bytes': 2**30},
        {'length': 256, 'prefill_ms': 30, 'decode_ms': 70, 'decode_ms_per_token': 10, 'peak_memory_bytes': 2**30}]}}))
    assert cli.main(['explain', str(path)]) == 0
    plan = capsys.readouterr().out
    assert '| 256 | 3 | 18.8% | 56.2% | 0.03 s | 10.00 ms | 1.0 GiB |' in plan
    assert 'w_prefill = 0.25, w_decode = 0.75' in plan and 'score improves by at least 1.5%' in plan
    assert "one case's prefill or decode speedup improves" not in plan   # the case rule is off
    assert 'at least 300 s' in plan and '## 9. Budget' in plan
    assert '**Starting point:** an empty project' in plan
    written = (tmp_path / 'plan.md').read_text()
    assert written.startswith('<!-- enginesmith plan ') and written.split('\n', 1)[1] in plan
    (tmp_path / 'seed').mkdir()
    path.write_text(path.read_text().replace('[engine]\n', '[engine]\nseed = "seed"\n'))
    cli.main(['explain', str(path)])
    assert '**Starting point:** the engine source in `seed` (`engine.seed`). It is evaluated first' in (
        capsys.readouterr().out)


def test_a_run_needs_a_current_plan(tmp_path, model_dir, capsys):
    path = write_task(tmp_path, model_dir)
    assert cli.main(['run', str(path), '--runs-dir', str(tmp_path / 'runs')]) == 2
    assert 'enginesmith explain' in output(capsys)['error'] and not (tmp_path / 'runs').exists()
    cli.main(['explain', str(path)])
    capsys.readouterr()
    from enginesmith.explain import plan_problem
    from enginesmith.task import load_task
    assert plan_problem(load_task(path)) is None
    path.write_text(path.read_text().replace('sessions = 3', 'sessions = 4'))   # the user changed the task
    assert 'changed since' in plan_problem(load_task(path))


def test_a_run_does_not_start_inside_an_agent_sandbox(tmp_path, model_dir, capsys, monkeypatch):
    from enginesmith import setup_tools
    path = write_task(tmp_path, model_dir)
    cli.main(['explain', str(path)])
    capsys.readouterr()
    monkeypatch.setattr(setup_tools, 'sandboxed', lambda: True)
    assert cli.main(['run', str(path), '--runs-dir', str(tmp_path / 'runs')]) == 2
    assert 'outside the sandbox' in output(capsys)['error'] and not (tmp_path / 'runs').exists()
    assert cli.main(['resume', str(tmp_path / 'runs' / 'x')]) == 2
