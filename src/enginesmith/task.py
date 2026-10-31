"""The task file (task.toml): what to build, how it is judged and the budget. See docs/task.md.

A run freezes its task when it starts (run/task.json); nothing a coding agent does can change it.
"""
from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from enginesmith import agents, workload

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SKILL = REPO / 'skills' / 'specialized-inference-engine'
AGENT_KINDS = ('claude', 'codex', 'kimi', 'pi', 'dsh', 'command')
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z')
_DURATION = re.compile(r'\s*(\d+(?:\.\d+)?)\s*(s|m|h)\s*\Z')
# What a credential in a command line usually looks like. The task is copied into the run directory and every
# session's brief, so credentials belong in the environment or the agent's own configuration.
_SECRET = re.compile(r'(?i)(api[_-]?key|token|secret|password)[A-Z_]*=|\b(sk|ghp|gho|xox[bp])-?[A-Za-z0-9_-]{16,}')


class TaskError(ValueError):
    pass


@dataclass(frozen=True)
class Objective:
    """The score is the weighted geometric mean of the per-case prefill and decode speedups over the reference; a
    case's weight is its length weight times the segment weight."""
    prefill: float = 0.5
    decode: float = 0.5
    length_weights: dict[int, float] = field(default_factory=dict)  # length -> weight; a missing length weighs 1

    def case_weight(self, length: int) -> float:
        return self.length_weights.get(length, 1.0)


@dataclass(frozen=True)
class Acceptance:
    """When a passing candidate replaces the incumbent (controller.decide). 0 turns a rule off."""
    min_improvement: float = 0.015       # score gain
    segment_improvement: float = 0.015   # prefill or decode mean gain, the other one not worse than max_regression
    case_improvement: float = 0.04       # one case's prefill or decode gain
    max_regression: float = 0.01         # how much worse the score may get under the segment and case rules


@dataclass(frozen=True)
class Gates:
    """Token agreement with the reference, judged by teacher forcing (docs/design.md, correctness)."""
    case_mismatch_percent: int = 20
    total_mismatch_percent: int = 10
    mismatch_rank: int = 100  # a mismatching token must be among the reference's top N choices (0: no limit)


@dataclass(frozen=True)
class Evaluation:
    rounds: int = 5          # paired reference/candidate rounds, alternating AB and BA
    warmup: int = 1          # untimed requests at the start of every engine process
    cooldown_s: float = 15.0  # idle time before every engine process starts
    final_rounds: int = 5


@dataclass(frozen=True)
class Budget:
    sessions: int = 24
    wall_time_s: float = 48 * 3600.0
    session_timeout_s: float = 2 * 3600.0


@dataclass(frozen=True)
class Quality:
    """Task accuracy after the final evaluation (enginesmith.quality): greedy, reported, never gated."""
    benchmarks: tuple[str, ...] = ('math500', 'ifeval')
    samples: int = 100            # items per benchmark (a fixed subset)
    max_tokens: int = 8192        # output tokens per item (greedy)
    time_limit_s: float = 2 * 3600.0  # for all benchmarks and both arms together


@dataclass(frozen=True)
class Task:
    path: Path
    name: str
    model: Path
    model_card: Path | None
    vocab_size: int
    reference: tuple[str, ...] | None   # a custom reference command; None: the reference_kind below
    seed: Path | None
    evaluate_seed: bool
    build: tuple[str, ...]
    executable: str
    native: bool
    skill: Path
    agent: str
    agent_model: str | None
    agent_command: tuple[str, ...] | None
    lengths: tuple[int, ...]
    output_tokens: int
    controls: bool
    objective: Objective
    acceptance: Acceptance
    gates: Gates
    evaluation: Evaluation
    budget: Budget
    quality: Quality | None = None
    model_source: str | None = None  # the Hugging Face repository the model was downloaded from (for sharing)
    reference_kind: str | None = None  # enginesmith.references.KINDS; None: the platform's default

    def cases(self) -> list[workload.Case]:
        return workload.cases(self.lengths, self.output_tokens, self.objective.length_weights, self.controls)

    def to_json(self) -> dict:
        def plain(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value
        return {name: plain(value) for name, value in asdict(self).items()}

    @classmethod
    def from_json(cls, data: dict) -> Task:
        """The inverse of to_json (a run's frozen task.json)."""
        def path(value):
            return None if value is None else Path(value)

        def argv(value):
            return None if value is None else tuple(value)
        o = data['objective']
        return cls(Path(data['path']), data['name'], Path(data['model']), path(data['model_card']),
                   data['vocab_size'], argv(data['reference']), path(data['seed']), data['evaluate_seed'],
                   tuple(data['build']), data['executable'], data['native'], Path(data['skill']), data['agent'],
                   data['agent_model'], argv(data['agent_command']), tuple(data['lengths']), data['output_tokens'],
                   data['controls'],
                   Objective(o['prefill'], o['decode'], {int(k): v for k, v in o['length_weights'].items()}),
                   Acceptance(**data['acceptance']), Gates(**data['gates']), Evaluation(**data['evaluation']),
                   Budget(**data['budget']),
                   Quality(**{**data['quality'], 'benchmarks': tuple(data['quality']['benchmarks'])})
                   if data.get('quality') else None, data.get('model_source'), data.get('reference_kind'))

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()


def duration(value, name: str, zero: bool = False) -> float:
    """Seconds from a number of seconds or a string such as "90s", "45m" or "2h"."""
    if type(value) in (int, float):
        seconds = float(value)
    else:
        match = _DURATION.match(value) if type(value) is str else None
        if not match:
            raise TaskError(f'{name}: expected a duration such as "90s", "45m" or "2h"')
        seconds = float(match[1]) * {'s': 1, 'm': 60, 'h': 3600}[match[2]]
    if not (seconds >= 0 if zero else seconds > 0):
        raise TaskError(f'{name} must be {"at least" if zero else "more than"} zero')
    return seconds


def _table(data: dict, name: str, keys: set[str]) -> dict:
    table = data.get(name, {})
    if type(table) is not dict:
        raise TaskError(f'[{name}] must be a table')
    unknown = set(table) - keys
    if unknown:
        raise TaskError(f'[{name}] has unknown keys: {", ".join(sorted(unknown))}')
    return table


def _number(table: dict, key: str, default, low: float, high: float, name: str, integer=False):
    value = table.get(key, default)
    if type(value) is bool or type(value) not in ((int,) if integer else (int, float)) or not low <= value <= high:
        kind = 'an integer' if integer else 'a number'
        raise TaskError(f'{name}.{key} must be {kind} in [{low}, {high}]')
    return value if integer else float(value)


def _argv(value, name: str) -> tuple[str, ...]:
    if type(value) is not list or not value or not all(type(v) is str and v for v in value):
        raise TaskError(f'{name} must be a nonempty list of strings')
    return tuple(value)


def _path(base: Path, value, name: str) -> Path:
    if type(value) is not str or not value:
        raise TaskError(f'{name} must be a path string')
    return (base / Path(value).expanduser()).resolve()


def model_config(model: Path) -> dict:
    """config.json of a model directory; the text model's part when the config nests one (text_config)."""
    config = json.loads((model / 'config.json').read_text())
    return {**config, **config['text_config']} if type(config.get('text_config')) is dict else config


def max_context(config: dict) -> int | None:
    """The model's context length as its config.json states it, if it does."""
    return next((config[k] for k in ('max_position_embeddings', 'max_seq_len', 'seq_length')
                 if type(config.get(k)) is int), None)


def default_lengths(context: int, output_tokens: int) -> list[int]:
    """The default prompt lengths: doubling from 4,096, then the longest prompt that fits the model's context
    together with the generated tokens (at most workload.MAX_LENGTH)."""
    longest = min(context - output_tokens, workload.MAX_LENGTH)
    lengths, n = [], 4096
    while n < longest:
        lengths.append(n)
        n *= 2
    return lengths + [longest]


def load_task(path: Path | str, *, check_paths: bool = True) -> Task:
    path = Path(path).resolve()
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise TaskError(f'{path.name}: {exc}') from None
    base = path.parent
    sections = {'model', 'reference', 'engine', 'agent', 'workload', 'objective', 'acceptance', 'gates',
                'evaluation', 'budget', 'quality'}
    unknown = set(data) - sections - {'name'}
    if unknown:
        raise TaskError(f'unknown top-level keys: {", ".join(sorted(unknown))}')
    name = data.get('name')
    if type(name) is not str or not _NAME.match(name):
        raise TaskError('name must be 1-64 characters of letters, digits, ".", "_" or "-"')

    m = _table(data, 'model', {'path', 'card', 'source'})
    if 'path' not in m:
        raise TaskError('[model] path is required')
    model = _path(base, m['path'], 'model.path')
    card = _path(base, m['card'], 'model.card') if 'card' in m else None
    source = m.get('source')
    if source is not None and (type(source) is not str or not re.fullmatch(r'[\w.-]+/[\w.-]+', source)):
        raise TaskError('model.source must be a Hugging Face repository id such as "mlx-community/Qwen3-0.6B-4bit"')

    r = _table(data, 'reference', {'command', 'kind'})
    reference = _argv(r['command'], 'reference.command') if 'command' in r else None
    from enginesmith.references import KINDS
    reference_kind = r.get('kind')
    if reference_kind is not None and reference_kind not in KINDS:
        raise TaskError(f'reference.kind must be one of {", ".join(KINDS)}')
    if reference_kind and reference:
        raise TaskError('give reference.kind or reference.command, not both')

    e = _table(data, 'engine', {'seed', 'evaluate_seed', 'build', 'executable', 'native', 'skill'})
    seed = _path(base, e['seed'], 'engine.seed') if 'seed' in e else None
    evaluate_seed = e.get('evaluate_seed', seed is not None)
    native = e.get('native', True)
    if type(evaluate_seed) is not bool or type(native) is not bool:
        raise TaskError('engine.evaluate_seed and engine.native must be true or false')
    if evaluate_seed and seed is None:
        raise TaskError('engine.evaluate_seed needs engine.seed')
    build = _argv(e.get('build', ['cargo', 'build', '--release']), 'engine.build')
    executable = e.get('executable', 'target/release/engine')
    if type(executable) is not str or not executable or Path(executable).is_absolute() or '..' in Path(executable).parts:
        raise TaskError('engine.executable must be a relative path inside the engine source')
    skill = _path(base, e['skill'], 'engine.skill') if 'skill' in e else DEFAULT_SKILL

    a = _table(data, 'agent', {'kind', 'model', 'command', 'session_timeout'})
    # By default the sessions run on the agent the user is talking to.
    agent = a.get('kind') or agents.current()
    if agent is None:
        raise TaskError(f'agent.kind is not set and this shell runs in no agent EngineSmith recognizes; set it to one '
                        f'of {", ".join(AGENT_KINDS)}')
    if agent not in AGENT_KINDS:
        raise TaskError(f'agent.kind must be one of {", ".join(AGENT_KINDS)}')
    agent_model = a.get('model')
    if agent_model is not None and (type(agent_model) is not str or not agent_model):
        raise TaskError('agent.model must be a nonempty string')
    if agent_model is not None and agent == 'dsh':
        raise TaskError('agent.model does not apply to kind = "dsh": DeepSeek Harness uses the default model of its '
                        'own settings')
    agent_command = _argv(a['command'], 'agent.command') if 'command' in a else None
    if (agent == 'command') != (agent_command is not None):
        raise TaskError('agent.command is required with kind = "command" and only then')
    if agent_command and any(_SECRET.search(part) for part in agent_command):
        raise TaskError('agent.command looks like it contains a credential; the task is copied into every session, '
                        'so set API keys in your environment or in the agent\'s own configuration')

    w = _table(data, 'workload', {'lengths', 'output_tokens', 'controls', 'vocab_size'})
    output_tokens = _number(w, 'output_tokens', 65, 2, 8192, 'workload', integer=True)
    controls = w.get('controls', True)
    if type(controls) is not bool:
        raise TaskError('workload.controls must be true or false')
    config = model_config(model) if check_paths and (model / 'config.json').is_file() else {}
    if 'vocab_size' in w:
        vocab = _number(w, 'vocab_size', None, 2, 2**31, 'workload', integer=True)
    elif check_paths:
        vocab = config.get('vocab_size')
        if type(vocab) is not int or vocab < 2:
            raise TaskError('no vocab_size in the model config.json; set workload.vocab_size')
    else:
        vocab = 0
    if 'lengths' in w:
        lengths = w['lengths']
    elif context := max_context(config):
        lengths = default_lengths(context, output_tokens)
    else:
        raise TaskError('the model config.json states no context length (max_position_embeddings); set '
                        'workload.lengths')
    if (type(lengths) is not list or not lengths or len(set(lengths)) != len(lengths)
            or not all(type(n) is int and 1 <= n <= workload.MAX_LENGTH for n in lengths)):
        raise TaskError(f'workload.lengths must be distinct integers in [1, {workload.MAX_LENGTH}]')

    o = _table(data, 'objective', {'prefill', 'decode', 'length_weights'})
    weights = o.get('length_weights', {})
    if type(weights) is not dict:
        raise TaskError('objective.length_weights must be a table of length = weight')
    length_weights = {}
    for key, value in weights.items():
        if not key.isdigit() or int(key) not in lengths:
            raise TaskError(f'objective.length_weights: {key} is not one of workload.lengths')
        length_weights[int(key)] = _number(weights, key, None, 0, 1e6, 'objective.length_weights')
    objective = Objective(_number(o, 'prefill', 0.5, 0, 1e6, 'objective'),
                          _number(o, 'decode', 0.5, 0, 1e6, 'objective'), length_weights)
    if objective.prefill + objective.decode <= 0 or not any(objective.case_weight(n) > 0 for n in lengths):
        raise TaskError('the objective weights must not all be zero')

    c = _table(data, 'acceptance', {'min_improvement', 'segment_improvement', 'case_improvement', 'max_regression'})
    acceptance = Acceptance(*(_number(c, key, getattr(Acceptance, key), 0, 0.99, 'acceptance')
                              for key in ('min_improvement', 'segment_improvement', 'case_improvement',
                                          'max_regression')))

    g = _table(data, 'gates', {'case_mismatch_percent', 'total_mismatch_percent', 'mismatch_rank'})
    gates = Gates(_number(g, 'case_mismatch_percent', 20, 0, 100, 'gates', integer=True),
                  _number(g, 'total_mismatch_percent', 10, 0, 100, 'gates', integer=True),
                  _number(g, 'mismatch_rank', 100, 0, 2**31, 'gates', integer=True))

    v = _table(data, 'evaluation', {'rounds', 'warmup', 'cooldown', 'final_rounds'})
    evaluation = Evaluation(_number(v, 'rounds', 5, 1, 100, 'evaluation', integer=True),
                            _number(v, 'warmup', 1, 0, 100, 'evaluation', integer=True),
                            duration(v['cooldown'], 'evaluation.cooldown', zero=True) if 'cooldown' in v else 15.0,
                            _number(v, 'final_rounds', 5, 1, 100, 'evaluation', integer=True))

    b = _table(data, 'budget', {'sessions', 'wall_time'})
    budget = Budget(_number(b, 'sessions', 24, 1, 10000, 'budget', integer=True),
                    duration(b['wall_time'], 'budget.wall_time') if 'wall_time' in b else 48 * 3600.0,
                    duration(a['session_timeout'], 'agent.session_timeout') if 'session_timeout' in a else 7200.0)

    quality = _quality(data) if 'quality' in data else None

    if check_paths:
        for label, item in (('model.path', model), ('engine.skill', skill), ('engine.seed', seed)):
            if item is not None and not item.is_dir():
                raise TaskError(f'{label}: {item} is not a directory')
        if card is not None and not card.is_file():
            raise TaskError(f'model.card: {card} is not a file')
        if not (skill / 'SKILL.md').is_file():
            raise TaskError(f'engine.skill: {skill} has no SKILL.md')
    return Task(path, name, model, card, vocab, reference, seed, evaluate_seed, build, executable, native, skill,
                agent, agent_model, agent_command, tuple(sorted(lengths)), output_tokens, controls, objective,
                acceptance, gates, evaluation, budget, quality, source, reference_kind)


def _quality(data: dict) -> Quality:
    q = _table(data, 'quality', {'benchmarks', 'samples', 'max_tokens', 'time_limit'})
    benchmarks = q.get('benchmarks', ['math500', 'ifeval'])
    if (type(benchmarks) is not list or not benchmarks or len(set(benchmarks)) != len(benchmarks)
            or not all(b in ('math500', 'ifeval') for b in benchmarks)):
        raise TaskError('quality.benchmarks must be distinct names among "math500" and "ifeval"')
    return Quality(tuple(benchmarks), _number(q, 'samples', 100, 1, 10000, 'quality', integer=True),
                   _number(q, 'max_tokens', 8192, 16, 131072, 'quality', integer=True),
                   duration(q['time_limit'], 'quality.time_limit') if 'time_limit' in q else 2 * 3600.0)
