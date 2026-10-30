import pytest

from enginesmith import workload


def test_the_lily_formula_and_the_prefix_property():
    assert workload.prompt(3, 248320) == [0, 2654435761 % 248320, (2 * 2654435761 & 0xFFFFFFFF) % 248320]
    assert workload.prompt(1000, 50000)[:300] == workload.prompt(300, 50000)
    assert workload.prompt(5, 97, 10) == workload.prompt(15, 97)[10:]


def test_the_longest_prompt_is_512k_tokens_and_the_ranges_are_disjoint():
    assert workload.MAX_LENGTH == 524288 and workload.LENGTHS[-1] == 524288
    assert workload.PUBLIC[1] == workload.HELD_OUT[0] and workload.HELD_OUT[1] == workload.NONCE[0]
    assert len(workload.prompt(524288, 151936)) == 524288


@pytest.mark.parametrize('span', [workload.HELD_OUT, workload.NONCE])
def test_drawn_windows_lie_inside_their_range(span):
    for n in (1, 4096, workload.MAX_LENGTH):
        for label in ('a', 'b', 'c'):
            start = workload.draw_offset(b'seed', label, n, span)
            assert span[0] <= start and start + n <= span[1]
    assert workload.draw_offset(b's', 'x', 10, span) == workload.draw_offset(b's', 'x', 10, span)
    assert workload.draw_offset(b's', 'x', 10, span) != workload.draw_offset(b't', 'x', 10, span)


def test_cases_add_the_controls_that_are_not_timed_lengths():
    cases = workload.cases([64, 1024], 65, {1024: 2.0})
    timed = [c for c in cases if c.timed]
    assert [(c.id, c.weight) for c in timed] == [('p64', 1.0), ('p1024', 2.0)]
    controls = [c.length for c in cases if not c.timed]
    assert 64 not in controls and 65 in controls and 4097 in controls
    assert workload.cases([64], 8, controls=False) == [workload.Case('p64', 64, 8, 1.0)]


def test_invalid_prompts_are_refused():
    for args in ((0, 10), (5, 1), (5, 10, 2**32 - 4)):
        with pytest.raises(ValueError):
            workload.prompt(*args)
