from patchforge.contracts import AgentEvent, AgentOutput, SandboxResult


def test_event_defaults_and_roundtrip():
    ev = AgentEvent(stage="plan", message="hi")
    assert ev.tokens_in == 0 and ev.tokens_out == 0 and ev.cost_usd == 0 and ev.model is None
    assert AgentEvent.model_validate_json(ev.model_dump_json()) == ev


def test_output_roundtrip():
    out = AgentOutput(changed_files=["a.py"], summary="s",
                      test_results=SandboxResult(passed=False, failed_tests=["t::x"], error_log="boom"))
    assert AgentOutput.model_validate(out.model_dump()) == out
