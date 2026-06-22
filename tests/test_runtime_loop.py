from vetc.runtime.runtime_loop import RuntimeLoop


def test_runtime_loop_runs_dummy_cycles():
    loop = RuntimeLoop()

    summary = loop.run(cycles=10)

    assert summary["cycles"] == 10
    assert summary["final_action_shape"] == (2, 4)
    assert summary["nan_detected"] is False
    assert summary["avg_step_time_ms"] > 0.0

