import sys
from unittest.mock import Mock

import pytest

from credit_scoring.serving import main


def test_help_precedes_settings_and_loading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CREDIT_SCORING_SERVING_PORT", "invalid")
    settings = Mock(side_effect=AssertionError("must not construct settings"))
    monkeypatch.setattr(main, "ServingSettings", settings)
    with pytest.raises(SystemExit) as caught:
        main.parse_args(["--help"])
    assert caught.value.code == 0
    assert "--host" in capsys.readouterr().out
    settings.assert_not_called()


@pytest.mark.parametrize(
    "argv", [["--port", "0"], ["--port", "65536"], ["--port", "text"], ["--unknown"]]
)
def test_invalid_arguments(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDIT_SCORING_MLFLOW_TRACKING_URI", "http://registry")
    with pytest.raises(SystemExit) as caught:
        main.parse_args(argv)
    assert caught.value.code == 2


def test_host_port_precedence_and_uvicorn_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDIT_SCORING_MLFLOW_TRACKING_URI", "http://registry")
    defaults = main.parse_args([])
    assert (defaults.serving_host, defaults.serving_port) == ("127.0.0.1", 8000)
    monkeypatch.setenv("CREDIT_SCORING_SERVING_HOST", "localhost")
    monkeypatch.setenv("CREDIT_SCORING_SERVING_PORT", "9000")
    environment = main.parse_args([])
    assert (environment.serving_host, environment.serving_port) == ("localhost", 9000)
    monkeypatch.setattr(sys, "argv", ["serve", "--host", "127.0.0.2", "--port", "9001"])
    app = Mock()
    factory = Mock(return_value=app)
    server = Mock()
    monkeypatch.setattr(main, "create_app", factory)
    monkeypatch.setattr(main.uvicorn, "run", server)
    main.main()
    settings = factory.call_args.args[0]
    assert (settings.serving_host, settings.serving_port) == ("127.0.0.2", 9001)
    server.assert_called_once_with(app, host="127.0.0.2", port=9001, access_log=False)


def test_invalid_configuration_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CREDIT_SCORING_SERVING_PORT", "secret-value")
    with pytest.raises(SystemExit):
        main.parse_args([])
    assert "secret-value" not in capsys.readouterr().err
