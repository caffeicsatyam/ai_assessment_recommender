import app.main as main_module
from fastapi.testclient import TestClient


def test_chat_accepts_newlines_inside_message_content(monkeypatch):
    client = TestClient(main_module.app)

    monkeypatch.setattr(
        main_module, "generate_response", lambda messages: ("ok", [], False)
    )

    body = b'{"conversation_id":"abc","messages":[{"role":"user","content":"Hello\nworld"}]}'

    response = client.post(
        "/chat",
        content=body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == "abc"
    assert payload["reply"] == "ok"
    assert payload["recommendations"] == []
