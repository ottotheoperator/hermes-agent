"""A live MCP connection must survive a ``tools.include`` edit, and a connection that is up while
this scope's registry holds none of its tools must be reported instead of silently thinning a
session's tool list (card t_5ef894b0).

The reported failure, reproduced by these tests at the unit level: Hanna's 4-hourly job logged
"125 MCP tool(s) available" until ``fluent_evergreen.tools.include`` gained one tool, and the next
run inside the same multiplexed gateway logged 111 (= the 14 tools of ``fluent_evergreen`` gone) with
no other line in any log. The include/exclude filter is part of ``config_fingerprint``, which
``_connection_identity`` uses as its route half, so the reconcile read the profile's OWN live
connection as a different connection: it removed that server's whole overlay and kept the session
(which went on answering keepalives), and discovery never rebuilt the tools because a connected
server is not a connect candidate.
"""

from __future__ import annotations

import copy
import json
import logging
from types import SimpleNamespace

import pytest

from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override

_URL = "https://mcp.example/x"


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description="d", inputSchema={"type": "object", "properties": {}},
                           annotations=None)


def _server(name: str, cfg: dict, tools: tuple[str, ...] = ("t",)) -> SimpleNamespace:
    """A live ``MCPServerTask`` stand-in: session up, config as connected, tools as listed."""
    from tools.mcp_tool_registration import _adopter_identity_digest
    return SimpleNamespace(name=name, session=object(), _config=cfg, _tools=[_tool(t) for t in tools],
                           tool_timeout=30, initialize_result=None, _registered_tool_names=[],
                           _sampling=None, _resolved_identity=_adopter_identity_digest(name, cfg))


@pytest.fixture
def profile(tmp_path, monkeypatch):
    """One served profile scope, clean MCP ledgers, and a scope entry helper; restores everything."""
    import tools.mcp_tool as core
    from tools import mcp_tool_config as _config
    from tools.registry import registry

    home = tmp_path / "profiles" / "hanna"
    (home / "cache").mkdir(parents=True)
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(_config, "_filter_suspicious_mcp_servers", lambda servers: servers)
    ledgers = ("_servers", "_server_scope_keys", "_server_tool_scopes", "_server_connecting",
               "_server_connect_errors", "_server_connect_retry_after", "_server_connect_failures",
               "_server_error_counts", "_server_breaker_opened_at", "_lazy_server_configs",
               "_lazy_server_fingerprints", "_lazy_server_tool_names", "_mcp_tool_server_names",
               "_orphaned_adopters", "_parallel_safe_servers", "_server_trust_levels",
               "_tool_read_only_hints", "_server_cost_policies")
    saved = {n: type(getattr(core, n))(getattr(core, n)) for n in ledgers}
    for n in ledgers:
        getattr(core, n).clear()
    registry.register_toolset_alias("x", "mcp-x")
    token = set_hermes_home_override(home)
    scope = hermes_home_key(home)

    yield SimpleNamespace(home=home, scope=scope, registry=registry, core=core)

    for tool_name in list(registry.get_tool_names_for_toolset("mcp-x")):
        registry.deregister(tool_name, scope=scope)
        registry.deregister(tool_name)
    registry._drop_toolset_aliases("mcp-x")
    reset_hermes_home_override(token)
    for n in ledgers:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])


def _config(include=None, url=_URL) -> dict:
    cfg = {"url": url, "headers": {"Authorization": "Bearer a"}}
    if include is not None:
        cfg["tools"] = {"include": list(include)}
    return cfg


def test_editing_tools_include_keeps_the_live_connection_and_its_tools(profile):
    """Hans adding a tool to ``tools.include`` must not cost the profile the tools it already has.

    On bf0203a794 the reconcile read the live connection as a DIFFERENT connection (the filter was
    part of ``config_fingerprint``), dropped its whole overlay and left the session up, and nothing
    re-registered it: that was the reported failure (t_5ef894b0). With the filter out of the
    identity the toolset survives the edit, and the discovery pass re-applies the new filter to it
    (t_aa995a52) instead of waiting for a restart.
    """
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    before = _config(include=["t"])
    after = copy.deepcopy(before)
    after["tools"]["include"] = ["t", "t2"]  # one more tool whitelisted; same URL, same credentials

    server = _server("x", before, tools=("t", "t2"))
    disc._adopt_server("x", server)
    assert reg._register_server_tools("x", server, before) == ["mcp__x__t"]

    disc.register_mcp_servers({"x": after})

    # The connection was not replaced, so its registration must stand: discovery skips a connected
    # server, which means nothing else in this process could put the lost tools back.
    assert profile.core._servers[(profile.scope, "x")] is server
    assert profile.registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t", "mcp__x__t2"]
    assert profile.scope in profile.core._server_tool_scopes[(profile.scope, "x")]


def test_a_changed_endpoint_still_detaches_the_overlay(profile):
    """The stale verdict must keep working for what it is for: the connection itself changed."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    before = _config()
    after = _config(url="https://mcp.example/y")

    server = _server("x", before)
    disc._adopt_server("x", server)
    assert reg._register_server_tools("x", server, before) == ["mcp__x__t"]

    reg.register_connected_into_current_scope({"x": after})

    assert profile.registry.get_tool_names_for_toolset("mcp-x") == []
    assert profile.scope not in profile.core._server_tool_scopes.get((profile.scope, "x"), ())


def test_live_connection_with_no_registered_tools_warns(profile, caplog):
    """The state that took Hanna's run down logged nothing; it must name the server and the repair."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    cfg = _config()
    server = _server("x", cfg)
    disc._adopt_server("x", server)
    server._registered_tool_names = reg._register_server_tools("x", server, cfg)
    # Exactly the aftermath of the strip: session up, overlay gone, ``_registered_tool_names`` stale.
    reg._remove_server_scope((profile.scope, "x"), profile.scope)

    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        disc.register_mcp_servers({"x": cfg})

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    lost = [m for m in warnings if "the connection is live" in m]
    assert len(lost) == 1, warnings
    assert "'x'" in lost[0]
    assert "none of the 1 tool(s) it registered is in this scope's registry" in lost[0]
    assert "/reload-mcp" in lost[0] and "restart the gateway" in lost[0]


def test_a_connection_that_never_registered_anything_does_not_warn(profile, caplog):
    """Nothing was taken away (an all-collision or not-yet-registered server): registration logs that."""
    from tools import mcp_tool_discovery as disc

    server = _server("x", _config())
    disc._adopt_server("x", server)

    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        disc.register_mcp_servers({"x": _config()})

    assert not [r for r in caplog.records if "the connection is live" in r.getMessage()]


def test_a_healthy_connection_warns_nothing(profile, caplog):
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    cfg = _config()
    server = _server("x", cfg)
    disc._adopt_server("x", server)
    server._registered_tool_names = reg._register_server_tools("x", server, cfg)

    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        disc.register_mcp_servers({"x": cfg})

    assert not [r for r in caplog.records if "the connection is live" in r.getMessage()]


def test_an_empty_include_list_warns_nothing(profile, caplog):
    """``include: []`` registers nothing on purpose (the "uncheck everything" path): not a loss."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    cfg = _config(include=[])
    server = _server("x", cfg)
    disc._adopt_server("x", server)
    assert reg._register_server_tools("x", server, cfg) == []

    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        disc.register_mcp_servers({"x": cfg})

    assert not [r for r in caplog.records if "the connection is live" in r.getMessage()]


# ---- The filter edit must take effect while the session is live (t_aa995a52) ------------------
# The identity fix removes the old symptom (an edit no longer detaches the overlay) and leaves the
# other half: the connection stays up, so nothing re-applies the new filter, and a tool the user
# just removed keeps answering until the process restarts. These tests pin the re-apply.


def test_a_tool_removed_from_include_stops_being_callable_without_a_restart(profile):
    """Narrowing ``tools.include`` on a live connection must take effect in that same session."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    before = _config(include=["t", "t2"])
    after = copy.deepcopy(before)
    after["tools"]["include"] = ["t"]  # one tool taken away; same URL, same credentials

    server = _server("x", before, tools=("t", "t2"))
    disc._adopt_server("x", server)
    assert reg._register_server_tools("x", server, before) == ["mcp__x__t", "mcp__x__t2"]

    disc.register_mcp_servers({"x": after})  # the next discovery pass: a new session, same process

    assert profile.core._servers[(profile.scope, "x")] is server  # never reconnected
    assert profile.registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]
    assert profile.registry.get_entry("mcp__x__t2") is None
    assert json.loads(profile.registry.dispatch("mcp__x__t2", {}))["error"].startswith("Unknown tool")


def test_a_tool_added_to_include_becomes_usable_without_a_restart(profile):
    """Widening ``tools.include`` must hand the live session the new tool, not the next process."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    before = _config(include=["t"])
    after = copy.deepcopy(before)
    after["tools"]["include"] = ["t", "t2"]

    server = _server("x", before, tools=("t", "t2"))
    disc._adopt_server("x", server)
    assert reg._register_server_tools("x", server, before) == ["mcp__x__t"]

    disc.register_mcp_servers({"x": after})

    assert profile.core._servers[(profile.scope, "x")] is server
    assert profile.registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t", "mcp__x__t2"]
    entry = profile.registry.get_entry("mcp__x__t2")
    assert entry is not None and entry.toolset == "mcp-x"
    assert profile.registry.get_schema("mcp__x__t2")["parameters"]["type"] == "object"


def test_the_reapplied_filter_is_reported_once_per_change_not_once_per_pass(profile, caplog):
    """A change is worth one line. A discovery pass that finds nothing to do is worth none."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    before = _config(include=["t", "t2"])
    after = copy.deepcopy(before)
    after["tools"]["include"] = ["t"]

    server = _server("x", before, tools=("t", "t2"))
    disc._adopt_server("x", server)
    reg._register_server_tools("x", server, before)

    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        for _ in range(3):
            disc.register_mcp_servers({"x": after})

    changed = [r.getMessage() for r in caplog.records if "re-applied" in r.getMessage()]
    assert len(changed) == 1, changed
    assert "'x'" in changed[0]
    assert "removed mcp__x__t2" in changed[0]


def test_a_filter_edit_leaves_the_generated_utilities_alone(profile):
    """``tools.include`` selects the server's own tools; the resource/prompt utilities are not its to drop."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    before = _config(include=["t", "t2"])
    after = copy.deepcopy(before)
    after["tools"]["include"] = ["t"]

    server = _server("x", before, tools=("t", "t2"))
    server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(resources=object()))
    disc._adopt_server("x", server)
    registered = reg._register_server_tools("x", server, before)
    assert "mcp__x__list_resources" in registered

    disc.register_mcp_servers({"x": after})

    assert profile.registry.get_tool_names_for_toolset("mcp-x") == [
        "mcp__x__list_resources", "mcp__x__read_resource", "mcp__x__t"]


def test_a_tool_comes_back_when_the_include_list_does(profile):
    """An emptied overlay must stay adopted, or the reverse edit could never find it again."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    cfg = _config(include=["t"])
    empty = copy.deepcopy(cfg)
    empty["tools"]["include"] = []  # the install checklist's "uncheck everything" path

    server = _server("x", cfg, tools=("t",))
    disc._adopt_server("x", server)
    assert reg._register_server_tools("x", server, cfg) == ["mcp__x__t"]

    disc.register_mcp_servers({"x": empty})
    assert profile.registry.get_tool_names_for_toolset("mcp-x") == []

    disc.register_mcp_servers({"x": cfg})
    assert profile.registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]


def test_a_single_profile_process_reapplies_the_filter_too(profile, monkeypatch):
    """Without a multiplexer the connection is keyed by its bare name and its tools sit in the global
    map; the re-apply must reach that path as well, since most processes are single-profile."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    monkeypatch.setattr(profile.core, "_mcp_registry_scope", lambda: None)

    before = _config(include=["t", "t2"])
    after = copy.deepcopy(before)
    after["tools"]["include"] = ["t"]

    server = _server("x", before, tools=("t", "t2"))
    disc._adopt_server("x", server)
    assert reg._register_server_tools("x", server, before) == ["mcp__x__t", "mcp__x__t2"]

    disc.register_mcp_servers({"x": after})

    assert profile.registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]
    assert profile.registry.get_entry("mcp__x__t2") is None


def test_a_widened_filter_never_takes_a_builtin_tools_name(profile, caplog):
    """Widening the filter must not steal a name a built-in toolset already owns, and the refusal
    must be reported: silently dropping the tool the user just added is the failure this card is about."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    profile.registry.register(
        name="mcp__x__t2", toolset="core",
        schema={"name": "mcp__x__t2", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kwargs: "built-in")
    try:
        before = _config(include=["t"])
        after = copy.deepcopy(before)
        after["tools"]["include"] = ["t", "t2"]

        server = _server("x", before, tools=("t", "t2"))
        disc._adopt_server("x", server)
        assert reg._register_server_tools("x", server, before) == ["mcp__x__t"]

        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            disc.register_mcp_servers({"x": after})

        assert profile.registry.get_toolset_for_tool("mcp__x__t2") == "core"
        assert profile.registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]
        assert any("collides with built-in" in r.getMessage() for r in caplog.records), \
            [r.getMessage() for r in caplog.records]
    finally:
        profile.registry.deregister("mcp__x__t2")


def test_a_route_and_filter_edit_together_is_still_a_different_connection(profile):
    """An edit that moves the endpoint AND narrows the filter is not a filter edit: the live session
    keeps the config it was opened with, so it can never be adopted later under the new identity."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    before = _config(include=["t", "t2"])
    after = {"url": "https://mcp.example/y", "headers": dict(before["headers"]), "tools": {"include": ["t"]}}

    server = _server("x", before, tools=("t", "t2"))
    disc._adopt_server("x", server)
    assert reg._register_server_tools("x", server, before) == ["mcp__x__t", "mcp__x__t2"]

    disc.register_mcp_servers({"x": after})

    assert server._config == before, "the live session must keep the config it was opened with"
    assert not reg._same_server_route(server, after)
    assert profile.registry.get_tool_names_for_toolset("mcp-x") == []  # detached, as landed


def test_another_profiles_overlay_keeps_its_tool_and_its_provenance(profile):
    """Removal is scope-precise: a second profile's overlay of the same live connection keeps the tool,
    and the process-wide provenance stays, because that map is keyed by tool name alone."""
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    before = _config(include=["t", "t2"])
    after = copy.deepcopy(before)
    after["tools"]["include"] = ["t"]

    server = _server("x", before, tools=("t", "t2"))
    server._registered_tool_names = ["mcp__x__t", "mcp__x__t2"]
    key = ("p1", "x")
    with profile.core._lock:
        profile.core._servers[key] = server
        profile.core._server_scope_keys[key] = "p1"
        profile.core._server_tool_scopes[key] = {"p1", "p2"}
    for owner in ("p1", "p2"):
        for tool_name in ("mcp__x__t", "mcp__x__t2"):
            registry.register(name=tool_name, toolset="mcp-x",
                              schema={"name": tool_name, "parameters": {"type": "object", "properties": {}}},
                              handler=lambda args, **kwargs: "x", scope=owner)
    reg._track_mcp_tool_server("mcp__x__t", "x")
    reg._track_mcp_tool_server("mcp__x__t2", "x")
    try:
        assert registry.snapshot_registration("mcp__x__t2", scope="p2") is not None

        reg.reapply_tool_filters({"x": after}, scope="p1")

        assert registry.snapshot_registration("mcp__x__t2", scope="p1") is None
        assert registry.snapshot_registration("mcp__x__t2", scope="p2") is not None
        assert registry.snapshot_registration("mcp__x__t", scope="p1") is not None
        assert profile.core._mcp_tool_server_names.get("mcp__x__t2") == "x"
        assert "mcp__x__t2" in server._registered_tool_names, \
            "teardown walks this list; dropping a tool another profile still exposes leaves it behind"
    finally:
        with profile.core._lock:
            profile.core._servers.pop(key, None)
            profile.core._server_scope_keys.pop(key, None)
            profile.core._server_tool_scopes.pop(key, None)
        for owner in ("p1", "p2"):
            for tool_name in ("mcp__x__t", "mcp__x__t2"):
                registry.deregister(tool_name, scope=owner)
