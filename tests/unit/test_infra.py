"""Unit tests for infra: caches, cache keys, experiment store, runtimes, logs."""
import logging
from pathlib import Path

import pytest

from infra.cache import cache_keys as ck
from infra.cache.in_memory_cache import InMemoryCache
from infra.cache.redis_cache import RedisCache
from infra.logging import recent_logs
from infra.logging.runtime.base import Runtime
from infra.logging.runtime.factory import get_runtime
from infra.logging.runtime.silent_runtime import SilentRuntime
from infra.logging.runtime.simple_runtime import SimpleRuntime
from infra.storage.experiment_store import ExperimentStore


# --------------------------------------------------------------------------- #
# cache_keys
# --------------------------------------------------------------------------- #
def test_normalize_handles_many_types(tmp_path):
    import dataclasses
    import datetime as dt

    @dataclasses.dataclass
    class _D:
        x: int

    f = tmp_path / "f.txt"
    f.write_text("x")
    payload = {
        "dc": _D(1),
        "path": f,
        "when": dt.date(2020, 1, 1),
        "nested": [{"b": 2, "a": 1}],
        "set": {3, 1, 2},
        "bytes": b"\x01\x02",
    }
    out = ck._normalize(payload)
    assert out["dc"] == {"x": 1}
    assert out["bytes"] == "0102"
    assert out["set"] == [1, 2, 3]


def test_normalize_model_dump_and_dict_and_object():
    class _Model:
        def model_dump(self):
            return {"m": 1}

    class _Dictish:
        def dict(self):
            return {"d": 2}

    class _Obj:
        def __init__(self):
            self.a = 1

    assert ck._normalize(_Model()) == {"m": 1}
    assert ck._normalize(_Dictish()) == {"d": 2}
    assert ck._normalize(_Obj()) == {"a": 1}
    assert ck._normalize(5) == 5


def test_stable_hash_and_make_cache_key():
    assert ck.stable_hash({"a": 1}) == ck.stable_hash({"a": 1})
    key = ck.make_cache_key("name space", "v1", "dev", "feat/x", {"q": 1})
    assert key.startswith("name_space:v1:dev:feat_x:")
    assert ck.text_hash("hi") == ck.text_hash("hi")


def test_file_signature_and_fingerprint(tmp_path):
    missing = ck.file_signature(tmp_path / "no.txt")
    assert missing["exists"] is False
    f = tmp_path / "f.txt"
    f.write_text("data")
    sig = ck.file_signature(f)
    assert sig["exists"] is True and sig["size"] == 4
    assert ck.fingerprint_files([f]) == ck.fingerprint_files([f])


# --------------------------------------------------------------------------- #
# InMemoryCache
# --------------------------------------------------------------------------- #
def test_in_memory_cache_basic_and_ttl():
    cache = InMemoryCache(max_entries=10, default_ttl_sec=None)
    cache.set("a", 1)
    assert cache.get("a") == 1 and cache.has("a") is True
    cache.delete("a")
    assert cache.get("a") is None and cache.has("a") is False
    # ttl <= 0 deletes / never stores
    cache.set("b", 2, ttl_sec=0)
    assert cache.get("b") is None


def test_in_memory_cache_expiry(monkeypatch):
    cache = InMemoryCache(max_entries=10, default_ttl_sec=5)
    t = {"v": 1000.0}
    monkeypatch.setattr(cache, "_now", lambda: t["v"])
    cache.set("k", "val")
    assert cache.get("k") == "val"
    t["v"] = 2000.0  # advance past ttl
    assert cache.get("k") is None
    assert cache.has("k") is False


def test_in_memory_cache_lru_eviction():
    cache = InMemoryCache(max_entries=2, default_ttl_sec=None)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # a becomes most-recent
    cache.set("c", 3)  # evicts b (lru)
    assert cache.has("a") and cache.has("c") and not cache.has("b")


def test_in_memory_cache_clear_prefix_and_validation():
    cache = InMemoryCache()
    cache.set("p:1", 1)
    cache.set("p:2", 2)
    cache.set("q:1", 3)
    assert cache.clear(prefix="p:") == 2
    assert cache.clear() == 1
    with pytest.raises(ValueError):
        InMemoryCache(max_entries=0)


# --------------------------------------------------------------------------- #
# RedisCache (fake client)
# --------------------------------------------------------------------------- #
class _FakeRedis:
    def __init__(self):
        self.kv = {}
        self.sets = {}

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, value, ex=None):
        self.kv[key] = value

    def delete(self, *keys):
        for k in keys:
            self.kv.pop(k, None)

    def exists(self, key):
        return 1 if key in self.kv else 0

    def sadd(self, key, *members):
        self.sets.setdefault(key, set()).update(members)

    def srem(self, key, *members):
        self.sets.setdefault(key, set()).difference_update(members)

    def smembers(self, key):
        return set(self.sets.get(key, set()))


def _redis():
    return RedisCache("redis://x", default_ttl_sec=100, key_prefix="rag:", redis_client=_FakeRedis())


def test_redis_cache_roundtrip_and_index():
    cache = _redis()
    cache.set("a", {"v": 1})
    assert cache.get("a") == {"v": 1}
    assert cache.has("a") is True
    cache.delete("a")
    assert cache.get("a") is None and cache.has("a") is False


def test_redis_cache_ttl_zero_and_no_ttl():
    cache = _redis()
    cache.set("z", 1, ttl_sec=0)  # deletes path
    assert cache.get("z") is None
    cache.set("n", 2, ttl_sec=None)
    assert cache.get("n") == 2


def test_redis_cache_encode_error():
    cache = _redis()
    with pytest.raises(TypeError):
        cache.set("bad", object())


def test_redis_cache_clear():
    cache = _redis()
    cache.set("p1", 1)
    cache.set("p2", 2)
    cache.set("other", 3)
    assert cache.clear(prefix="p") == 2
    assert cache.clear() == 1
    assert cache.clear() == 0  # nothing left


def test_redis_cache_default_client(monkeypatch):
    import infra.cache.redis_cache as rcmod

    created = {}

    class _Factory:
        @staticmethod
        def from_url(url):
            created["url"] = url
            return _FakeRedis()

    monkeypatch.setattr(rcmod.redis, "Redis", _Factory)
    RedisCache("redis://host:1/0")
    assert created["url"] == "redis://host:1/0"


# --------------------------------------------------------------------------- #
# runtimes
# --------------------------------------------------------------------------- #
def test_runtime_base_run_step_and_log():
    rt = Runtime()
    assert rt.run_step("s", lambda x: x + 1, 1) == 2
    rt.log("hello")  # routes to recent_logs
    with pytest.raises(NotImplementedError):
        rt.start("x")


def test_silent_and_simple_runtime(capsys):
    silent = SilentRuntime()
    silent.start("x")
    silent.add_step("s")
    assert silent.run_step("s", lambda: 7) == 7
    silent.log("nope")
    silent.stop()

    simple = SimpleRuntime()
    simple.start("go")
    assert simple.run_step("step", lambda: 5) == 5
    simple.stop("done")
    out = capsys.readouterr().out
    assert "Running: step" in out and "done" in out


def test_get_runtime_modes():
    from infra.logging.runtime.rich_runtime import RichRuntime

    assert isinstance(get_runtime({"runtime": {"mode": "api"}}), SilentRuntime)
    assert isinstance(get_runtime({"runtime": {"mode": "other"}}), SimpleRuntime)
    assert isinstance(get_runtime({}), RichRuntime)


# --------------------------------------------------------------------------- #
# recent_logs
# --------------------------------------------------------------------------- #
def test_recent_logs_ring_buffer_and_callback():
    recent_logs.clear_logs()
    fired = {"n": 0}
    recent_logs.set_refresh_callback(lambda: fired.__setitem__("n", fired["n"] + 1))
    for i in range(5):
        recent_logs.add_log(f"m{i}")
    logs = recent_logs.get_recent_logs()
    assert len(logs) == 3 and logs[-1] == "m4"  # maxlen=3
    assert fired["n"] == 5
    recent_logs.set_refresh_callback(None)


def test_recent_logs_handler_emits():
    recent_logs.clear_logs()
    handler = recent_logs.RecentLogsHandler()
    record = logging.LogRecord("n", logging.WARNING, __file__, 1, "msg", None, None)
    handler.emit(record)
    assert any("WARNING" in entry for entry in recent_logs.get_recent_logs())


# --------------------------------------------------------------------------- #
# ExperimentStore
# --------------------------------------------------------------------------- #
def test_experiment_store_roundtrip(tmp_path, monkeypatch):
    import infra.storage.experiment_store as es

    monkeypatch.setattr(es, "_git_commit", lambda: "abc123")
    store = ExperimentStore(root=tmp_path)
    run_dir = store.create_run({"name": "exp"})
    assert store.load_manifest(run_dir)["git_commit"] == "abc123"

    result = {
        "variant": "v1",
        "pipeline": "simple",
        "workspace_id": "ws",
        "records": [{"question": "q", "answer": "a"}, {"question": "q2", "error": "boom"}],
        "config_snapshot": {"k": 1},
    }
    store.write_variant_runs(run_dir, result)
    store.write_variant_metrics(run_dir, "v1", {"recall": {"value": 0.5}})
    store.write_comparison(run_dir, {"matrix": []})

    assert store.list_variants(run_dir) == ["v1"]
    assert len(store.load_runs(run_dir, "v1")) == 2
    assert store.load_metrics(run_dir, "v1")["recall"]["value"] == 0.5
    summary = es._git_commit  # noqa: F841


def test_experiment_store_missing_paths(tmp_path):
    store = ExperimentStore(root=tmp_path)
    missing = tmp_path / "nope"
    assert store.list_variants(missing) == []
    assert store.load_runs(missing, "v") == []
    assert store.load_metrics(missing, "v") == {}
    assert store.load_manifest(missing) == {}


def test_git_commit_handles_failure(monkeypatch):
    import infra.storage.experiment_store as es

    def _boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(es.subprocess, "run", _boom)
    assert es._git_commit() is None
