import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

pytest.importorskip("numpy")      # building the test run uses the (numpy) scoring pipeline

from test_digits_scoring import DOC, FakeScorer, _wav  # noqa: E402
from indic_normalizer.digits.scoring import load_config  # noqa: E402
from indic_normalizer.digits.scoring.details import read_tsv  # noqa: E402
from indic_normalizer.digits.scoring.pipeline import verbalize_dir  # noqa: E402
from indic_normalizer.verifier import server as V  # noqa: E402


class Flat(FakeScorer):
    def score(self, pairs, head="tdt"):
        return [1.0 for _ in pairs]                    # every number a tie -> verify


@pytest.fixture()
def run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(V, "SETTINGS", str(tmp_path / "settings.json"))
    data, audio = tmp_path / "data", tmp_path / "audio"
    data.mkdir()
    audio.mkdir()
    for name in ("a", "b"):
        (data / f"{name}.json").write_text(json.dumps(DOC, ensure_ascii=False), encoding="utf-8")
    _wav(audio / "a.wav", 3.5)                         # b has no audio -> failed
    verbalize_dir(str(data), str(audio), Flat(), load_config(None, out_dir=str(tmp_path / "out")), log=lambda *_: None)
    return tmp_path / "out" / "data", audio


def test_verifier_flow(run):
    rd, audio = run
    st = {}
    s = V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    assert s["loaded"] and s["summary"]["verify"] == 1 and s["summary"]["failed"] == 1
    files = {f["file"]: f for f in V.api_files(st)["files"]}
    assert files["a"]["kind"] == "verify" and files["a"]["n_pending"] == 2 and files["a"]["has_audio"]
    assert files["b"]["kind"] == "failed" and files["b"]["stage"] == "find_audio" and not files["b"]["has_audio"]

    d = V.api_file(st, "a", "verify")
    assert [n["core"] for x in d["sentences"] for n in x["numbers"]] == ["25", "4"]
    n0 = d["sentences"][0]["numbers"][0]
    assert n0["model_pick"] == "ഇരുപത്തഞ്ച്" and n0["scored"] and n0["candidates"][0]["delta"] == 0
    s0, s1 = d["sentences"][0]["idx"], d["sentences"][1]["idx"]
    pv = V.api_preview(st, {"file": "a", "kind": "verify", "choices": {s0: ["ട്വന്റി ഫൈവ്"], s1: [None]}})["preview"]
    assert pv[s0] == "ഫീസ് ട്വന്റി ഫൈവ് ആണ്" and pv[s1] == "ശരി 4 പേര്"

    # partial: 25 decided, 4 left for later -> stays in verify/
    r = V.api_save(st, {"file": "a", "kind": "verify", "choices": {s0: ["ട്വന്റി ഫൈവ്"], s1: [None]}})
    assert r == {"ok": True, "file": "a", "decided": 1, "open": 1, "done": False, "typed": 0}
    part = json.load(open(rd / "verify" / "a.json", encoding="utf-8"))
    assert [x["text"] for x in part["sentences"]][:2] == ["ഫീസ് ട്വന്റി ഫൈവ് ആണ്", "ശരി 4 പേര്"]
    assert not (rd / "verbalized" / "a.json").exists()
    rows = read_tsv(str(rd / "verify.tsv"))
    assert [(x["number"], x["status"], x["final_pick"]) for x in rows] == [("25", "verified", "ട്വന്റി ഫൈവ്"), ("4", "pending", "")]
    d = V.api_file(st, "a", "verify")                                     # only the open one is shown
    assert [n["core"] for x in d["sentences"] for n in x["numbers"]] == ["4"] and d["info"]["decided"] == 1
    assert d["sentences"][0]["numbers"][0]["model_pick"] == "നാല്"

    # complete (custom reading) -> verbalized
    r = V.api_save(st, {"file": "a", "kind": "verify", "choices": {s1: ["നാല്  ആള്"]}})
    assert r["done"] and r["open"] == 0
    out = json.load(open(rd / "verbalized" / "a.json", encoding="utf-8"))
    assert out["sentences"][0]["text"] == "ഫീസ് ട്വന്റി ഫൈവ് ആണ്" and out["sentences"][1]["text"] == "ശരി നാല് ആള് പേര്"
    assert "ട്വന്റി" in [w["word"] for w in out["words"]] and "25" not in [w["word"] for w in out["words"]]
    assert not (rd / "verify" / "a.json").exists() and read_tsv(str(rd / "verify.tsv")) == []
    v = read_tsv(str(rd / "verbalized.tsv"))
    assert [(x["file"], x["status"], x["final_pick"], x["model_pick"]) for x in v] == [
        ("a", "verified", "ട്വന്റി ഫൈവ്", "ഇരുപത്തഞ്ച്"), ("a", "verified", "നാല് ആള്", "നാല്")]
    assert v[0]["verified_at"] and v[0]["scores"]
    with pytest.raises(ValueError):
        V.api_save(st, {"file": "a", "kind": "verify", "choices": {}})       # already done

    # failed: partly decided -> becomes a verify file; then completed
    d = V.api_file(st, "b", "failed")
    assert d["info"]["stage"] == "find_audio" and not d["sentences"][0]["numbers"][0]["scored"]
    b0, b1 = d["sentences"][0]["idx"], d["sentences"][1]["idx"]
    r = V.api_save(st, {"file": "b", "kind": "failed", "choices": {b0: ["ഇരുപത്തഞ്ച്"], b1: [None]}})
    assert not r["done"] and read_tsv(str(rd / "failed.tsv")) == [] and not (rd / "failed" / "b.json").exists()
    assert (rd / "verify" / "b.json").exists()
    assert [(x["number"], x["status"]) for x in read_tsv(str(rd / "verify.tsv"))] == [("25", "verified"), ("4", "pending")]
    assert [(f["file"], f["kind"]) for f in V.api_files(st)["files"]] == [("b", "verify"), ("a", "verbalized")]
    V.api_save(st, {"file": "b", "kind": "verify", "choices": {b1: ["നാല്"]}})
    assert {x["file"] for x in read_tsv(str(rd / "verbalized.tsv"))} == {"a", "b"}
    s = json.load(open(rd / "summary.json", encoding="utf-8"))
    assert (s["verbalized"], s["verify"], s["failed"], s["verified"]) == (2, 0, 0, 2)
    assert [f["kind"] for f in V.api_files(st)["files"]] == ["verbalized", "verbalized"]   # only finished ones left
    with pytest.raises(ValueError):
        V.api_file(st, "../x", "verify")


def test_verifier_http_and_audio_range(run):
    rd, audio = run
    srv = V.ThreadingHTTPServer(("127.0.0.1", 0), V.Handler)
    srv.state = {}
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"<h1>Verifier</h1>" in urllib.request.urlopen(base + "/").read()
        req = urllib.request.Request(base + "/api/open", json.dumps({"run_dir": str(rd), "audio_dir": str(audio)}).encode(),
                                     {"Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(req))["loaded"]
        r = urllib.request.urlopen(urllib.request.Request(base + "/api/audio?file=a", headers={"Range": "bytes=10-19"}))
        assert r.status == 206 and len(r.read()) == 10 and r.headers["Content-Range"].endswith(f"/{os.path.getsize(audio / 'a.wav')}")
        b = json.load(urllib.request.urlopen(base + "/api/browse?path=" + urllib.request.quote(str(rd.parent))))
        assert "data" in b["dirs"]
        g = json.load(urllib.request.urlopen(base + "/api/candidates?q=" + urllib.request.quote("55000 രൂപയുടെ")))
        assert g["groups"][0]["core"] == "55000"
        assert g["groups"][0]["candidates"][0]["phrase"] == "അമ്പത്തയ്യായിരം രൂപയുടെ"
        g = json.load(urllib.request.urlopen(base + "/api/candidates?style=glide&q=" + urllib.request.quote("55000-യുടെ")))
        assert g["groups"][0]["suffix_issue"] and g["groups"][0]["candidates"][0]["form"] == "glide"
        try:
            urllib.request.urlopen(base + "/api/candidates?q=abc")
            assert False
        except urllib.error.HTTPError as e:
            assert "no number" in json.load(e)["error"]
    finally:
        srv.shutdown()


def test_verifier_imports_without_numpy_or_torch():
    code = ("import sys; import indic_normalizer.verifier.server; "
            "print(sorted(m for m in ('numpy','torch','soundfile','yaml') if m in sys.modules))")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    assert out.strip() == "[]"


def _post(base, path, obj, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(base + path, json.dumps(obj, ensure_ascii=False).encode(), h)
    try:
        r = urllib.request.urlopen(req)
        return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_save_safety(run):
    rd, audio = run
    st = {}
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    d = V.api_file(st, "a", "verify")
    ids = [n["id"] for x in d["sentences"] for n in x["numbers"]]
    assert len(ids) == 2 and d["rev"]
    with pytest.raises(ValueError, match="digits"):                       # custom reading with digits
        V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {ids[0]: "25 രൂപ"}})
    r = V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {ids[0]: "ട്വന്റി ഫൈവ്", ids[1]: None}})
    assert r["open"] == 1
    with pytest.raises(V.Conflict):                                       # the same (stale) save again
        V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {ids[0]: "ട്വന്റി ഫൈവ്"}})
    d2 = V.api_file(st, "a", "verify")
    assert [n["id"] for x in d2["sentences"] for n in x["numbers"]] == [ids[1]]   # ids are stable
    with pytest.raises(ValueError, match="unknown number id"):
        V.api_save(st, {"file": "a", "kind": "verify", "rev": d2["rev"], "picks": {ids[0]: "x"}})
    rows = read_tsv(str(rd / "verify.tsv"))
    assert [(x["number"], x["status"]) for x in rows] == [("25", "verified"), ("4", "pending")]


def test_complete_save_survives_a_locked_copy(run, monkeypatch):
    rd, audio = run
    st = {}
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    d = V.api_file(st, "a", "verify")
    picks = {n["id"]: n["model_pick"] for x in d["sentences"] for n in x["numbers"]}
    real_remove, real_replace = os.remove, os.replace

    def locked_replace(a, b):                                            # the copy can't be moved ...
        if os.sep + "verify" + os.sep in str(a):
            raise PermissionError("locked")
        return real_replace(a, b)
    monkeypatch.setattr(V.os, "remove", lambda p: (_ for _ in ()).throw(PermissionError("locked")))
    monkeypatch.setattr(V.os, "replace", locked_replace)
    assert V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": picks})["done"]
    monkeypatch.setattr(V.os, "remove", real_remove)
    monkeypatch.setattr(V.os, "replace", real_replace)
    assert (rd / "verify" / "a.json").exists()                           # delete failed ...
    assert len(read_tsv(str(rd / "verbalized.tsv"))) == 2                # ... but nothing lost
    assert [f["file"] for f in V.api_files(st)["files"] if f["kind"] == "verify"] == []   # leftover cleaned
    assert not (rd / "verify" / "a.json").exists() and len(read_tsv(str(rd / "verbalized.tsv"))) == 2


def test_http_guards_and_ranges(run):
    rd, audio = run
    srv = V.ThreadingHTTPServer(("127.0.0.1", 0), V.Handler)
    srv.state = {}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        body = {"run_dir": str(rd), "audio_dir": str(audio)}
        assert _post(base, "/api/open", body, {"Origin": "http://evil.example"})[0] == 403
        assert _post(base, "/api/open", body, {"Content-Type": "text/plain"})[0] == 403
        assert _post(base, "/api/open", body, {"Host": "evil.example"})[0] == 403
        assert _post(base, "/api/open", body, {"Origin": base})[0] == 200
        size = os.path.getsize(audio / "a.wav")

        def get(rng):
            req = urllib.request.Request(base + "/api/audio?file=a", headers={"Range": rng})
            try:
                r = urllib.request.urlopen(req)
                return r.status, r.headers.get("Content-Range"), len(r.read())
            except urllib.error.HTTPError as e:
                return e.code, e.headers.get("Content-Range"), 0
        assert get(f"bytes={size}-") == (416, f"bytes */{size}", 0)
        assert get("bytes=600-100")[0] == 416 and get("bytes=-0")[0] == 416
        assert get("bytes=-10") == (206, f"bytes {size - 10}-{size - 1}/{size}", 10)
        assert get("bytes=0-1,5-6")[0] == 200                              # multi-range: whole file
    finally:
        srv.shutdown()


def test_edit_already_decided_numbers(run):
    rd, audio = run
    st = {}
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    d = V.api_file(st, "a", "verify")
    i25, i4 = [n["id"] for x in d["sentences"] for n in x["numbers"]]
    V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {i25: "ട്വന്റി ഫൈവ്", i4: None}})
    d = V.api_file(st, "a", "verify")
    assert d["can_edit_decided"] and [(x["id"], x["status"], x["final_pick"]) for x in d["decided"]] == \
        [(i25, "verified", "ട്വന്റി ഫൈവ്")]
    pv = V.api_preview(st, {"file": "a", "kind": "verify", "picks": {i4: "നാല്"}, "edits": {i25: "ഇരുപത്തഞ്ച്"}})
    assert pv["preview"][0] == "ഫീസ് ഇരുപത്തഞ്ച് ആണ്" and pv["preview"][1] == "ശരി നാല് പേര്"
    with pytest.raises(ValueError, match="digits"):
        V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "edits": {i25: "25"}})
    # change the decided one only: file stays in verify, 4 still open (digits)
    r = V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {}, "edits": {i25: "ഇരുപത്തഞ്ച്"}})
    assert r["open"] == 1 and not r["done"]
    part = json.load(open(rd / "verify" / "a.json", encoding="utf-8"))
    assert [x["text"] for x in part["sentences"]][:2] == ["ഫീസ് ഇരുപത്തഞ്ച് ആണ്", "ശരി 4 പേര്"]
    d = V.api_file(st, "a", "verify")
    assert [n["core"] for x in d["sentences"] for n in x["numbers"]] == ["4"]
    # edit again + finish
    r = V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {i4: "നാല്"},
                        "edits": {i25: "ഇരുപത്തിയഞ്ച്"}})
    assert r["done"]
    out = json.load(open(rd / "verbalized" / "a.json", encoding="utf-8"))
    assert [x["text"] for x in out["sentences"]][:2] == ["ഫീസ് ഇരുപത്തിയഞ്ച് ആണ്", "ശരി നാല് പേര്"]
    v = read_tsv(str(rd / "verbalized.tsv"))
    assert [(x["number"], x["status"], x["final_pick"]) for x in v] == [("25", "verified", "ഇരുപത്തിയഞ്ച്"), ("4", "verified", "നാല്")]


def test_input_copy_and_input_folder(run, tmp_path):
    rd, audio = run
    assert (rd / "input" / "a.json").exists()                      # verbalize keeps the input of verify files
    st = {}
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    d = V.api_file(st, "a", "verify")
    i25, i4 = [n["id"] for x in d["sentences"] for n in x["numbers"]]
    V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {i25: "ട്വന്റി ഫൈവ്", i4: None}})
    # run moved to another machine: no input/ copy, data_dir of summary.json does not exist here
    os.remove(rd / "input" / "a.json")
    s = json.load(open(rd / "summary.json", encoding="utf-8"))
    s["data_dir"] = "/nowhere/eval_frozen"
    (rd / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    assert not V.api_file(st, "a", "verify")["can_edit_decided"]
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio), "data_dir": str(tmp_path / "data")})
    d = V.api_file(st, "a", "verify")
    assert d["can_edit_decided"]
    r = V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {i4: "നാല്"},
                        "edits": {i25: "ഇരുപത്തഞ്ച്"}})
    assert r["done"] and json.load(open(rd / "verbalized" / "a.json", encoding="utf-8"))["sentences"][0]["text"] \
        == "ഫീസ് ഇരുപത്തഞ്ച് ആണ്"


def test_decisions_log(run):
    rd, audio = run
    st = {}
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    d = V.api_file(st, "a", "verify")
    i25, i4 = [n["id"] for x in d["sentences"] for n in x["numbers"]]
    model25 = d["sentences"][0]["numbers"][0]["model_pick"]
    V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {i25: model25, i4: "നാല് ആള്"}})
    log = read_tsv(str(rd / "verifier_decisions.tsv"))
    assert [(r["number"], r["action"], r["choice"]) for r in log] == [("25", "pick", "model_pick"), ("4", "pick", "typed")]
    assert log[0]["chosen_rank"] == "1" and log[0]["chosen_delta"] == "0.0" and log[0]["scores"]
    # failed file: picks logged too (no scores)
    d = V.api_file(st, "b", "failed")
    b0, b1 = d["sentences"][0]["idx"], d["sentences"][1]["idx"]
    V.api_save(st, {"file": "b", "kind": "failed", "rev": d["rev"], "choices": {b0: ["ട്വന്റി ഫൈവ്"], b1: [None]}})
    log = read_tsv(str(rd / "verifier_decisions.tsv"))
    assert (log[-1]["file"], log[-1]["kind"], log[-1]["choice"], log[-1]["chosen"]) == ("b", "failed", "candidate", "ട്വന്റി ഫൈവ്")
    # re-edit of a decided number: action edit, previous kept
    d = V.api_file(st, "b", "verify")
    b4 = [n["id"] for x in d["sentences"] for n in x["numbers"]][0]
    b25 = d["decided"][0]["id"]
    V.api_save(st, {"file": "b", "kind": "verify", "rev": d["rev"], "picks": {b4: "നാല്"}, "edits": {b25: "ഇരുപത്തഞ്ച്"}})
    log = read_tsv(str(rd / "verifier_decisions.tsv"))
    assert {(r["action"], r["chosen"], r["previous_pick"]) for r in log[-2:]} == \
        {("edit", "ഇരുപത്തഞ്ച്", "ട്വന്റി ഫൈവ്"), ("pick", "നാല്", "")}


def test_candidates_sorted_by_score(run):
    rd, audio = run
    st = {}
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    for x in V.api_file(st, "a", "verify")["sentences"]:
        for n in x["numbers"]:
            ds = [c["delta"] for c in n["candidates"] if c["delta"] is not None]
            assert ds == sorted(ds) and ds[0] == 0 and all(c["order"] for c in n["candidates"])


def test_superseded_copy_name_disappears_even_if_delete_is_refused(run, monkeypatch):
    rd, audio = run
    st = {}
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    d = V.api_file(st, "a", "verify")
    picks = {n["id"]: n["model_pick"] for x in d["sentences"] for n in x["numbers"]}
    real_remove = os.remove
    monkeypatch.setattr(V.os, "remove", lambda p: (_ for _ in ()).throw(PermissionError("open elsewhere")))
    assert V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": picks})["done"]
    assert not (rd / "verify" / "a.json").exists() and (rd / "verbalized" / "a.json").exists()
    assert len(os.listdir(rd / ".trash")) >= 1                           # parked, deleted later
    monkeypatch.setattr(V.os, "remove", real_remove)
    V.api_files(st)                                                      # listing empties the trash
    assert os.listdir(rd / ".trash") == []


def test_fix_run_repairs_leftovers_and_counts(run):
    from indic_normalizer.digits.scoring.fixrun import fix_run
    rd, audio = run
    st = {}
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    d = V.api_file(st, "a", "verify")
    picks = {n["id"]: n["model_pick"] for x in d["sentences"] for n in x["numbers"]}
    V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": picks})
    import shutil
    shutil.copyfile(rd / "verbalized" / "a.json", rd / "verify" / "a.json")      # a left-behind copy
    (rd / ".trash").mkdir(exist_ok=True)
    (rd / ".trash" / "x.json.1").write_text("{}")
    s = json.load(open(rd / "summary.json", encoding="utf-8"))
    s["verify"] = 7
    (rd / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    rep = fix_run(str(rd), dry_run=True, log=lambda *_: None)
    assert rep["leftover_copies_removed"] == ["verify/a.json"] and (rd / "verify" / "a.json").exists()
    rep = fix_run(str(rd), log=lambda *_: None)
    assert not (rd / "verify" / "a.json").exists() and os.listdir(rd / ".trash") == []
    s = json.load(open(rd / "summary.json", encoding="utf-8"))
    assert (s["verbalized"], s["verify"], s["failed"]) == (1, 0, 1)
    assert not any(v for k, v in fix_run(str(rd), log=lambda *_: None).items() if k != "summary")


def test_status_command(run, tmp_path):
    from indic_normalizer.digits.scoring.status import run_status
    rd, audio = run
    rep = run_status(str(rd), str(tmp_path / "data"))
    assert not rep["complete"] and rep["inputs"] == 2
    assert rep["missing"] == {"verify": ["a"], "failed": ["b"]} and rep["open_numbers"]["verify"] == 2
    st = {}
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    for stem, kind in (("a", "verify"), ("b", "failed")):
        d = V.api_file(st, stem, kind)
        picks = {n["id"]: (n["model_pick"] or n["candidates"][0]["text"]) for x in d["sentences"] for n in x["numbers"]}
        V.api_save(st, {"file": stem, "kind": kind, "rev": d["rev"], "picks": picks})
    rep = run_status(str(rd), str(tmp_path / "data"))
    assert rep["complete"] and rep["counts"]["verbalized"] == 2 and not rep["digits_left"]
    import json as _j
    p = rd / "verbalized" / "a.json"
    doc = _j.load(open(p, encoding="utf-8"))
    doc["sentences"][0]["text"] += " 7"
    p.write_text(_j.dumps(doc, ensure_ascii=False), encoding="utf-8")
    rep = run_status(str(rd), str(tmp_path / "data"))
    assert not rep["complete"] and list(rep["digits_left"]) == ["a"]


def test_status_files_without_numbers_are_fine(run, tmp_path):
    from indic_normalizer.digits.scoring.status import run_status
    rd, audio = run
    (tmp_path / "data" / "c.json").write_text(json.dumps({"sentences": [{"text": "ശരി"}]}, ensure_ascii=False),
                                              encoding="utf-8")
    (rd / "verbalized" / "c.json").write_text(json.dumps({"sentences": [{"text": "ശരി"}]}, ensure_ascii=False),
                                              encoding="utf-8")
    rep = run_status(str(rd), str(tmp_path / "data"))
    assert rep["verbalized_without_numbers"] == ["c"] and "c" not in rep["verbalized_without_rows"]


def test_edit_finished_file(run, tmp_path):
    rd, audio = run
    st = {}
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    d = V.api_file(st, "a", "verify")
    i25, i4 = [n["id"] for x in d["sentences"] for n in x["numbers"]]
    V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {i25: "ട്വന്റി ഫൈവ്", i4: "നാല്"}})
    assert (rd / "verbalized" / "a.json").exists() and not (rd / "verify" / "a.json").exists()

    files = {(f["file"], f["kind"]): f for f in V.api_files(st)["files"]}
    f = files[("a", "verbalized")]
    assert f["n_numbers"] == 2 and f["n_verified"] == 2 and f["n_pending"] == 0
    d = V.api_file(st, "a", "verbalized")
    assert d["sentences"] == [] and d["can_edit_decided"]
    assert [(x["id"], x["final_pick"]) for x in d["decided"]] == [(i25, "ട്വന്റി ഫൈവ്"), (i4, "നാല്")]
    with pytest.raises(ValueError, match="nothing changed"):
        V.api_save(st, {"file": "a", "kind": "verbalized", "rev": d["rev"]})
    pv = V.api_preview(st, {"file": "a", "kind": "verbalized", "edits": {i4: "ഫോർ"}})["preview"]
    assert pv[1] == "ശരി ഫോർ പേര്"

    r = V.api_save(st, {"file": "a", "kind": "verbalized", "rev": d["rev"], "edits": {i4: "ഫോർ"}})
    assert r["decided"] == 1 and r["done"]
    out = json.load(open(rd / "verbalized" / "a.json", encoding="utf-8"))
    assert [x["text"] for x in out["sentences"]][:2] == ["ഫീസ് ട്വന്റി ഫൈവ് ആണ്", "ശരി ഫോർ പേര്"]
    v = read_tsv(str(rd / "verbalized.tsv"))
    assert [(x["number"], x["status"], x["final_pick"]) for x in v if x["file"] == "a"] == \
        [("25", "verified", "ട്വന്റി ഫൈവ്"), ("4", "verified", "ഫോർ")]
    assert not (rd / "verify" / "a.json").exists() and not read_tsv(str(rd / "verify.tsv"))
    dec = read_tsv(str(rd / "verifier_decisions.tsv"))
    assert dec[-1]["kind"] == "verbalized" and dec[-1]["chosen"] == "ഫോർ" and dec[-1]["action"] == "edit"
    with pytest.raises(V.Conflict):                    # the old rev is stale now
        V.api_save(st, {"file": "a", "kind": "verbalized", "rev": d["rev"], "edits": {i4: "നാല്"}})

    # without the original input the finished file can only be viewed
    os.replace(tmp_path / "data" / "a.json", tmp_path / "a.json")
    d = V.api_file(st, "a", "verbalized")
    assert not d["can_edit_decided"]
    with pytest.raises(ValueError, match="original input"):
        V.api_save(st, {"file": "a", "kind": "verbalized", "rev": d["rev"], "edits": {i4: "നാല്"}})


def test_old_import_path_and_command_still_work():
    import indic_normalizer.digits.verifier.server as old
    from indic_normalizer.verifier.cli import main
    assert old is V
    with pytest.raises(SystemExit):
        main(["--help"])


def _sent(d, k):
    return next(s for s in d["all_sentences"] if s["idx"] == k)


def test_edit_sentences(run):
    from indic_normalizer.verifier.sentences import edits_for
    rd, audio = run
    st = {}
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    d = V.api_file(st, "a", "verify")
    assert d["can_edit_sentences"] and [s["idx"] for s in d["all_sentences"]] == [0, 1, 2]
    assert _sent(d, 0)["locked"] and _sent(d, 1)["locked"] and not _sent(d, 2)["locked"]
    with pytest.raises(ValueError, match="open numbers"):
        V.api_sentence(st, {"file": "a", "kind": "verify", "rev": d["rev"], "sentence": 0, "text": "ഫീസ് ആണ്"})
    with pytest.raises(ValueError, match="digits"):
        V.api_sentence(st, {"file": "a", "kind": "verify", "rev": d["rev"], "sentence": 2, "text": "ഓക്കേ 5"})
    with pytest.raises(ValueError, match="failed file"):
        V.api_sentence(st, {"file": "b", "kind": "failed", "sentence": 2, "text": "ഓക്കേ"})

    # sentence 2 (no numbers, no words in the word list): saved at once, normalized, words added
    r = V.api_sentence(st, {"file": "a", "kind": "verify", "rev": d["rev"], "sentence": 2, "text": " ഓക്കേ,  താങ്ക്സ്! "})
    assert r["text"] == "ഓക്കേ താങ്ക്സ്" and r["before"] == "ഓക്കേ" and r["rev"] != d["rev"]
    with pytest.raises(V.Conflict):                    # the page's old rev is stale now
        V.api_sentence(st, {"file": "a", "kind": "verify", "rev": d["rev"], "sentence": 2, "text": "ശരി"})
    with pytest.raises(ValueError, match="nothing changed"):
        V.api_sentence(st, {"file": "a", "kind": "verify", "rev": r["rev"], "sentence": 2, "text": "ഓക്കേ താങ്ക്സ്"})
    doc = json.load(open(rd / "verify" / "a.json", encoding="utf-8"))
    assert doc["sentences"][2]["text"] == "ഓക്കേ താങ്ക്സ്" and doc["transcription"].endswith("ഓക്കേ താങ്ക്സ്")
    new = [w for w in doc["words"] if w.get("edited")]
    assert [w["word"] for w in new] == ["ഓക്കേ", "താങ്ക്സ്"] and new[0]["start_time"] == 3.2 and new[-1]["end_time"] == 3.5
    assert doc["words"][-2:] == new                         # placed by time, after the earlier words
    assert edits_for(str(rd), "a")[2]["after"] == "ഓക്കേ താങ്ക്സ്"
    d = V.api_file(st, "a", "verify")                       # numbers unaffected
    assert [n["core"] for x in d["sentences"] for n in x["numbers"]] == ["25", "4"]
    assert _sent(d, 2)["edited"] and _sent(d, 2)["original"] == "ഓക്കേ"

    # decide 25; then sentence 0 can be edited (its words re-split over the sentence's word span)
    i25, i4 = [n["id"] for x in d["sentences"] for n in x["numbers"]]
    V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {i25: "ഇരുപത്തഞ്ച്", i4: None}})
    d = V.api_file(st, "a", "verify")
    assert not _sent(d, 0)["locked"] and _sent(d, 2)["text"] == "ഓക്കേ താങ്ക്സ്"   # the number save kept the edit
    r = V.api_sentence(st, {"file": "a", "kind": "verify", "rev": d["rev"], "sentence": 0,
                            "text": "ഫീസ് ഇരുപത്തഞ്ച് രൂപ ആണ്"})
    doc = json.load(open(rd / "verify" / "a.json", encoding="utf-8"))
    w0 = [w for w in doc["words"] if w["end_time"] <= 1.3]
    assert [w["word"] for w in w0] == ["ഫീസ്", "ഇരുപത്തഞ്ച്", "രൂപ", "ആണ്"]
    assert w0[0]["start_time"] == 0.0 and w0[-1]["end_time"] == 1.3 and all(w["edited"] for w in w0)
    d = V.api_file(st, "a", "verify")
    dec = {x["id"]: x for x in d["decided"]}
    assert dec[i25]["sentence_edited"]
    with pytest.raises(ValueError, match="edited by hand"):  # its number is changed in the sentence text
        V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {i4: "നാല്"},
                        "edits": {i25: "ട്വന്റി ഫൈവ്"}})

    # finish the file: the rebuild from the original keeps both hand edits
    r = V.api_save(st, {"file": "a", "kind": "verify", "rev": d["rev"], "picks": {i4: "നാല്"}})
    assert r["done"]
    out = json.load(open(rd / "verbalized" / "a.json", encoding="utf-8"))
    assert [x["text"] for x in out["sentences"]] == ["ഫീസ് ഇരുപത്തഞ്ച് രൂപ ആണ്", "ശരി നാല് പേര്", "ഓക്കേ താങ്ക്സ്"]

    # finished file: change the decided 4 (rebuilt from the input) -> the edited sentences stay
    d = V.api_file(st, "a", "verbalized")
    pv = V.api_preview(st, {"file": "a", "kind": "verbalized", "edits": {i4: "ഫോർ"}})["preview"]
    assert pv[0] == "ഫീസ് ഇരുപത്തഞ്ച് രൂപ ആണ്" and pv[1] == "ശരി ഫോർ പേര്"
    V.api_save(st, {"file": "a", "kind": "verbalized", "rev": d["rev"], "edits": {i4: "ഫോർ"}})
    out = json.load(open(rd / "verbalized" / "a.json", encoding="utf-8"))
    assert [x["text"] for x in out["sentences"]] == ["ഫീസ് ഇരുപത്തഞ്ച് രൂപ ആണ്", "ശരി ഫോർ പേര്", "ഓക്കേ താങ്ക്സ്"]
    assert "ഫോർ" in [w["word"] for w in out["words"]] and "താങ്ക്സ്" in [w["word"] for w in out["words"]]
    # and edit a finished file's sentence directly
    d = V.api_file(st, "a", "verbalized")
    V.api_sentence(st, {"file": "a", "kind": "verbalized", "rev": d["rev"], "sentence": 1, "text": "ശരി ഫോർ ആള്"})
    out = json.load(open(rd / "verbalized" / "a.json", encoding="utf-8"))
    assert out["sentences"][1]["text"] == "ശരി ഫോർ ആള്"
    log = read_tsv(str(rd / "sentence_edits.tsv"))
    assert [(x["sentence"], x["kind"], x["after"]) for x in log] == [
        ("2", "verify", "ഓക്കേ താങ്ക്സ്"), ("0", "verify", "ഫീസ് ഇരുപത്തഞ്ച് രൂപ ആണ്"), ("1", "verbalized", "ശരി ഫോർ ആള്")]


def test_fast_tsv_write_matches_write_tsv(tmp_path):
    from indic_normalizer.digits.scoring.details import NUMBER_COLS, write_tsv
    rows = [{"file": "a", "sentence": 0, "number": '5"', "candidates": "x\ty", "sentence_text": "a\nb", "status": None,
             "needs_review": True, "margin": 1.5},
            {"file": "b", "sentence": "1", "number": "ഇരുപത്", "final_pick": "", "candidates": "p | q"}]
    a, b = str(tmp_path / "a.tsv"), str(tmp_path / "b.tsv")
    write_tsv(a, NUMBER_COLS, rows)
    V._write_rows(b, NUMBER_COLS, rows)
    assert open(a, "rb").read() == open(b, "rb").read()
    assert V._rows(b) == read_tsv(b)                       # the cache holds what a fresh read gives
    again = V._rows(b, copy=False)                          # cached rows written back unchanged
    V._write_rows(a, NUMBER_COLS, list(again))
    assert open(a, "rb").read() == open(b, "rb").read()
    V._append_rows(b, NUMBER_COLS, [{"file": "c", "sentence": 2}])
    assert V._rows(b) == read_tsv(b) and [r["file"] for r in read_tsv(b)] == ["a", "b", "c"]
