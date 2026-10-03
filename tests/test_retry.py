import requests
from src import common
def test_retry(tmp_path, monkeypatch):
    calls={"n":0}
    def flaky(url,dest,tmp,method,**k):
        calls["n"]+=1
        if calls["n"]<3: raise requests.ConnectionError("reset")
        dest.write_text("ok")
    monkeypatch.setattr(common,"_fetch",flaky); monkeypatch.setattr(common.time,"sleep",lambda s:None)
    common.download("http://x", tmp_path/"f.tsv")
    assert calls["n"]==3 and (tmp_path/"f.tsv").read_text()=="ok"
