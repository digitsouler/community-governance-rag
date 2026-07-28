"""端到端测试 B 方案：跨角色 question_log 回忆注入"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"
USER_ID = "u_test_b_py_001"

def post(path, body, headers=None):
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"), headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

def get(path, headers=None):
    req = urllib.request.Request(BASE + path, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

print("=== 1) 新建会话 ===")
sid = post("/api/sessions", {}, headers={"X-User-Id": USER_ID})["session_id"]
print("session_id =", sid)
print()

print("=== 2) 居民身份问：楼上漏水 ===")
r2 = post("/api/chat", {
    "question": "楼上漏水把我家天花板和刚装修好的新房泡坏了",
    "session_id": sid, "provider": "deepseek", "user_role": "resident", "stream": False,
}, headers={"X-User-Id": USER_ID})
print("route =", r2.get("route"))
print("answer =", (r2.get("answer") or "")[:300])
print()

print("=== 3) 物业身份问：我刚才问了什么问题（应命中 recall） ===")
r3 = post("/api/chat", {
    "question": "我刚才问了什么问题",
    "session_id": sid, "provider": "deepseek", "user_role": "property", "stream": False,
}, headers={"X-User-Id": USER_ID})
print("route =", r3.get("route"))
print("answer =", (r3.get("answer") or "")[:600])
print()

print("=== 4) 居民身份问：之前我以物业身份问过什么（跨角色回忆） ===")
r4 = post("/api/chat", {
    "question": "我之前以其他身份问过什么问题？",
    "session_id": sid, "provider": "deepseek", "user_role": "resident", "stream": False,
}, headers={"X-User-Id": USER_ID})
print("route =", r4.get("route"))
print("answer =", (r4.get("answer") or "")[:600])
print()

print("=== 5) 案件档案按角色隔离测试：读 resident 角色档案 ===")
prof_r = get(f"/api/sessions/{sid}/profile?role=resident", headers={"X-User-Id": USER_ID})
print("resident profile =", json.dumps(prof_r.get("profile", {}), ensure_ascii=False)[:400])
print()

print("=== 6) 案件档案按角色隔离测试：读 property 角色档案（应为空，因为本次没以物业身份沉淀） ===")
prof_p = get(f"/api/sessions/{sid}/profile?role=property", headers={"X-User-Id": USER_ID})
print("property profile =", json.dumps(prof_p.get("profile", {}), ensure_ascii=False)[:400])
