"""Open-loop sanity check of a served policy against recorded demos (no simulator).

Sends the demo's own state + camera frames at a few timesteps to the policy bridge and compares
the predicted action chunk with the demo's recorded actions. A healthy policy tracks the demo
closely here; a large error at t=0 means an action/state layout or normalisation mismatch.

usage: python scripts/open_loop_check.py --dataset datasets_local/<name> --port 5601 [--episodes 30 40] [--steps 0 100 300]
"""
import argparse, os, pickle, socket, struct
import numpy as np, pandas as pd, imageio.v2 as iio

TASK = "hand the object from the right hand to the left hand and place it on the left"

def rpc(sock, obj):
    p = pickle.dumps(obj, protocol=4); sock.sendall(struct.pack(">Q", len(p)) + p)
    def rn(n):
        b = bytearray()
        while len(b) < n:
            c = sock.recv(n - len(b))
            if not c: raise ConnectionError("closed")
            b.extend(c)
        return bytes(b)
    n = struct.unpack(">Q", rn(8))[0]; return pickle.loads(rn(n))

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--port", type=int, default=5601)
ap.add_argument("--episodes", type=int, nargs="+", default=[30, 40])
ap.add_argument("--steps", type=int, nargs="+", default=[0, 100, 300])
ap.add_argument("--arms", type=int, default=14)
a = ap.parse_args()
sock = socket.create_connection(("127.0.0.1", a.port), timeout=300)
print("ping", rpc(sock, {"op": "ping"}))
for e in a.episodes:
    df = pd.read_parquet(os.path.join(a.dataset, "data", "chunk-000", f"episode_{e:06d}.parquet"))
    S = np.stack(df["observation.state"].to_numpy()); A = np.stack(df["action"].to_numpy())
    rd = {t: iio.get_reader(os.path.join(a.dataset, "videos", "chunk-000", f"observation.images.{t}", f"episode_{e:06d}.mp4"))
          for t in ("front", "ego_view")}
    for t in a.steps:
        if t + 16 > len(A): continue
        video = {k: r.get_data(t)[None].astype(np.uint8) for k, r in rd.items()}
        s = S[t].astype(np.float32)
        r = rpc(sock, {"op": "act", "state": {"arms": s[None, :a.arms], "hands": s[None, a.arms:]}, "video": video, "text": TASK})
        if "error" in r: print(r["error"][-800:]); raise SystemExit(1)
        P = np.asarray(r["action"]); H = min(len(P), 16)
        err = np.abs(P[:H] - A[t:t + H])
        print(f"ep{e} t={t} H={len(P)} arm_err mean={err[:, :a.arms].mean():.3f} max={err[:, :a.arms].max():.3f} "
              f"hand_err mean={err[:, a.arms:].mean():.3f} | pred0-state0 arm max={np.abs(P[0, :a.arms] - s[:a.arms]).max():.3f} "
              f"| demo a0-state0 arm max={np.abs(A[t, :a.arms] - s[:a.arms]).max():.3f}")
    for r in rd.values(): r.close()
try:
    rpc(sock, {"op": "bye"})
except (ConnectionError, OSError):
    pass
sock.close()
