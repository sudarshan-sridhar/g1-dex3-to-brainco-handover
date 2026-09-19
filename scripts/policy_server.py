r"""GR00T policy server: loads one fine-tuned checkpoint on a GPU machine and serves action
chunks to the Isaac Lab client (scripts/handover_rollout.py) over TCP.

WHY NOT `gr00t/eval/run_gr00t_server.py`
    Its wire format is a msgpack dialect that lives inside the `gr00t` package, so the
    simulator side would need `gr00t` installed next to the torch build Isaac Sim pins.
    Here both ends are plain: length-prefixed pickle over one TCP socket, and the
    simulator client needs only numpy.

TOPOLOGY
    Simulation and inference can run on different machines. In the reported runs the
    simulator ran on a workstation and the policy on a cluster GPU node, joined by an
    SSH port forward, which keeps the port local on both ends:

        ssh -N -L 5601:<gpu-node>:5601 <cluster-login-host>

REQUEST   {"state": {name: (1,D) float32}, "video": {name: (1,H,W,3) uint8},
           "text": str}
RESPONSE  {"action": (horizon, D) float32}  or  {"error": "..."}

The D columns come back in the modality config's action order, concatenated over
modality_keys, which is the same layout the scripted demos used, so the client can
feed it straight back into the env. D is whatever the loaded config declares:
Dex3 28 = arms(14)|hands(14);
BrainCo 26 = arms(14)|hands(12).
"""

import argparse
import pickle
import socket
import struct
import sys
import traceback

import numpy as np


def recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    (length,) = struct.unpack(">Q", recv_exactly(sock, 8))
    return pickle.loads(recv_exactly(sock, length))


def send_msg(sock, obj):
    payload = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">Q", len(payload)) + payload)


def build_policy(args):
    """Load the finetuned checkpoint as a local policy.

    The modality config has to be registered here too, exactly as in training: a
    NEW_EMBODIMENT tag has no entry in the built-in MODALITY_CONFIGS registry, so
    without this import the policy cannot resolve its own action layout.
    """
    import importlib
    from pathlib import Path

    # Only NEW_EMBODIMENT needs a config registered from a .py. The base
    # checkpoint's PRETRAIN tags (REAL_G1, XDOF, the R1 Pro family) carry their own
    # modality configs inside experiment_cfg/config.yaml, so for those the flag is
    # omitted and the policy supplies them.
    if args.modality_config_path:
        cfg_path = Path(args.modality_config_path)
        if not (cfg_path.exists() and cfg_path.suffix == ".py"):
            raise SystemExit(f"modality config must be an existing .py: {cfg_path}")
        sys.path.append(str(cfg_path.parent))
        importlib.import_module(cfg_path.stem)
        print(f"registered modality config: {cfg_path}", flush=True)
    else:
        print("no modality config supplied; expecting a built-in tag", flush=True)

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    policy = Gr00tPolicy(model_path=args.model_path, embodiment_tag=tag,
                         device=args.device)
    print(f"loaded {args.model_path} as {tag.value} on {args.device}", flush=True)
    return policy


def main():
    ap = argparse.ArgumentParser()
    # --checkpoint / --modality-config are accepted as aliases.
    ap.add_argument("--model-path", "--checkpoint", dest="model_path", required=True)
    ap.add_argument("--modality-config-path", "--modality-config",
                    dest="modality_config_path", default=None)
    ap.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback on purpose; reach it through the ssh forward")
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()

    policy = build_policy(args)

    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag

    tag_value = EmbodimentTag.resolve(args.embodiment_tag).value
    if tag_value in MODALITY_CONFIGS:
        modality_configs = MODALITY_CONFIGS[tag_value]
    else:
        # A pretrained tag whose config lives in the checkpoint rather than the
        # repo registry. REAL_G1 is exactly this case: `real_g1_relative_eef_...`
        # is absent from gr00t/configs/data/embodiment_configs.py but present in
        # experiment_cfg/config.yaml, which is why the tag looks unusable until
        # you ask the loaded policy instead of the registry.
        for attr in ("modality_configs", "modality_config", "_modality_configs"):
            got = getattr(policy, attr, None)
            if isinstance(got, dict) and got:
                modality_configs = got.get(tag_value, got)
                break
        else:
            raise SystemExit(
                f"no modality config for {tag_value}: not in MODALITY_CONFIGS "
                f"({sorted(MODALITY_CONFIGS)}) and the policy exposes none")
    print(f"modality keys: {sorted(modality_configs)}", flush=True)
    action_keys = modality_configs["action"].modality_keys

    # GPU nodes can be shared, so the requested port may already be taken by another
    # job. Walk forward until a port is free and announce which one was bound.
    srv = None
    for port in range(args.port, args.port + 20):
        cand = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cand.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            cand.bind((args.host, port))
        except OSError as exc:
            cand.close()
            print(f"port {port} unavailable ({exc.errno}), trying next", flush=True)
            continue
        cand.listen(1)
        srv, args.port = cand, port
        break
    if srv is None:
        raise SystemExit(f"no free port in {args.port}..{args.port + 19}")
    print(f"POLICY_SERVER_READY {args.host}:{args.port}", flush=True)

    while True:
        conn, addr = srv.accept()
        print(f"client connected: {addr}", flush=True)
        n = 0
        try:
            while True:
                req = recv_msg(conn)
                if req.get("op") == "ping":
                    send_msg(conn, {"pong": True, "action_keys": list(action_keys)})
                    continue
                if req.get("op") == "bye":
                    break
                try:
                    obs = {}
                    for k, v in req["state"].items():
                        obs[f"state.{k}"] = np.asarray(v, dtype=np.float32)
                    for k, v in req["video"].items():
                        obs[f"video.{k}"] = np.asarray(v, dtype=np.uint8)
                    for lang_key in modality_configs["language"].modality_keys:
                        obs[lang_key] = req["text"]

                    parsed = parse_observation_gr00t(obs, modality_configs)
                    raw_chunk, _ = policy.get_action(parsed)

                    # parse_action_gr00t is NOT in gr00t.data.utils -- it is a
                    # four-line local helper defined inside open_loop_eval.py
                    # (line 131). Importing it from the library fails with
                    # ImportError. It only unbatches and prefixes, so inline it.
                    chunk = {f"action.{k}": v[0] for k, v in raw_chunk.items()}
                    horizon = len(np.atleast_1d(chunk[f"action.{action_keys[0]}"]))
                    stacked = np.stack([
                        np.concatenate(
                            [np.atleast_1d(np.atleast_1d(chunk[f"action.{k}"])[j])
                             for k in action_keys], axis=0)
                        for j in range(horizon)
                    ]).astype(np.float32)
                    send_msg(conn, {"action": stacked})
                    n += 1
                    if n % 20 == 1:
                        print(f"  served {n} chunks, shape {stacked.shape}", flush=True)
                except Exception:
                    send_msg(conn, {"error": traceback.format_exc()})
        except (ConnectionError, EOFError):
            print("client disconnected", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
