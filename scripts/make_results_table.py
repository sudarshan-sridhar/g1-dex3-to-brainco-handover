"""Build the results table, the per-trial CSV and the plots from the evaluation outputs.

Reads results/stage<S>/<configuration>/results_<S>.json written by scripts/handover_rollout.py
(one folder per configuration, see scripts/evaluate_stage.sh) and the open-loop replay summary
of Stage B. Writes:

    results/results_table.md      one row per stage and configuration (metrics M1-M6)
    results/trials.csv            one row per trial
    results/plots/*.png           success per configuration, where trials stop

usage: python scripts/make_results_table.py
"""
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGES = [("A", "Original Dex-hand task (Dex3)"),
          ("B", "Brainco retargeting only"),
          ("C", "Brainco retargeting + fine-tuning")]
CONFIGS = [("stick_30x240", "Square stick 3.0 x 24 cm"),
           ("stick_30x220", "Square stick 3.0 x 22 cm"),
           ("stick_35x240", "Square stick 3.5 x 24 cm"),
           ("heldout_object_round", "Held-out object: round stick 3.0 x 24 cm"),
           ("heldout_initial_pose", "Held-out initial pose")]
FAILURE = {"success": "success", "no_grasp": "no grasp",
           "drop_before_transfer": "dropped before handover", "no_transfer": "handover not completed",
           "drop_after_transfer": "dropped after handover", "no_place": "not placed",
           "not_released": "not released cleanly", "placed_but_propped_or_held": "not released cleanly",
           "moved_after_place": "moved after placing", "final_check_failed": "moved after placing",
           "fall": "robot fall"}
REPLAY = "results/stageB/open_loop_replay/handover_summary.json"


def tracking_error(folder):
    """M2: commanded joint target at step t vs measured joint position at t+1 (rad)."""
    ea, eh = [], []
    for a_path in glob.glob(os.path.join(folder, "*_actions.npy")):
        s_path = a_path.replace("_actions.npy", "_states.npy")
        if not os.path.isfile(s_path):
            continue
        A, S = np.load(a_path), np.load(s_path)
        n = min(len(A), len(S)) - 1
        if n > 0:
            ea.append(np.abs(A[:n, :14] - S[1:n + 1, :14]).mean())
            eh.append(np.abs(A[:n, 14:] - S[1:n + 1, 14:]).mean())
    return (float(np.mean(ea)) if ea else None, float(np.mean(eh)) if eh else None)


def load(stage, cfg):
    folder = f"results/stage{stage}/{cfg}"
    p = os.path.join(folder, f"results_{stage}.json")
    if not os.path.isfile(p):
        return None, []
    s = json.load(open(p))
    c = dict(s["per_config"][0])
    eps = s.get("episodes") or []
    c["place_rate"] = sum(1 for e in eps if e.get("t_place") is not None) / max(len(eps), 1)
    fails = {}
    for e in eps:
        k = FAILURE.get(e.get("failure"), e.get("failure"))
        fails[k] = fails.get(k, 0) + 1
    worst = {k: v for k, v in fails.items() if k != "success"}
    c["dominant_failure"] = max(worst.items(), key=lambda kv: kv[1])[0] if worst else "none"
    c["arm_err"], c["hand_err"] = tracking_error(folder)
    return c, eps


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    os.makedirs("results/plots", exist_ok=True)
    table, rows, trials = {}, [], []
    rows.append("| Stage | Configuration | Trials | M1 success | M1 grasp / handover / place rate | "
                "M2 joint tracking error, arm / hand (rad) | M3 drops / object impacts / joint-limit violations | "
                "M4 completion time (s) / control rate (Hz) | M6 dominant failure mode |")
    rows.append("|" + "---|" * 9)
    for st, st_name in STAGES:
        for cfg, label in CONFIGS:
            c, eps = load(st, cfg)
            table[(st, cfg)] = c
            if c is None:
                rows.append(f"| {st} | {label} | not run | | | | | | |")
                continue
            rows.append(
                f"| {st} | {label} | {c['trials']} | **{c['successes']}/{c['trials']}** | "
                f"{fmt(c.get('grasp_rate'))} / {fmt(c.get('transfer_rate'))} / {fmt(c['place_rate'])} | "
                f"{fmt(c['arm_err'], 3)} / {fmt(c['hand_err'], 3)} | "
                f"{c.get('drops')} / {c.get('contact_spikes')} / {c.get('joint_limit_violations')} | "
                f"{fmt(c.get('completion_s_mean'), 1)} / {fmt(c.get('control_hz_achieved_mean'), 1)} | "
                f"{c['dominant_failure']} |")
            for i, e in enumerate(eps):
                trials.append({
                    "stage": st, "stage_name": st_name, "configuration": label, "trial": i,
                    "success": int(bool(e.get("success"))),
                    "grasp_step": e.get("t_grasp"), "handover_step": e.get("t_transfer"),
                    "place_step": e.get("t_place"), "completion_s": e.get("completion_s"),
                    "failure_mode": FAILURE.get(e.get("failure"), e.get("failure")),
                    "dropped": int(bool(e.get("dropped"))), "object_impacts": e.get("contact_spikes"),
                    "joint_limit_violations": e.get("joint_limit_violations"),
                    "final_tilt_deg": e.get("final_tilt_deg"), "final_hand_clearance_m": e.get("final_hand_clear_m"),
                    "control_hz": e.get("control_hz_achieved")})
        if st == "B" and os.path.isfile(REPLAY):
            r = json.load(open(REPLAY))
            n_ok = sum(1 for e in r["per_episode"] if e.get("success"))
            rows.append(f"| B | Open-loop replay of Dex3 demonstrations (square stick 3.0 x 24 cm) | "
                        f"{len(r['per_episode'])} | **{n_ok}/{len(r['per_episode'])}** | | | | | no grasp |")
    open("results/results_table.md", "w", encoding="utf-8").write("\n".join(rows) + "\n")
    with open("results/trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trials[0].keys()) if trials else ["stage"])
        w.writeheader()
        w.writerows(trials)
    print("\n".join(rows))

    colors = {"A": "#3b6ea5", "B": "#c8553d", "C": "#2e8b57"}
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.26
    for i, (st, st_name) in enumerate(STAGES):
        vals = [100 * ((table[(st, c)] or {}).get("success_rate") or 0.0) for c, _ in CONFIGS]
        ax.bar([k + (i - 1) * w for k in range(len(CONFIGS))], vals, w, label=f"Stage {st}: {st_name}",
               color=colors[st])
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([lbl.replace(": ", ":\n") for _, lbl in CONFIGS], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("task success (%)")
    ax.set_title("Handover success per configuration (10 trials each)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/plots/success_by_configuration.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for st, st_name in STAGES:
        cs = [table[(st, c)] for c, _ in CONFIGS if table[(st, c)]]
        n = sum(c["trials"] for c in cs)
        if not n:
            continue
        g = sum((c.get("grasp_rate") or 0) * c["trials"] for c in cs) / n
        t = sum((c.get("transfer_rate") or 0) * c["trials"] for c in cs) / n
        p = sum(c["place_rate"] * c["trials"] for c in cs) / n
        s = sum(c["successes"] for c in cs) / n
        ax.plot(["grasp", "handover", "place", "success"], [100 * v for v in (g, t, p, s)], marker="o",
                label=f"Stage {st} (n={n})", color=colors[st])
    ax.set_ylim(0, 100)
    ax.set_ylabel("trials reaching the step (%)")
    ax.set_title("How far trials get, all configurations pooled")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/plots/task_progress_by_stage.png", dpi=150)
    print("wrote results/results_table.md, results/trials.csv, results/plots/")


if __name__ == "__main__":
    main()
