"""Pool of MuJoCo worker processes stepping evaluation environments in lockstep.

Workers are spawned (the parent has CUDA initialised), inherit the parent's single allowed GPU UUID and
run the MuJoCo EGL guard before creating environments. The parent batches all observations through one
policy forward pass per step.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _worker(conn, task: str, seed: int, count: int, env_kwargs: dict) -> None:
    from _bootstrap import guard_mujoco  # scripts/ is on the inherited sys.path

    guard_mujoco()
    from s4d.rl.env import MetaWorldCameraEnv

    first = MetaWorldCameraEnv(task, seed, **env_kwargs)  # one renderer per worker process (GPU memory)
    envs = [first] + [MetaWorldCameraEnv(task, seed, renderer=first.renderer, **env_kwargs) for _ in range(count - 1)]
    conn.send("ready")
    while True:
        command, payload = conn.recv()
        if command == "reset":
            conn.send([None if job is None else envs[i].reset(*job) for i, job in enumerate(payload)])
        elif command == "step":
            out = []
            for i, action in enumerate(payload):
                if action is None:
                    out.append(None)
                    continue
                obs, proprio, reward, done, info = envs[i].step(action)
                out.append((obs, proprio, reward, done, info["success"]))
            conn.send(out)
        else:
            for env in reversed(envs):  # the owner of the shared renderer closes last
                env.close()
            conn.send("closed")
            return


class EnvPool:
    def __init__(self, task: str, seed: int, workers: int, envs_per_worker: int, env_kwargs: dict):
        """All environments share the construction ``seed`` (it fixes the 50 MT1 configurations)."""
        ctx = mp.get_context("spawn")
        self.per_worker = int(envs_per_worker)
        self.conns, self.procs = [], []
        for _ in range(int(workers)):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, task, int(seed), self.per_worker, env_kwargs), daemon=True)
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)
        for conn in self.conns:
            if conn.recv() != "ready":
                raise RuntimeError("evaluation worker failed to start")

    @property
    def size(self) -> int:
        return len(self.conns) * self.per_worker

    def _broadcast(self, command: str, items: list) -> list:
        for w, conn in enumerate(self.conns):
            conn.send((command, items[w * self.per_worker : (w + 1) * self.per_worker]))
        return [result for conn in self.conns for result in conn.recv()]

    def run(self, act, episodes: list[tuple[list, int]]) -> tuple[list[float], list[float]]:
        """Roll out ``(camera path, reset seed)`` episodes; ``act(obs (N,3T,H,W) uint8, proprio (N,P))`` -> actions."""
        successes, returns = [0.0] * len(episodes), [0.0] * len(episodes)
        for start in range(0, len(episodes), self.size):
            wave = list(range(start, min(start + self.size, len(episodes))))
            jobs = [episodes[i] for i in wave] + [None] * (self.size - len(wave))
            state = self._broadcast("reset", jobs)
            obs = [None if s is None else s[0] for s in state]
            proprio = [None if s is None else s[1] for s in state]
            active = list(range(len(wave)))
            while active:
                actions = act(np.stack([obs[k] for k in active]), np.stack([proprio[k] for k in active]))
                commands = [None] * self.size
                for k, action in zip(active, actions):
                    commands[k] = action
                results = self._broadcast("step", commands)
                still = []
                for k in active:
                    obs[k], proprio[k], reward, done, success = results[k]
                    returns[wave[k]] += reward
                    successes[wave[k]] = max(successes[wave[k]], success)
                    if not done:
                        still.append(k)
                active = still
        return successes, returns

    def close(self) -> None:
        for conn in self.conns:
            conn.send(("close", None))
            conn.recv()
        for proc in self.procs:
            proc.join(timeout=30)
