"""Private encoder-pilot calculations; launchers supply their dependencies at call time."""


def targets(probe_state, dt):
    hand, obj = probe_state[:, :, 0:3].double(), probe_state[:, :, 4:7].double()
    dt = dt.double()[:, None]
    return {'hand_pos': hand[:, 2], 'obj_pos': obj[:, 2], 'rel_pos': obj[:, 2] - hand[:, 2],
            'hand_vel': (hand[:, 2] - hand[:, 0]) / (2 * dt), 'obj_vel': (obj[:, 2] - obj[:, 0]) / (2 * dt)}


def mlp_probe(x, y, groups, seed=0, device='cuda', *, torch, F, math):
    """Fit Linear-GELU-Linear; return the fitted values for the launcher-owned predictor."""
    gen = torch.Generator().manual_seed(seed)
    eps = sorted(set(groups))
    order = torch.randperm(len(eps), generator=gen).tolist()
    held = {eps[i] for i in order[:max(1, round(0.2 * len(eps)))]}
    hold = torch.tensor([g in held for g in groups])
    x, y = x.float(), y.float()
    xm, xs = x[~hold].mean(0), x[~hold].std(0).clamp_min(1e-6)
    ym, ys = y[~hold].mean(0), y[~hold].std(0).clamp_min(1e-6)
    xt, yt = ((x - xm) / xs).to(device), ((y - ym) / ys).to(device)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, y.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tr, va = (~hold).nonzero().flatten().to(device), hold.nonzero().flatten().to(device)
    best, best_state, wait = math.inf, None, 0
    for epoch in range(300):  # noqa: B007 - epoch is reported after early stopping
        net.train()
        perm = tr[torch.randperm(len(tr), generator=gen).to(device)]
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            loss = F.mse_loss(net(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(F.mse_loss(net(xt[va]), yt[va]))
        if val < best - 1e-6:
            best, best_state, wait = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 20:
                break
    net.load_state_dict(best_state)
    net.eval()

    return (net, xm, xs, ym, ys), {'epochs': epoch + 1, 'best_val_mse_std': best, 'held_back_episodes': len(held)}


def probe_report(fit_x, fit_y, fit_groups, sets, device, *, TARGETS, ridge_fit, ridge_predict, mlp_probe, r2_score):
    out = {'ridge': {}, 'mlp': {}, 'mlp_fit': {}}
    for name in TARGETS:
        W, m, s = ridge_fit(fit_x, fit_y[name])
        ridge = lambda xq, W=W, m=m, s=s: ridge_predict(W, m, s, xq)  # noqa: E731
        mlp, info = mlp_probe(fit_x, fit_y[name], fit_groups, device=device)
        out['mlp_fit'][name] = info
        for kind, predict in (('ridge', ridge), ('mlp', mlp)):
            for g, (xs, ys, ph) in sets.items():
                pred, true = predict(xs), ys[name].double()
                err = (pred - true).norm(dim=-1) * 1000.0  # mm (positions) or mm/s (velocities)
                rec = {'r2': r2_score(pred, true), 'err_mm': float(err.mean()),
                       'spread_mm': float((true - true.mean(0)).norm(dim=-1).mean() * 1000.0), 'n': len(true)}
                if name == 'rel_pos':
                    for label, sel in (('approach', ph), ('contact', ~ph)):
                        rec[f'err_mm_{label}'] = float(err[sel].mean()) if sel.any() else None
                        rec[f'spread_mm_{label}'] = float((true[sel] - true[sel].mean(0)).norm(dim=-1).mean() * 1000.0) \
                            if sel.any() else None
                        rec[f'n_{label}'] = int(sel.sum())
                out[kind][f'{name}/{g}'] = rec
    return out


def episode_windows(path, episodes, *, h5py, np, robot_state):
    """Every stride-2 window of each episode: rgb (L,6,H,W,3) per episode and the (L,4) robot states."""
    with h5py.File(path, 'r') as f:
        train_cams = np.where(f['cameras']['is_train'][:].astype(bool))[0]
        V = len(train_cams)
        for ep in episodes:
            g = f['episodes'][ep]
            yield ep, g['rgb'][:, :V], robot_state(g['obs'][:, :4], 'metaworld')


def within_episode(arms, path, episodes, device, *, torch, np, F, math, episode_windows, HardNegativeSampler,
                   DEFAULTS, STRIDE, MIN_GAP, MAX_GAP, hardneg_metrics):
    """Within-episode retrieval (gaps 6..40) and hn metrics with identical sampler draws for every arm."""
    sums = {a.name: {'within_top1': [], 'hn': []} for a in arms}
    gen = torch.Generator().manual_seed(0)
    draws_total = 0
    for ep, rgb, robot in episode_windows(path, episodes):
        L = len(rgb)
        starts = np.arange(0, L - 2 * STRIDE)
        sampler = HardNegativeSampler({ep: robot}, **DEFAULTS)
        negs = [sampler.sample(ep, int(t0), STRIDE, gen) for t0 in starts]
        draws_total += len(negs)
        frames = torch.from_numpy(rgb).permute(0, 1, 4, 2, 3)  # (L,V,3,H,W)
        for a in arms:
            states = []
            for i in range(0, len(starts), 64):
                t0 = torch.from_numpy(starts[i:i + 64])
                idx = torch.stack([t0, t0 + STRIDE, t0 + 2 * STRIDE], 1)  # (b,3)
                states.append(a.states(frames[idx]))  # (b,3,V,3,H,W) -> (b,V,D)
            S = F.normalize(torch.cat(states).float(), dim=-1)  # (N,V,D), window index == t0
            N, V = S.shape[:2]
            gap = (torch.arange(N)[:, None] - torch.arange(N)[None]).abs()
            allowed = (gap >= MIN_GAP) & (gap <= MAX_GAP)
            allowed.fill_diagonal_(True)
            correct = []
            for ca in range(V):
                for cb in range(V):
                    if ca == cb:
                        continue
                    sim = (S[:, ca] @ S[:, cb].T).masked_fill(~allowed, -math.inf)
                    correct.append(sim.argmax(1) == torch.arange(N))
            sums[a.name]['within_top1'].append(torch.stack(correct).float())  # (pairs, N)
            valid = torch.tensor([n is not None for n in negs])
            neg_idx = torch.tensor([n if n is not None else 0 for n in negs])
            m = hardneg_metrics(S, S[neg_idx], valid, torch.zeros(N))
            sums[a.name]['hn'].append({k: float(v) for k, v in m.items() if k != 'hardneg_hand_disp_mean'} | {'n': N})
    out = {}
    for a in arms:
        top1 = torch.cat(sums[a.name]['within_top1'], 1)
        hn = sums[a.name]['hn']
        n = sum(h['n'] for h in hn)
        out[a.name] = {'within_episode_top1': float(top1.mean()), 'within_windows': int(top1.shape[1]),
                       **{k: sum(h[k] * h['n'] for h in hn if not math.isnan(h[k])) /
                          max(1, sum(h['n'] for h in hn if not math.isnan(h[k])))
                          for k in ('hardneg_cosine', 'hardneg_within_episode_top1', 'hardneg_gallery_top1',
                                    'hardneg_valid_fraction')}, 'hn_windows': n}
    return out, draws_total
