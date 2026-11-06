"""Four-arm campaign (arm4): writes runs/setup/arm4-definition-20261106.json (registered before any result) from
arm4_common.py (arms, naming), the configuration audit runs/setup/arm4-config-audit-20261106.json (resolved diffs,
config hashes, references, drift), the smoke summary and the fixed analysis plan below. Re-running rewrites the file;
after publication the definition only changes through appended 'revisions' (with reasons).

    python -I arm4_write_definition_20261106.py
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm4_common as C  # noqa: E402

ROOT = C.ROOT
OUT = ROOT / 'runs/setup/arm4-definition-20261106.json'
audit = json.loads((ROOT / 'runs/setup/arm4-config-audit-20261106.json').read_text())
smoke_path = ROOT / 'runs/setup/arm4-smoke-20261106.json'
smoke = json.loads(smoke_path.read_text()) if smoke_path.is_file() else None
REVIEWS_PATH = ROOT / 'runs/setup/arm4-reviews-20261106.json'  # Codex rounds: findings, accepted / rejected, reasons
reviews = json.loads(REVIEWS_PATH.read_text()) if REVIEWS_PATH.is_file() else {'rounds': [], 'revisions': []}
eta_path = ROOT / 'runs/setup/arm4-eta-20261106.json'
eta = json.loads(eta_path.read_text()) if eta_path.is_file() else None


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


DRIFT = {
    'method': ['reference rows re-resolved under the release vs the runs\' stored config.yaml (all 16 reference '
               'pretrainings: no difference)',
               'AST comparison of literal code defaults (cfg.get / get(cfg, ...) defaults, dataclass field defaults, '
               'function argument defaults, module constants) in 21 training-path files at every reference commit vs '
               'the release',
               'bitwise cross-commit training-path fingerprints (same resolved config through each source tree: fixture '
               'window samples, initial parameters, 3 training steps with every loss term, gradient norm, parameters, '
               'RNG state; CPU, one thread, synthetic rasterizer) - runs/setup/tools/arm4_xcommit_fingerprint_20261106.py',
               'DrM: resolve_config of each new trainer row (reference export substituted) vs the stored reference DrM '
               'configs (16 of 16 identical); RL source drift listed per reference commit'],
    'findings': {
        'yaml': 'none: every reference row resolves to its stored configuration under the release',
        'code_defaults': {
            '226d8ae (Original, 6 tasks)': 'no changed or removed default; 104 entries added (flags of later variants, '
                                           'all off / default-preserving; covered by the bitwise fingerprints)',
            '10b3116 / 3fba521 (Original hammer / pick-place)': [
                "loss .get('motion_space') default 'image' -> 'gaussian': not effective - every Original-based row "
                "(the references and arms A / B) sets loss.motion_space=gaussian explicitly (the reference value is "
                "pinned), and the release rejects anything but 'gaussian'",
                's4d/data/contract.py REQUIRED batch keys lost motion3d / motion_weight (batch validation only; the '
                'Meta-World windows still carry both, and M3D consumes them)'],
            'fd5e950 / 7643e9d / cec4b00 (cst50)': 'none (training-path, export and RL sources byte-identical to the '
                                                    'release)'},
        'fingerprints': 'bitwise identical to the release: Original config at 226d8ae, 10b3116, 3fba521; cst50 config at '
                        'fd5e950, 7643e9d (sensitivity shown: a visibility margin 4 -> 5 px or depth-gradient levels 4 -> 3 '
                        'change the fingerprint). Scope (Codex arm4-a #5): bounded equivalence of the training path on a '
                        'CPU fixture for three steps with a synthetic rasterizer - not a proof for real kernels, bf16, '
                        'loader workers, real data, full-size Chamfer, scheduled evaluations or later steps',
        'gpu_step0_check': 'smoke (runs/setup/arm4-smoke-20261106.json): with the real rasterizer, real hammer data and '
                           'the release, every arm\'s step-0 evaluation reproduces its matched-initialisation reference '
                           'run: A / C / D vs cst50-seed0-hammer-200k (7643e9d, MIG) bitwise on all 155 deterministic '
                           'reconstruction metrics; B vs i2-m3d-seed0-hammer (10b3116, full GPU) within 2.7e-7 relative '
                           '(PSNR bitwise); only sampled diagnostics (M3D point sample, linear-probe R^2, effective rank) '
                           'differ',
        'random_stream': 'adding or removing the Chamfer term changes the random stream: its target sampling draws '
                         'before the M3D point sampling and shares the global generator, so the M3D sample and every '
                         'later draw (masks, drop-path, synthetic cameras, data-independent sampling of later steps) '
                         'diverge in distribution-equal ways; the restored depth-gradient / visibility terms and the '
                         'encoder swap consume no extra random numbers in the loss (Codex arm4-a #7)',
        'drm': 'resolved RL configs identical (all 8 tasks, both reference families). RL source: cst50\'s DrM code '
               '(fd5e950 / cec4b00) is byte-identical to the release; Original\'s DrM ran at 226d8ae and the release adds '
               'flag-gated paths (agent.random_shift, other algorithms, camera suite v2, multi-view / depth inputs, '
               'replay variants) whose defaults keep the DrM frozen-encoder path - reviewed (Codex arm4-a)',
        'not_covered_by_the_cpu_fingerprints': 'the gsplat rasterizer kernels (venv unchanged), bf16 autocast, data-loader '
                                                'workers and real data: only checked at step 0 on the GPU by the smoke '
                                                '(step-0 evaluation of each arm vs its matched-initialisation reference '
                                                'run) - not training-path equivalence beyond step 0',
        'chamfer_backend': 'local runs use the fused Triton nearest-neighbour kernel, the remote image the exact tiled '
                           'fallback (tests/test_chamfer_lean.py: equal indices, distances within 1e-5): cst50\'s five '
                           'remote tasks used the fallback, the new local arms B / C / D the fused kernel',
        'pins_added': 'none needed: the only drifted default (motion_space) is already pinned in every row'},
}
ROWS = {
    'count': {'pretraining': 32, 'export': 32, 'drm_trainer': 96, 'evaluator': 96, 'render_analysis': 48, 'total': 304},
    'template': 'each campaign row is the cst50 HAMMER row of the same kind (host local, local output roots, '
                'resources, max_restarts, stage, required results) with names substituted and, for pretraining, the '
                'arm\'s configuration (register_arm4_eight_task_20261106.py asserts every other field equal)',
    'order': 'registration (serialised) order: per (task, arm) chain, task-major (hammer A B C D, pick-place A B C D, '
             'bin-picking, coffee-push, door-open, peg-insert-side, peg-unplug-side, stick-push): pretraining, export, '
             '(DrM, evaluator) x 3; then the 48 render rows. Admission order among READY rows is the driver\'s sort: '
             '(1) evaluators whose trainer has already finished, (2) priority, (3) non-evaluators before evaluators, (4) '
             'registration order (enc- rows last); with the head-of-line reservation among non-evaluators. Hence a '
             'ready DrM trainer of a finished chain is offered before a pretraining of a later chain',
    'priorities': {'pretraining': 1, 'export': 0, 'drm_trainer': 1, 'evaluator': 1, 'render_analysis': 1},
    'render_priority_note': 'render rows at 1 and registered after every chain row, so they never hold the head-of-line '
                            'reservation ahead of a training row (Codex arm4-b #3)',
    'dependencies': 'pretraining -> export -> DrM trainers and evaluators (an evaluator is admitted only with its '
                    'trainer running or done; it polls snapshots); render rows of the arms -> their pretraining; render '
                    'rows of orig / cst50 have none (checkpoints exist)',
    'render_rows': 'ana-arm4-render-<model>-<task>, model in ocsw4, ochf, c50dg, c50vis, orig, cst50: '
                   'reports/results-artifact/src/run_arm4_render_20261106.py (unchanged chl_analysis_render.py protocol, '
                   'outputs reports/results-artifact/arm4/, done.json with checksums)',
    'pin_host': 'local on every row (pool_placement never moves them; the remote role admits no Meta-World row)',
}
ANALYSIS = {
    'principles': [
        'registered before any result of these arms; estimation with confidence intervals and descriptive readings',
        'objective: keep Original\'s control performance while improving reconstruction; no winner from PSNR (or any '
        'single reconstruction metric) alone',
        'nonsignificance is not equivalence: an interval that includes 0 is "direction unresolved", never "no '
        'difference" or "equivalent"; every reading prints "compatible with differences from L to U"',
        'per-task results (the three seed values per arm as dots, seed means, sign) are shown next to every 8-task '
        'aggregate; per-task intervals (3 vs 3 seeds) are uncalibrated and never carry control labels',
        'conditionality: "All primary intervals condition on the realised pretrained encoders (one pretraining seed per '
        'arm and task, and one per reference) and on the eight registered tasks. They quantify RL-seed variability for '
        'these encoders, not uncertainty from repeating pretraining; they are statements about these encoders, not '
        'established properties of the encoder design or loss term." Pretraining variation enters only the task-level '
        'sensitivity, confounded there with genuine task x arm heterogeneity; if the two disagree, the reading is '
        '"supported for these encoders, not established for the recipe"',
        'a comparison is final only when both sides have all 8 tasks x 3 seeds complete; earlier results are INTERIM, '
        'carry no reading and never trigger extra seeds or changes; nothing is imputed',
        'deviations from this plan are logged in the definition\'s revisions with reasons'],
    'comparisons': {
        'main_mechanism': {
            'A_vs_original': {'arm': 'ocsw4', 'reference': 'Original (s3-drm-splatter4d-<task>-s<seed>; method-frozen-v1 pretraining)',
                              'intended_change': 'the encoder (patch-local conv stem; width 256, depth 4, 4 heads)'},
            'B_vs_original': {'arm': 'ochf', 'reference': 'Original', 'intended_change': 'adding the Chamfer loss'},
            'C_vs_cst50': {'arm': 'c50dg', 'reference': 'cst50 (cst50-drm-<task>-s<seed>)', 'intended_change': 'restoring the depth-gradient loss'},
            'D_vs_cst50': {'arm': 'c50vis', 'reference': 'cst50', 'intended_change': 'restoring the visibility regulariser'},
            'caveat': 'each observed difference also contains the particular pretrained realisations and the recorded '
                      'execution-environment differences (references: Original trained and evaluated on the remote at '
                      '226d8ae with 3 x 8 evaluators; cst50 five tasks remote, three local; arms: local MIG, release '
                      'b5ab9a6 + routing); A vs Original is the least matched (different architecture, so no shared '
                      'initialisation), B / C / D share initialisation and data order with their reference'},
        'replacement_decision_family': {
            'contrasts': ['A_vs_original', 'B_vs_original', 'C_vs_original', 'D_vs_original'],
            'why': 'the decision is whether an arm can replace Original, so every candidate is read against Original; '
                   'C / D vs cst50 explain the mechanism but do not decide candidacy'},
        'secondary_descriptive': {
            'A_vs_B': 'host-free contrast (both local, same release): a common host offset against Original cancels',
            'A_vs_cst50': 'loss recipe on the same CSW4 + 50 % tube model (Original losses vs chamfer_lean), identical '
                          'initial parameters; evaluator layouts differ (3 x 8 vs 6 x 8)',
            'B_vs_chl': 'chamfer_lean (chl) = B without hard depth, depth gradient and visibility (original encoder)',
            'C_vs_D': 'which restored term matters more on cst50 (host-free)',
            'host_matched_subset': 'C / D vs cst50 on hammer, pick-place, bin-picking (cst50 trained locally there)',
            'context': 'chl, csw4 (enc-csw4), cst75 and the pretrained baselines; contextual only'}},
    'rl': {
        'data': 'runs/<id>-eval/eval.jsonl; a run counts only when final.json says completed at 1M agent steps AND the '
                'evaluations are exactly the 100-point 10k..1M grid (chlhm_rl.load_checked); the last record per step',
        'per_evaluation': 'success = fraction of successful episodes averaged over the suite\'s cameras; return = '
                          'per-episode reward sum averaged over the camera\'s 20 episodes, then over cameras',
        'suites': {'train': 'train0-5', 'heldout': 'eval0-3 (held-out static cameras)',
                   'moving': 'traj_lateral + traj_circular (the registered in-training moving-camera protocol, 1x)'},
        'normalisation': 'nR = (R - random) / (expert - random), per-task references of '
                         'reports/results-artifact/campaigns/meta.json norm_refs (no clipping)',
        'summaries': {'auc': 'arithmetic mean of the 100 registered evaluations 10k..1M (the full-budget normalised area '
                             'under the learning curve)',
                      'last5': 'mean of the evaluations at 960k..1M', 'final': 'the evaluation at 1M',
                      'curves': 'success and nR per suite, per task and 8-task mean, on the shared 10k grid (matched '
                                'windows), seed means with whole-curve seed-bootstrap bands'},
        'primary': {
            'endpoint': 'training-camera (train0-5) success AUC over the full 1M budget, arm - reference, equal-task mean '
                        'over the 8 tasks of per-task seed means; the same primary for every comparison',
            'why': 'success is the task outcome the control objective is about; the full-budget AUC uses the whole learning '
                   'trajectory and avoids selecting a favourable window (its variance advantage over last-5 is not '
                   'guaranteed); it is the endpoint pre-registered for cst50 vs cst75, keeping the contrasts commensurable '
                   'with the existing ranking. nR depends on reward shaping and normalisation (first secondary); last-5 / '
                   'final measure only the end of training; held-out / moving measure robustness (decision guards)',
            'floor_ceiling': 'tasks where both arms\' seed-mean success AUC is < 0.05 or > 0.95 are flagged in the per-task '
                             'table (little information); they stay in the aggregate'},
        'intervals': {
            'primary_display': 'the established task-stratified percentile bootstrap, 95 %: B = 10000, within each task '
                               'each arm\'s available seeds resampled with replacement (cell size n_t,arm, 3 or 2), cell '
                               'means, then the equal-task mean; arms independent; numpy default_rng(20261106). '
                               'Implementation must handle ragged cells (the existing rectangular seed_draws does not): '
                               'resample each cell with its own n and average tasks equally; tested on synthetic ragged '
                               'input before any result is read',
            'small_sample_sensitivity': 'Welch-Satterthwaite interval for the same estimand: SE^2 = (1/64) sum_t '
                                        '(s_A,t^2/n_A,t + s_R,t^2/n_R,t), nu by Welch-Satterthwaite, d +/- t_(nu,0.975) SE '
                                        '(the percentile bootstrap with 3 values per cell has variance (n-1)/n = 2/3 of '
                                        'the unbiased plug-in, so it undercovers: ~86-89 % actual for nominal 95 %)',
            'task_level_sensitivity': 'paired task-level interval d_bar +/- t_(7,0.975) s_d / sqrt(8) over the 8 per-task '
                                      'differences, the exact sign-flip test (256 patterns, chlhm_rl.sign_flip_p) and the '
                                      'sign count; it reflects task x arm heterogeneity including realised pretraining '
                                      'variation; tasks were selected, not sampled',
            'leave_one_task_out': 'range of the primary estimate leaving out each task; flagged if any omission changes a '
                                  'reading',
            'precision': 'expected (if seed variability resembles cst50 vs cst75, bootstrap SE ~2.1 pp): primary bootstrap '
                         'half-width ~4 pp, Welch ~5-6 pp, task-level ~6-9 pp; per-task 3 vs 3 ~10-20 pp '
                         '(uninformative). With a true difference of 0 the margin reading is reached roughly half the '
                         'time at 95 % and less often at the 98.75 % decision level; "unresolved" and "not excluded" '
                         'will be common and are reported as such',
            'multiplicity': 'mechanism family (4 main comparisons): unadjusted 95 % intervals are the main display and do '
                            'not give simultaneous coverage; Bonferroni 98.75 % intervals are shown alongside. Replacement '
                            'decision family (4 contrasts vs Original): a candidacy reading must hold on the Bonferroni '
                            '98.75 % bootstrap interval AND on the 98.75 % Welch interval; guards are vetoes (they can only '
                            'remove candidates) and need no adjustment; secondary endpoints are descriptive'},
        'control_reading': {
            'direction': {'gain supported': 'interval lower bound > 0', 'loss supported': 'interval upper bound < 0',
                          'direction unresolved': 'interval includes 0'},
            'margin': {'delta': 5.0, 'unit': 'percentage points of training success AUC (8-task equal-task mean)',
                       'reading': 'lower bound > -delta: "no loss larger than delta (descriptive)"; otherwise "a loss larger '
                                  'than delta is not excluded"',
                       'why_tolerable': 'the user\'s objective is to keep Original\'s control while improving '
                                        'reconstruction: a mean loss under 5 pp of success AUC is below the run-to-run '
                                        'resolution of this design (expected primary half-width ~4 pp) and about half the '
                                        'smallest loss this campaign has already treated as clearly supported (cst75 - cst50 '
                                        '-9.4 pp [-13.5, -5.2]); it is a descriptive benchmark, not a power-based '
                                        'non-inferiority margin and never an equivalence bound; an 8-task mean tolerance '
                                        'allows larger single-task losses, which the per-task table shows'},
            'candidacy_condition': 'the margin reading on the primary ("no loss larger than delta"; a small supported loss '
                                   'within delta still qualifies) at the decision-family level, robust to the Welch '
                                   'sensitivity (a reading that disappears under it is "method-sensitive" and does not '
                                   'qualify)',
            'wording': 'e.g. "the conditional interval\'s lower bound exceeds the descriptive -5 pp benchmark; this does '
                       'not establish equivalence or robustness to another pretraining seed"'},
        'guards': {
            'heldout': 'held-out static success AUC vs the same reference',
            'moving': 'moving-camera success AUC vs the same reference',
            'veto': 'conservative veto, chosen explicitly: an arm is not a candidate if a guard\'s 95 % primary-method '
                    'interval shows a supported loss that may exceed delta (upper bound < 0 AND lower bound <= -delta). '
                    'This is NOT the claim "a loss larger than delta is supported" (that would need upper bound < -delta); '
                    'an unresolved guard interval (including 0) does not veto. Guards only remove candidates, so they are '
                    'not multiplicity-adjusted'},
        'secondary': ['training nR AUC, last-5, final', 'training success last-5 and final',
                      'moving-camera and held-out success and nR (AUC, last-5)',
                      'return drop per run: training nR - moving nR and training nR - held-out nR (AUC and last-5), arm '
                      'difference with interval',
                      'early learning: success AUC differences up to 250k and 500k (the primary machinery); first step '
                      'at which the SEED-MEAN training success curve reaches 0.5 / 0.9 (first crossing, censored at 1M; '
                      'labelled as a property of the seed-mean curve, seed-bootstrap interval with 2000 draws as '
                      'chlhm_rl.crossing_ci); per-task steps-to-threshold, 8-task summary as a restricted mean censored at 1M',
                      'IQM over runs of training success AUC (rliable convention) with the same stratified resampling - a '
                      'different estimand, not a robustness check of the equal-task mean'],
        'pretrained_baseline_rankings': {
            'method': 'work/arch_compare ranking.py (sha256 in analysis_inputs_sha256): HEADLINE8 criteria (training nR '
                      'and success, held-out success, moving nR; each last-5 and AUC), per-task seed means, equal-task '
                      'means, one joint seed resample per draw applied to all criteria, B = 10000; ranks by point estimate; '
                      'ties = equal point estimates share the mean rank',
            'set': 'ours: full (Original), ocsw4, ochf, c50dg, c50vis, cst50, cst75, chl, chlrm, chlhm, cs, w4, csw4; '
                   'pretrained baselines: SinCro, ReViWo, DynaMo, 3D-MVP A (multi-view RGB-D), 3D-MVP B (single RGB-D), '
                   'SNeRL; end-to-end CNN (with / without shift) as non-pretrained context; all DrM, 240-episode protocol',
            'reading': 'P(rank 1) / P(top 3) are conditional bootstrap rank frequencies (they inherit the n = 3 '
                       'overconfidence), not probabilities of being best; a host column is shown; rankings are '
                       'descriptive and never a decision input'},
        'missing_and_failures': {
            'infrastructure': 'interruptions (OOM, MIG / host faults, crashes) resume the same run from its own latest '
                              'checkpoint under the registered max_restarts (pretraining --resume auto; DrM latest.pt); '
                              'every attempt is logged in the registry; a rerun decision is made without looking at '
                              'evaluations; the first completed attempt counts',
            'algorithmic': 'divergence is an outcome, never a reason to reseed: a DrM run that diverges but still '
                           'completes 1M steps with the full evaluation grid counts with its evaluated performance; a '
                           'pretraining without a usable encoder is reported as an arm failure on that task',
            'failed_run': 'a run that does not complete 1M steps and the 100-point grid within its registered restarts '
                          '(algorithmic or exhausted infrastructure retries) is FAILED: its partial curve is shown '
                          'descriptively, its cell enters the primary with the remaining seeds (ragged n), and the arm '
                          'receives NO candidacy reading for that comparison (failures can only block candidacy); a '
                          'relaunch is allowed only for an infrastructure cause, with the same seed, decided before '
                          'looking at evaluations, and is logged as a deviation',
            'small_cells': 'cell = (arm or reference, task); every cell\'s actual count of complete seeds is reported. '
                           'n >= 2: enters the primary and the Welch sensitivity with its n. n <= 1, or no usable encoder '
                           '(pretraining failure): the task is excluded from that comparison\'s aggregate (equal-task '
                           'mean over the remaining k tasks, the excluded task and its partial runs listed), the task-level '
                           'sensitivity uses k tasks (df k - 1), and the affected arm receives no candidacy reading for '
                           'that comparison; if k < 8 the comparison is reported as "k-task" and never as the 8-task result',
            'missing_evaluation': 'regenerated from the stored snapshot with the same reset seeds (eval_rl.py on that '
                                  'snapshot), the difference to any earlier record logged',
            'duplicates': 'the last record per evaluation step (as load_checked)',
            'cst50_stick_push_s2002': 'if its evaluator is still missing when the last arm run completes, the 8-task '
                                      'analysis with that cell at n = 2 becomes final and the 7-task analysis (without '
                                      'stick-push) is the sensitivity; the two are different estimands and agreement is '
                                      'not proof that the missingness is harmless'}},
    'reconstruction': {
        'checkpoints': 'step 200000 of every arm\'s pretraining and of the references (Original: the method-frozen-v1 '
                       'pretraining the Original DrM exports came from; cst50: its pretraining) - matched step; '
                       'reconstruction is host-free (one render code path, the same local checkout for every model)',
        'render_pipeline': {
            'rows': 'ana-arm4-render-<model>-<task> (48): the unchanged chl_analysis_render.py protocol, the registered 40 '
                    'validation samples (10 episodes x 4) x 22 target cameras per task (input camera c, frames t-4, t-2, '
                    't; rendered frame t), the existing ground truth, row index and segmentation (render_inputs_sha256); '
                    'every model, including orig and cst50, rendered by the same code into reports/results-artifact/arm4/',
            'metrics': 'chl200_analysis_metrics.py definitions: PSNR of the pooled squared error (full image and regions: '
                       'dynamic = robot + objects; static near < 0.9 m, mid 0.9-1.3 m, distant background >= 1.3 m), SSIM '
                       'and LPIPS (AlexNet v0.1) of the full image, LPIPS on 64 x 64 crops centred on the dynamic region and '
                       'the distant background (images with >= 16 region pixels; the same crops for every model), alpha '
                       'coverage per region (fraction with alpha < 0.5), median |expected depth - ground-truth depth| per '
                       'region',
            'camera_groups': 'training views = input camera + novel training cameras (the six training cameras); '
                             'held-out; moving range; all 22 - reported separately',
            'geometry': '3D Chamfer at frame t between all decoded centres (and the static group) and the fused '
                        'training-camera depth (1 cm voxels): mean squared distance both directions, medians, fraction '
                        'within 2 cm',
            'gaussian_diagnostics': 'Gaussian classes over the six training cameras (outside / low-opacity / hidden / '
                                    'contributing), opacity and largest-scale distributions, scale caps / floors, '
                                    'utilisation (weight share of the top 10 %)',
            'conditional_interval': 'paired on identical images; resampling unit = the validation EPISODE (all its '
                                    'samples and cameras together, 10 clusters per task), PSNR recomputed from the pooled '
                                    'squared error in each resample; per task B = 1000, 8-task mean task-stratified '
                                    'B = 10000. Wording: "Render-pipeline intervals condition on one pretraining run per '
                                    'model and task and on the registered validation episodes; they quantify evaluation-set '
                                    'sampling, not training variability, and support statements about these checkpoints '
                                    'only."',
            'decision_interval': 'the paired task-level t interval over the 8 per-task differences (df 7) with the sign '
                                 'count - a task-level, checkpoint-conditional reading that does not separate pretraining '
                                 'variation from task heterogeneity; the conditional episode interval is displayed next '
                                 'to it. Level: 98.75 % (Bonferroni over the four replacement candidates) for the PSNR '
                                 'and geometry decision endpoints; 95 % for the dynamic and held-out guards (they only '
                                 'remove candidates); the AND across endpoints within an arm needs no adjustment',
            'pipeline_check': 'orig_200k must reproduce chl/render/<task>__ref200.npz: images bitwise, alpha and depth '
                              'within 1e-4 (bitwise on hammer in the smoke). Enforced: the orig render row exits non-zero '
                              'without done.json when the check fails or the reference is missing (record in '
                              'runs/<row>/reproduction.json), so the row fails and the reconstruction analysis stops '
                              'pending a documented resolution'},
        'in_training_eval': 'eval/step_0200000/summary.json (@s2 RL spacing, @s6 largest stride): metric/psnr, psnr_static, '
                            'psnr_moving, ssim (+ static / moving), depth_absrel, epe_moving_mm_01 / _12 (moving-point EPE: '
                            'the motion error), held-out PSNR; per-task differences, task-level t interval (df 7) and sign '
                            'count; loss/* values are recipe-specific and not compared',
        'decision_endpoints': {
            'psnr': 'render pipeline, training-view group, full-image PSNR of the pooled squared error per task; equal-task '
                    'mean of the dB differences',
            'geometry': 'render pipeline, HELD-OUT camera group, full-image median |expected depth - ground-truth depth| '
                        '(m; lower is better) per task: no arm optimises depth on held-out cameras, and every arm and '
                        'reference optimises training-camera expected depth (depth_l1) identically, so this endpoint is '
                        'not constructed by any arm\'s loss; Chamfer (optimised by B, C, D and cst50, not by Original / A) '
                        'is supporting only (construction-confounded: favours Chamfer arms vs Original); in-training '
                        'depth_absrel supporting',
            'dynamic_guard': 'render pipeline, training-view dynamic-region PSNR',
            'heldout_guard': 'render pipeline, held-out full-image PSNR (training-view gains through over-fitting)',
            'smallest_effects': {'psnr_gain_db': 0.3, 'geometry_relative_reduction': 0.10, 'dynamic_loss_margin_db': 0.3,
                                 'heldout_loss_margin_db': 0.3},
            'geometry_relative_definition': 'per task r_t = (reference_t - arm_t) / reference_t of the held-out median '
                                            'depth error; the decision uses the equal-task mean of r_t and its task-level '
                                            'interval (threshold 0.10 on the mean)'},
        'reading': {
            'reconstruction improved (whole scene)': 'decision interval of training-view PSNR excludes 0 favouring the arm '
                                                     'with point estimate >= 0.3 dB, AND held-out depth error reduction with '
                                                     'interval excluding 0 and point >= 10 % relative, AND dynamic-region '
                                                     'PSNR lower bound > -0.3 dB AND held-out PSNR lower bound > -0.3 dB',
            'reconstruction improved (task-relevant)': 'the whole-scene reading AND a training-view dynamic-region PSNR '
                                                       'gain whose 98.75 % task-level interval excludes 0; dynamic crop '
                                                       'LPIPS is supporting only (no OR)',
            'dynamic preservation unresolved': 'the other whole-scene criteria hold but the dynamic-region lower bound '
                                               '<= -0.3 dB: does not pass the gate',
            'otherwise': 'reported descriptively; a gain carried by the background or static geometry is never called '
                         'task-relevant'}},
    'cost': {'training': 'pretraining s/it (median train/step_time), GPU time per step (train/gpu_time), wall-clock and '
                         'GPU-hours from the registry, peak GPU memory, with MIG co-residence and host load during the run '
                         '(shared 1g.24gb instances confound speed); DrM trainer wall-clock and agent steps/s',
             'inference': 'encoder parameters and GFLOPs per frame (FlopCounterMode, the BASELINES.md counter); frozen '
                          'policy-encoder latency per step at batch 1 on one local MIG instance (common hardware)',
             'use': 'reported separately; never folded into the control or reconstruction readings; only the common-'
                    'hardware latency breaks ties'},
    'decision_framing': {
        'candidate': 'an arm is a replacement candidate only if, vs Original: candidacy_condition on the primary, no '
                     'guard veto, and "reconstruction improved (whole scene)" (task-relevant stated separately)',
        'mechanism': 'C / D vs cst50 and A vs Original / B vs Original describe what each change does; C / D vs Original '
                     'decide candidacy',
        'several_candidates': 'compared head to head on the primary with host-free contrasts (A vs B, C vs D, or local '
                              'arm vs local arm), then on training-view PSNR (task-level interval), then common-hardware '
                              'latency; a tie = the interval includes 0',
        'none': '"no replacement recommended; Original retained; evidence insufficient at n = 3 RL seeds and one '
                'pretraining seed" - never "Original is better" unless a loss is supported',
        'timing': 'the recommendation is made only when all four arms and both references are final',
        'statement': 'every recommendation states n = 3 RL seeds per cell (2 in any registered missing or failed '
                     'cell), one pretraining seed per arm-task, and the host / code / evaluator-layout differences of the '
                     'references'},
    'analysis_inputs_sha256': {
        'reports/results-artifact/src/cst_rl.py': '15aa2ccf46ba534e3f3bde8f45dc2ffeae2afab5d25480e49c58dee1bbfa3827',
        'reports/results-artifact/src/chlhm_rl.py': '54495bcadd85f84a09f6abe6af1bb181796bb791b239dd0d7855699af1003788',
        'reports/results-artifact/src/variants_rl.py': '8c056aac5a8b3b9f2b0405ad56baa20e06225bc4d95525bdb20bed04cda84bb2',
        'reports/results-artifact/src/chl200_analysis_metrics.py': '1cb5d9f1da4a47a3b8ea17f7e2dcd7631bb6db08f18f875db2c9820ead79144c',
        'reports/results-artifact/src/chl_analysis_render.py': '5d72981df62c122f16e95fd9f4dc97b318afce86fa704c37aee028bed83fcf2a',
        'reports/results-artifact/src/render_compare.py': '7ee72bd08a56a18288b3320e630ae80905b28e3c2e96104ab327d0da9f5cbc23',
        'reports/results-artifact/campaigns/meta.json': 'ecf70746ef1c1450202148f474080d518c536c9085487aab3fb37f45c5ea8e48',
        '/home/ws/ws/presentation-assets-tools/work/arch_compare/scripts/ranking.py': '69b3d1b7935ee16e88e2cf5cf21540d4c77c90e94194ade52678c38d6982334e',
        '/home/ws/ws/presentation-assets-tools/work/arch_compare/scripts/rl_common.py': 'b6e4a2867377d19370bc66fc7491fd8b27df734bc17cd30ba81107fd4ec202ce',
        'note': 'the arm4 analysis scripts (RL with ragged cells + Welch + task-level; render metrics on arm4/) are written '
                'from these definitions before any arm result is read; changes to the listed files after registration '
                'are deviations'},
}
SCHEDULING = {
    'host': 'LOCAL only: the 8 authorized MIG 1g.24gb instances (GPU-76871c16 / GPU-d09f0338 slots); per instance '
            'max_jobs 3, max_mem_gb 22 (experiments/queue.yaml limits); RAM: admission ceiling 150 GB - 10 GB transient '
            '(memory_budget.py option (b)), steady target 130 GB, test-suite reservation via '
            'runs/setup/local-validation-reservation.json',
    'routing': 'ops/campaign/baseline_local_driver.py REGISTRATIONS gains arm4-registration-eight-task-20261106.json '
               '(branch arm4-20261106). experiments/HOLD keeps the default-queue daemon from admitting; the scoped local '
               'driver admits its registrations through jobs.tick on a staged queue (its HOLD check looks beside the '
               'staged file, as for cst / chlhm); server roles admit Meta-World locally; admission pauses do not match '
               'these prefixes',
    'priority': 'below the authorized closeout (cst50-drm-stick-push-s2002-eval: priority 1, an evaluator of a finished '
                'trainer - offered first - and an earlier registration; it is remote under the drain exception, so it '
                'does not compete for local capacity), above every other unstarted local row of lower priority (all '
                'remaining local rows of the driver\'s registrations are paused FlashSAC / SNeRL rows at priority 2); '
                'the driver\'s existing order rules stay (evaluators of finished trainers first, enc- rows last, the '
                'head-of-line reservation among ready non-evaluators); no existing row changes priority',
    'backfill': 'capacity-based: ready rows are admitted job by job while they fit, subject to the driver\'s existing '
                'head-of-line reservation (the first ready non-evaluator that does not fit stops later non-evaluators; '
                'evaluators still fill); rows waiting on dependencies never block; no two-task gate',
    'footprints_measured': {'pretraining': '10.77 GB (campaign rule, max sampled PSS; smoke peaks below)',
                            'drm_trainer': '3.5 GB', 'evaluator': '4.5 GB', 'export': '4 GB declared (short)',
                            'render_row': '12 GB declared (short, inference)'},
}

arms = {}
for arm, spec in C.ARMS.items():
    files, sets = C.pretrain_stack(arm, '<task>')
    arms[arm] = {
        'prefix': spec['prefix'], 'label': spec['label'], 'variant': spec['variant'],
        'derived_from': spec['base'] + ' (the reference runs\' stored resolved configuration)',
        'config_stack': files, 'overrides': sets,
        'intended_resolved_difference': {k: {'reference': v[0], 'arm': v[1]} for k, v in spec['diff'].items()},
        'hammer_pick_place_note': ('the Original hammer / pick-place runs also store train.stop_step 200000 (0-100k + '
                                   '100k-200k resume under one 200k schedule); the new rows omit it: end step '
                                   'min(train.steps, stop_step) = 200000 either way') if spec['base'] == 'original' else None,
        'matched_initialisation_reference': C.MATCHED_INIT[arm],
        'run_ids': {'pretrain': f'{spec["prefix"]}-seed0-<task>-200k', 'export': f'{spec["prefix"]}-export-seed0-<task>-200k',
                    'drm': f'{spec["prefix"]}-drm-<task>-s<seed>', 'evaluator': f'{spec["prefix"]}-drm-<task>-s<seed>-eval',
                    'render': f'ana-arm4-render-{spec["prefix"]}-<task>'},
    }
refs = {}
for task in C.TASKS:
    r = audit['references'][task]
    refs[task] = {
        'original': {'pretraining_rows': r['original']['rows'], 'run_dir': r['original']['run_dir'],
                     'checkpoint_200k': r['original']['checkpoint_200k'], 'export': r['original']['export'],
                     'drm': [C.reference_drm('original', task, s) for s in C.SEEDS],
                     'commits': sorted({x['commit'] for v in r['original']['launches'].values() for x in v}),
                     'config_sha256': r['original']['stored_config_sha256']},
        'cst50': {'pretraining_row': r['cst50']['rows'][0], 'run_dir': r['cst50']['run_dir'],
                  'checkpoint_200k': r['cst50']['checkpoint_200k'], 'export': r['cst50']['export'],
                  'drm': [C.reference_drm('cst50', task, s) for s in C.SEEDS],
                  'commits': sorted({x['commit'] for x in r['cst50']['launches']}),
                  'config_sha256': r['cst50']['stored_config_sha256']},
        'chl_context': {'drm': [f'chl-drm-{task}-s{s}' for s in C.SEEDS]},
    }
config_sha = {C.ids(arm, task)['pretrain']: audit['arms'][arm]['per_task'][task]['config_sha256']
              for arm in C.ARMS for task in C.TASKS}
render_inputs = {}
for task in C.TASKS:
    for n in (f'render/{task}__index.json', f'render/{task}__gt.npz', f'diag/{task}__seg.npz'):
        render_inputs[f'reports/results-artifact/{n}'] = sha(f'reports/results-artifact/{n}')

definition = {
    'name': 'arm4: four-arm local Meta-World campaign (ocsw4, ochf, c50dg, c50vis)',
    'recorded_at': datetime.now().astimezone().isoformat(),
    'status': 'registered before any run of these arms',
    'directive': ('user 2026-11-06 ~12:00 KST (memory four-arm-local-campaign-directive): A Original + CSW4 encoder, '
                  'B Original + Chamfer, C cst50 + depth gradient, D cst50 + visibility; eight tasks; 200k pretraining '
                  'seed 0 + export + frozen-encoder DrM 1M seeds 2000-2002; all local; analysis pre-registered'),
    'research_questions': {
        'A': 'With every Original loss, augmentation and the 50 % motion-aware tube kept, does replacing the Original '
             'encoder (128 x 8, 8 heads, patchify) by CSW4 (patch-local conv stem, 256 x 4, 4 heads) keep Original\'s '
             'control and improve reconstruction?',
        'B': 'Does adding the bidirectional 3D Chamfer loss (registered chamfer_lean settings) to the complete Original '
             'recipe keep control and improve geometry?',
        'C': 'On cst50, does restoring Original\'s multi-scale depth-gradient loss (alone) change control and '
             'reconstruction?',
        'D': 'On cst50, does restoring Original\'s visibility regulariser (alone) change control and reconstruction?'},
    'arms': arms,
    'references': refs,
    'config_sha256': config_sha,
    'config_sha256_definition': 'scripts/train.py provenance hash: sha256 of canonical JSON of '
                                'ops/jobs/run_identity.numeric_config(resolved config) (data root normalised, anchor '
                                'statistics by file hash); reproduces the stored hashes of the reference runs',
    'drift': DRIFT,
    'protocol': {
        'tasks': list(C.TASKS),
        'pretraining': '200k steps (train.steps 200000; warm-up 10k, cosine to min_lr 1e-5), train.seed 0, from scratch',
        'matched_initialisation': audit['matched_initialisation']['protocol'] + '; matched to each architecture\'s '
                                  'reference: A, C, D start from cst50\'s exact initial parameters (same CSW4 '
                                  'architecture), B from Original\'s (state-dict hashes in the audit; step-0 evaluations '
                                  'in the smoke). A and Original differ in architecture, so their initial weights (decoder '
                                  'included) are not shared',
        'export': 'scripts/export_encoder.py of checkpoints/step_0200000.pt (frozen encoder, num_frames 3, joint state)',
        'drm': 'scripts/train_rl.py --encoder splatter4d --seed {2000,2001,2002} --set train.num_train_steps=1000000 '
               'vision.export_path=<export>; resolved RL config equal to the reference DrM runs\' stored configs apart '
               'from the export path (audit drm)',
        'evaluation': 'scripts/eval_rl.py on every 10k snapshot (100 evaluations, 10k-1M): 240 episodes per evaluation = '
                      'train0-5 (20 each, 120), eval0-3 held-out static (20 each, 80), traj_lateral + traj_circular '
                      'moving (20 each, 40), suites reported separately; evaluator invocation = the reference\'s: A / B '
                      '3 workers x 8 envs (Original), C / D 6 x 8 (cst50). Episode reset seeds do not depend on the '
                      'layout, but evaluation is not bit-reproducible (renderer +/-1 intensity noise, GPU '
                      'nondeterminism) and a measured 3 x 8 vs 6 x 8 difference exceeded a same-layout repeat '
                      '(runs/setup/arm4-evalcheck-20261106.json), so the layout is matched rather than assumed invariant',
        'host': 'local only (8 authorized MIG 1g.24gb instances); references ran on the remote (Original all DrM; cst50 '
                'five tasks) and locally (cst50 hammer / pick-place / bin-picking); host-neutral policy 2026-10-15, '
                'host recorded per run',
    },
    'rows': ROWS,
    'analysis': ANALYSIS,
    'scheduling': {**SCHEDULING, 'eta': eta},
    'smoke': smoke,
    'environment': {'local_venv': '/home/ws/ws/splatter4d/.venv (cpython 3.10)', 'torch': '2.10.0+cu129', 'cuda': '12.9',
                    'cudnn': 91002, 'gsplat': '1.5.3', 'mujoco': '3.10.0', 'triton': '3.6.0', 'numpy': '1.26.4',
                    'nvidia_driver': '580.178.04', 'devices': 'MIG 1g.24gb on GPU-76871c16 / GPU-d09f0338',
                    'freeze': 'the venv and the release stay fixed for the whole campaign; a change is a deviation'},
    'evaluator_layout_check': json.loads((ROOT / 'runs/setup/arm4-evalcheck-20261106.json').read_text()),
    'render_inputs_sha256': render_inputs,
    'tools': {p: sha(p) for p in ('runs/setup/tools/arm4_common.py', 'runs/setup/tools/arm4_config_audit_20261106.py',
                                  'runs/setup/tools/arm4_xcommit_fingerprint_20261106.py',
                                  'runs/setup/tools/arm4_smoke_local_20261106.py', 'runs/setup/tools/arm4_eta_20261106.py',
                                  'runs/setup/tools/register_arm4_eight_task_20261106.py',
                                  'reports/results-artifact/src/run_arm4_render_20261106.py')},
    'audit_record': {'path': 'runs/setup/arm4-config-audit-20261106.json',
                     'sha256': sha('runs/setup/arm4-config-audit-20261106.json'), 'problems': audit['problems']},
    'reviews': reviews.get('rounds', []),
    'revisions': reviews.get('revisions', []),
}
OUT.write_text(json.dumps(definition, indent=1) + '\n')
print('wrote', OUT, len(OUT.read_text()), 'bytes')
