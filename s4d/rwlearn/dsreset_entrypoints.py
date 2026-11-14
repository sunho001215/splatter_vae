"""Pure entrypoint aliases for input resolution; no runtime import changes."""
ENTRYPOINTS = {
    "scripts/dsreset_" + name: "scripts/" + name
    for name in ("train.py", "train_ours_continuation.py", "train_sincro.py", "train_reviwo.py",
                 "train_dynamo.py", "train_snerl.py", "train_mvp3d.py", "train_realworld_fql.py",
                 "realworld_fql_dataset.py", "compute_workspace_stats.py", "export_encoder.py",
                 "realworld_native_depth.py")
}


def legacy_row(row):
    script = ENTRYPOINTS.get(row.get("script"))
    return {**row, "script": script} if script else row

