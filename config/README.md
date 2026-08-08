# Configuration layout

Project configurations are grouped by model, benchmark, and experiment type:

```text
config/
└── splattervae/
    └── metaworld/
        ├── temporal/
        └── ablations/
            └── single_timestep/
```

- `temporal/` contains the standard three-frame SplatterVAE experiment.
- `ablations/single_timestep/` contains the t0-only reconstruction experiment.

Each directory contains one YAML file per Meta-World environment.

Examples:

```bash
uv run train_model.py \\
  --config config/splattervae/metaworld/temporal/button-press-wall.yaml

CONFIG_SET=ablations/single_timestep \\
  scripts/run_metaworld_env_trainings.sh
```
