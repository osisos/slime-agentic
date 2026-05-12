"""SwanLab integration for experiment tracking.

Multi-process follows the same pattern as W&B: the primary process creates a
run and stores ``args.swanlab_run_id``; secondary processes join that run with
``resume="allow"``.
"""

import logging
import os
from copy import deepcopy

logger = logging.getLogger(__name__)


def init_swanlab_primary(args):
    """Create a SwanLab run in the driver process."""
    if not getattr(args, "use_swanlab", False):
        args.swanlab_run_id = None
        return

    try:
        import swanlab
    except ImportError:
        logger.warning("SwanLab requested but 'swanlab' is not installed. Install with: pip install swanlab")
        args.swanlab_run_id = None
        return

    project = getattr(args, "swanlab_project", None) or getattr(args, "wandb_project", None) or "slime"
    experiment_name = (
        getattr(args, "swanlab_experiment_name", None)
        or getattr(args, "wandb_group", None)
        or _default_swanlab_run_name()
    )
    mode = getattr(args, "swanlab_mode", None) or "cloud"

    run = swanlab.init(
        project=project,
        experiment_name=experiment_name,
        config=_compute_config_for_logging(args),
        mode=mode,
    )
    args.swanlab_run_id = run.id
    logger.info(
        "SwanLab initialized (primary). project=%s experiment_name=%s run_id=%s",
        project,
        experiment_name,
        run.id,
    )


def init_swanlab_secondary(args):
    """Join the existing SwanLab run in a worker process."""
    if not getattr(args, "use_swanlab", False):
        return

    swanlab_run_id = getattr(args, "swanlab_run_id", None)
    if swanlab_run_id is None:
        return

    try:
        import swanlab
    except ImportError:
        return

    project = getattr(args, "swanlab_project", None) or getattr(args, "wandb_project", None) or "slime"
    mode = getattr(args, "swanlab_mode", None) or "cloud"

    swanlab.init(
        project=project,
        id=swanlab_run_id,
        resume="allow",
        config=_compute_config_for_logging(args),
        mode=mode,
    )
    logger.info("SwanLab initialized (secondary), joined run_id=%s", swanlab_run_id)


def _default_swanlab_run_name():
    try:
        from slime.utils.external_utils.command_utils import create_run_id

        return create_run_id()
    except Exception:
        import uuid

        return f"run-{uuid.uuid4().hex[:8]}"


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)
    whitelist_env_vars = ["SLURM_JOB_ID"]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}
    return output
