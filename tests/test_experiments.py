from pathlib import Path

import pyrootutils
import pytest
import yaml
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

from src.train import train


@pytest.mark.slow
def test_train_experiments(tmp_path):
    config_path = Path(root / "configs" / "experiment")
    for config in config_path.glob("**/*.yaml"):
        with initialize(
            version_base="1.2", config_path="../configs", job_name="test_train_experiment"
        ):
            cfg = compose(
                config_name="train.yaml",
                return_hydra_config=True,
                overrides=[f"experiment={str(config).split('experiment/')[-1]}"],
            )
            sample_path = cfg.datamodule.get("_target_").split(".")[-2]
            config_path = root / sample_path / "config.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    override_cfg = yaml.safe_load(f)
                cfg.datamodule.update(override_cfg)

            with open_dict(cfg):
                cfg.paths.root_dir = str(root)
                cfg.paths.log_dir = str(tmp_path)
                cfg.paths.output_dir = str(tmp_path)
                cfg.trainer.fast_dev_run = True
                cfg.trainer.accelerator = "auto"
                cfg.trainer.strategy = None
                cfg.trainer.devices = 1
                cfg.extras.print_config = False
                cfg.extras.enforce_tags = False
                cfg.datamodule.num_workers = 0
                cfg.logger = None
            HydraConfig().set_config(cfg)
            train(cfg)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
