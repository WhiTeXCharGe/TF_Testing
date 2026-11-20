from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from common import plan_settings


def resolve_application_directory(
    module_file: Path,
    *,
    frozen: bool,
    executable: Path,
) -> Path:
    if frozen:
        return executable.resolve().parent
    return module_file.resolve().parents[1]


@dataclass(frozen=True)
class ApplicationPaths:
    application_directory: Path
    work_directory: Path
    config_file: Path

    @classmethod
    def current(cls) -> ApplicationPaths:
        frozen = bool(getattr(sys, "frozen", False))
        application_directory = resolve_application_directory(
            Path(__file__),
            frozen=frozen,
            executable=Path(sys.executable),
        )
        config_directory = application_directory / "config"
        settings = plan_settings.load(program_dir=config_directory)
        return cls(
            application_directory=application_directory,
            work_directory=settings.work_dir.resolve(),
            config_file=settings.path or config_directory / plan_settings.CONFIG_NAME,
        )

    def prepare(self) -> None:
        if not self.application_directory.is_dir():
            raise NotADirectoryError(
                f"Application directory does not exist: {self.application_directory}"
            )
        if not self.config_file.is_file():
            raise FileNotFoundError(
                f"Required config file not found: {self.config_file}. "
                "Use the same plan_assist.config.txt as StartSchedule."
            )
        try:
            with self.config_file.open("rb") as config:
                config.read(1)
        except OSError as error:
            raise OSError(
                f"Cannot read the config file: {self.config_file}: {error}"
            ) from error

        try:
            self.work_directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=self.work_directory) as probe:
                probe.write(b"\0")
                probe.flush()
        except OSError as error:
            raise OSError(
                f"Cannot create or write to the Work directory: "
                f"{self.work_directory}: {error}"
            ) from error
