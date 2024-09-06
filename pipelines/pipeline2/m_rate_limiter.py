# m_rate_limiter.py
import time

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule, Status
from stream_pipeline.module_classes import Module, ExecutionModule, ModuleOptions

import data
import logger

log = logger.get_logger()

class Rate_Limiter(ExecutionModule):
    def __init__(self) -> None:
        super().__init__(ModuleOptions(
                                use_mutex=False,
                                timeout=5,
                            ),
                            name="Rate_Limiter"
                        )
        self.flowrate_per_second: int = 1

        self.last_package_time: float = 0

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        current_time = time.time()
        if current_time - self.last_package_time < 1/self.flowrate_per_second:
            dpm.message = "Rate limit exceeded"
            dpm.status = Status.EXIT
            return

        self.last_package_time = current_time
        return