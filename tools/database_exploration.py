from typing import Type

import fastdup
from tools.vpr_scenarios import Scenario_2023_11_27T90_37_23, VprScenario

if __name__ == "__main__":
    scenario: Type[VprScenario] = Scenario_2023_11_27T90_37_23()
    fd = fastdup.create(work_dir="/tmp/fastdup", input_dir=scenario.data_path / 'images')
    fd.run()
    fd.vis.component_gallery()