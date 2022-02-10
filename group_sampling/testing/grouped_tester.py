from typing import List

from testing.tester import Tester


class GroupedTester(Tester):

    def get_result(self, groups: List[List[str]]):
        return [
            [
                self.strategy.get_result([co for co in c if co != "root"])
                for c in configurations
            ]
            for configurations in groups
        ]
