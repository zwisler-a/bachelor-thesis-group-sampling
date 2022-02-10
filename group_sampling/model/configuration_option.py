from typing import List


class ConfigurationOption:
    name = ""
    outputString = ""
    prefix = ""
    postfix = ""
    parent = ""
    impliedOptions: List[str] = []
    excludedOptions: List[str] = []
    optional = False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'ConfigurationOption[' \
               f'name="{self.name}", ' \
               f'parent="{self.parent}", ' \
               f'outputString="{self.outputString}", ' \
               f'prefix="{self.prefix}", ' \
               f'postfix="{self.postfix}", ' \
               f'impliedOptions={self.impliedOptions} ' \
               f'excludedOptions={self.excludedOptions} ' \
               f'optional="{self.optional}" '

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(other) == str(self)
