from argparse import Namespace
from copy import deepcopy

from speedrun import BaseExperiment

from neural_compilers.training.deit.main import get_args_parser


class NotAvailable:
    pass


class DeiTWrapper(BaseExperiment):

    def as_namespace(self, **keys_and_config_keys):
        namespace = dict()
        for key, config_key in keys_and_config_keys.items():
            namespace[key] = self.get(config_key)
        return Namespace(**namespace)

    def make_args(self):
        # Load in the parser
        parser = get_args_parser()
        arg_dict = deepcopy(parser.parse_args([]).__dict__)
        not_available = NotAvailable()
        # Get the value from config if specified; if not, write the
        # default value to config.
        for key in arg_dict:
            value = self.get(key, not_available)
            if value is not_available:
                self.set(key, value)
            else:
                arg_dict[key] = value
        # Return the namespace for downstream
        return Namespace(**arg_dict)
