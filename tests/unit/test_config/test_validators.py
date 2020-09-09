from configparser import ConfigParser
from pathlib import Path
import unittest

import vak.config.validators

HERE = Path(__file__).parent
TEST_CONFIGS_PATH = HERE.joinpath('..', '..', 'test_data', 'configs')


class TestValidators(unittest.TestCase):
    def test_are_sections_valid(self):
        invalid_section_config = TEST_CONFIGS_PATH.joinpath(
            'invalid_section_config.ini'
        )
        invalid_section_config = str(invalid_section_config)
        user_config_parser = ConfigParser()
        user_config_parser.read(invalid_section_config)
        with self.assertRaises(ValueError):
            vak.config.validators.are_sections_valid(user_config_parser,
                                                     invalid_section_config)

    def test_are_options_valid(self):
        invalid_option_config = TEST_CONFIGS_PATH.joinpath(
            'invalid_option_config.ini'
        )
        invalid_option_config = str(invalid_option_config)
        section_with_invalid_option = 'PREP'
        user_config_parser = ConfigParser()
        user_config_parser.read(invalid_option_config)
        with self.assertRaises(ValueError):
            vak.config.validators.are_options_valid(user_config_parser,
                                                    section_with_invalid_option,
                                                    invalid_option_config)


if __name__ == '__main__':
    unittest.main()
