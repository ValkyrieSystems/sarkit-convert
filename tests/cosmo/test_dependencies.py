import importlib
import unittest

MODULES = [
    "sarkit_convert.cosmo",
]


class TestImports(unittest.TestCase):
    def test_can_import(self):
        for name in MODULES:
            with self.subTest(name=name):
                module = importlib.import_module(name)
                self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
