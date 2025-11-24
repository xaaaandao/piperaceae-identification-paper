import unittest

from sample import Level

class TestLevel(unittest.TestCase):
    specific_epithet = "Peperomia blanda"

    def setUp(self):
        self.level_no_name = Level(1)
        self.level = Level(2, name=self.specific_epithet)

    def test_exists_level(self):
        self.assertTrue(isinstance(self.level, Level))

    def test_level_default_name(self):
        self.assertEqual(self.level_no_name.name, "Classe")

    def test_level_with_name(self):
        self.assertEqual(self.level.name, self.specific_epithet)

    def test_increment_tp(self):
        self.level.tp += 1
        self.assertEqual(self.level.tp, 1)

    def test_increment_tn(self):
        self.level.tn += 1
        self.assertEqual(self.level.tn, 1)

    def test_increment_fp(self):
        self.level.fp += 1
        self.assertEqual(self.level.fp, 1)

    def test_increment_fn(self):
        self.level.fn += 1
        self.assertEqual(self.level.fn, 1)


if __name__ == '__main__':
    unittest.main()
