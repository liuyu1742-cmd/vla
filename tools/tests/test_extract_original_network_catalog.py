import unittest


class OriginalNetworkCatalogTests(unittest.TestCase):
    def test_clean_cell_normalizes_line_breaks_and_empty_markers(self):
        from tools.extract_original_network_catalog import clean_cell

        self.assertEqual(clean_cell(" 閺呴缚鍏橀幓鎺戦獓\n\n "), "閺呴缚鍏橀幓鎺戦獓")
        self.assertEqual(clean_cell("\r\n"), "")


if __name__ == "__main__":
    unittest.main()
