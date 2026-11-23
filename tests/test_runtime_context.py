from pathlib import Path
import tempfile
import unittest

from meaning_to_feeling.backend import LocalStore, Runtime


class RuntimeContextTests(unittest.TestCase):
    def runtime(self, root, context_size=2048):
        runtime = Runtime(root, root / "runtime", context_size=context_size)
        runtime.post = lambda endpoint, body=None, **kwargs: (
            {"prompt": "text"} if endpoint == "/apply-template" else {"tokens": list(range(1522))})
        return runtime

    def test_window_includes_generated_output_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            body = dict(messages=[], max_tokens=512)
            self.assertEqual(self.runtime(root).check_tokens(body), 1522)
            with self.assertRaises(ValueError):
                self.runtime(root).check_tokens(dict(body, max_tokens=1024))
            self.assertEqual(self.runtime(root, 4096).check_tokens(dict(body, max_tokens=1024)), 1522)

    def test_cache_records_actual_window_and_default_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.gguf").write_bytes(b"test")
            ordinary = LocalStore(self.runtime(root), root / "normal.sqlite")
            extended = LocalStore(self.runtime(root, 4096), root / "extended.sqlite")
            try:
                self.assertEqual(ordinary.context["context_size"], 2048)
                self.assertEqual(extended.context["context_size"], 4096)
                self.assertNotEqual(ordinary.context, extended.context)
            finally:
                ordinary.close()
                extended.close()


if __name__ == "__main__":
    unittest.main()
