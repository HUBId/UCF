import tempfile
from pathlib import Path
import unittest

import importlib.util

_spec = importlib.util.spec_from_file_location("build_bundle", Path(__file__).with_name("build_bundle.py"))
build_bundle = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(build_bundle)


class BuildBundleTests(unittest.TestCase):
    def test_version_body_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            (repo / "configs").mkdir(parents=True)
            (repo / "configs" / "prod.toml").write_text("[x]\na=1\n", encoding="utf-8")
            (repo / "policies" / "packs" / "base_v1").mkdir(parents=True)
            (repo / "policies" / "packs" / "base_v1" / "a.toml").write_text("k=1\n", encoding="utf-8")
            (repo / "models").mkdir(parents=True)
            (repo / "models" / "manifest.toml").write_text("m=1\n", encoding="utf-8")
            (repo / "target" / "release").mkdir(parents=True)
            for bin_name in ("ucf-runtime", "ucf-ops"):
                p = repo / "target" / "release" / bin_name
                p.write_text("bin", encoding="utf-8")
                p.chmod(0o755)

            bundle1 = build_bundle.build_bundle(repo, root / "b1", "prod", repo / "target" / "release")
            bundle2 = build_bundle.build_bundle(repo, root / "b2", "prod", repo / "target" / "release")
            v1 = (bundle1 / "VERSION.txt").read_text(encoding="utf-8")
            v2 = (bundle2 / "VERSION.txt").read_text(encoding="utf-8")
            self.assertEqual(v1, v2)


if __name__ == "__main__":
    unittest.main()
