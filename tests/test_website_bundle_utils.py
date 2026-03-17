import json
import tempfile
import unittest
from pathlib import Path

import website_bundle_utils as wbu


class WebsiteBundleUtilsTests(unittest.TestCase):
    def test_write_hashed_bundle_manifest(self):
        self.assertNotEqual(
            wbu.hashed_bundle_filename(prefix="sample", payload=b"data"),
            wbu.hashed_bundle_filename(prefix="sample", payload=b"other"),
        )
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "current.json"
            stale = Path(td) / "sample.000000000000.bin"
            stale.write_bytes(b"stale")
            wbu.write_hashed_bundle_manifest(
                out_path=out_path,
                bundles={"main": wbu.BundlePayload(prefix="sample", payload=b"data")},
                stale_globs=["sample.*.bin"],
                manifest_from_bundle_names=lambda names: {"bundles": names},
            )
            manifest = json.loads(out_path.read_text(encoding="utf-8"))
            bundle_name = manifest["bundles"]["main"]
            bundle_path = Path(td) / bundle_name
            self.assertRegex(bundle_name, r"^sample\.[0-9a-f]{12}\.bin$")
            self.assertEqual(bundle_path.read_bytes(), b"data")
            self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main()
