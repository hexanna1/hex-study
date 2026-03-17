import base64
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import artifact_json as aj
import joseki_website_data as jwd
import opening_website_data as owd
import pattern_website_data as pwd
import website_bundle_utils as wbu
from hex_symmetry import apply_transform_ax

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "Node is required to exercise the website decoders")
class WebsiteCodecTests(unittest.TestCase):
    def _node(self, script, payload):
        result = subprocess.run(
            [NODE, "-e", script], input=json.dumps(payload), text=True,
            capture_output=True, cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_pattern_bundle_round_trip_geometry_and_markers(self):
        coordinates = json.loads((ROOT / "tests/fixtures/pattern_coordinates.json").read_text())
        values = [0, 1, 127, 128, 500, 999, 1000, 1001, 1002]
        tenuki_values = [401, None, 402, 403, 406, 410]
        patterns = {}
        for row, (pattern, pairs) in enumerate(coordinates.items()):
            patterns[pattern] = {
                "p": pwd._to_play_from_packed_pattern_key(pwd._pack_pattern_key(pattern)),
                "c": [[q, r, values[(row + col) % len(values)]] for col, (q, r) in enumerate(pairs)],
            }
            if tenuki_values[row] is not None:
                patterns[pattern]["t"] = tenuki_values[row]
        bundle = pwd._build_pattern_bundle_from_index({"patterns": patterns, "pattern_count": len(patterns)})
        decoded = self._node("""
            const fs = require('node:fs');
            const {loadCodec,arrayBuffer} = require('./tests/website_codec_harness.cjs');
            const input = JSON.parse(fs.readFileSync(0,'utf8'));
            const api = loadCodec('docs','patterns');
            const data = api.normalizeLoadedData(arrayBuffer(Buffer.from(input.bundle,'base64')));
            const out = {};
            // Reverse lookup order exercises coordinate ranks independently of key order.
            for (let row = data.pattern_count - 1; row >= 0; row -= 1) {
                const key = api.patternForIndex(data, row);
                out[key] = api.patternEntryForLookupInData(data, api.parsePattern(key));
            }
            console.log(JSON.stringify(out));
        """, {"bundle": base64.b64encode(bundle).decode()})
        self.assertEqual(decoded, patterns)

    def test_opening_bundle_decodes_shared_child_and_inherited_metric(self):
        def candidate(move, prior, child, winrate):
            return {"move": move, "prior": prior, "child": child, "tree_mover_winrate": winrate}

        artifact = {"board_size": 3, "root": 0, "nodes": [
            {"ply": 0, "importance": 1.0, "candidates": [
                candidate("a1", 0.6, 1, 0.65), candidate("b1", 0.4, 2, 0.65),
            ]},
            {"ply": 1, "importance": 0.2, "tree_red_winrate": 0.65,
             "candidates": [candidate("c1", 1.0, 3, 0.35)]},
            {"ply": 1, "importance": 0.2, "tree_red_winrate": 0.65,
             "candidates": [candidate("c2", 1.0, 3, 0.35)]},
            {"ply": 2, "importance": 0.2, "tree_red_winrate": 0.65, "candidates": []},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            aj.dump_tree(Path(tmp) / "openings-s3.json", artifact)
            bundle = owd.build_opening_bundle(artifacts_root=Path(tmp), board_size=3)
        decoded = self._node("""
            const fs=require('node:fs'),{loadCodec,arrayBuffer}=require('./tests/website_codec_harness.cjs');
            const api=loadCodec('docs','openings'),input=JSON.parse(fs.readFileSync(0,'utf8'));
            const data=api.normalizeLoadedData(arrayBuffer(Buffer.from(input.bundle,'base64')));
            const edges=Array.from({length:data.nodeCount},(_,i)=>api.openingChildIndices(data,i));
            const root=api.decodeOpeningNode(data,0,'',null,0);
            const child=api.decodeOpeningNode(data,root.candidates[0].childIndex,'a1',root.candidates[0].redMetric,1);
            console.log(JSON.stringify({edges,continuations:edges.map((_,i)=>api.openingDescendantCount(data,i)),
                root:root.candidates.map(c=>[c.move,c.prior,c.redMetric]),
                child:child.candidates.map(c=>[c.move,c.prior,c.redMetric,c.tree_mover_winrate])}));
        """, {"bundle": base64.b64encode(bundle).decode()})
        self.assertEqual(decoded, {
            "edges": [[1, 2], [3], [3], []], "continuations": [4, 1, 1, 0],
            "root": [["a1", 0.6, 650], ["b1", 0.4, 650]], "child": [["c1", 1, 650, 0.35]],
        })

    def test_joseki_interning_preserves_random_path_ranks(self):
        def candidate(kind, fraction, child, local=None):
            out = {"kind": kind, "stone_fraction": fraction, "child": child}
            if local is not None:
                out["local"] = local
            return out

        def node(line, candidates):
            return {"line": line, "importance": 1.0, "candidates": candidates}

        artifact = {"family": "A", "board_size": 19, "nodes": [
            node("", [candidate("local", 1.0, 1, [1, 1]), candidate("local", 0.95, 2, [2, 1])]),
            node("A[1,1]", [candidate("local", 0.9, 3, [3, 1]), candidate("tenuki", 0.6, 5)]),
            node("A[2,1]", [candidate("local", 0.9, 4, [3, 1]), candidate("tenuki", 0.6, 6)]),
            node("A[1,1:3,1]", [candidate("tenuki", 0.8, None)]),
            node("A[2,1:3,1]", [candidate("tenuki", 0.8, None)]),
            node("A[1,1:]", [candidate("tenuki", 0.8, None)]),
            node("A[2,1:]", [candidate("tenuki", 0.8, None)]),
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            aj.dump_tree(Path(tmp) / "joseki-a-s19.json", artifact)
            bundle = jwd.build_family_bundle(artifacts_root=Path(tmp), family="A", board_size=19)
        decoded = self._node("""
            const fs=require('node:fs'),{loadCodec,arrayBuffer}=require('./tests/website_codec_harness.cjs');
            const api=loadCodec('docs','joseki'),input=JSON.parse(fs.readFileSync(0,'utf8'));
            const data=api.normalizeLoadedData(arrayBuffer(Buffer.from(input.bundle,'base64')));
            const root=data.rootIndex;
            const paths=(kind,localOnly,count)=>Array.from({length:count},(_,rank)=>api.josekiLineForRandomRank(data,kind,localOnly,rank));
            console.log(JSON.stringify({nodes:data.nodeCount,
                core:paths('core',false,data.coreCounts[root]),coreLocal:paths('core',true,data.localCoreDescendants[root]),
                leaf:paths('leaf',false,data.leafCounts[root]),leafLocal:paths('leaf',true,data.localLeafDescendants[root])}));
        """, {"bundle": base64.b64encode(bundle).decode()})
        self.assertEqual(decoded, {
            "nodes": 3,
            "core": ["A[1,1]", "A[1,1:3,1]", "A[1,1:]", "A[2,1]", "A[2,1:3,1]", "A[2,1:]"],
            "coreLocal": ["A[1,1]", "A[1,1:3,1]", "A[2,1]", "A[2,1:3,1]"],
            "leaf": ["A[1,1:3,1]", "A[1,1:]", "A[2,1:3,1]", "A[2,1:]"],
            "leafLocal": ["A[1,1:3,1]", "A[2,1:3,1]"],
        })


    def test_uvarints_preserve_large_unsigned_values(self):
        values = [0, 127, 128, 16383, 16384, 2**31 - 1, 2**31, 2**32 - 1, 2**33 - 2, 2**53 - 1]
        encoded = bytearray()
        for value in values:
            wbu.write_uvarint(encoded, value)
        decoded = self._node("""
            const fs=require('node:fs'),window={};
            Function('window',fs.readFileSync('docs/ui/study.js','utf8'))(window);
            const input=JSON.parse(fs.readFileSync(0,'utf8')),bytes=Buffer.from(input.bytes,'base64');
            let offset=0;const values=[];
            for(let i=0;i<input.count;i++) {
                const result=window.HexStudyUI.readUvarintAt(bytes,offset);
                values.push(result.value);offset=result.offset;
            }
            console.log(JSON.stringify({values,offset}));
        """, {"bytes": base64.b64encode(encoded).decode(), "count": len(values)})
        self.assertEqual(decoded, {"values": values, "offset": len(encoded)})

    def test_pattern_round_trip_extreme_values_and_empty_streams(self):
        coordinates = json.loads((ROOT / "tests/fixtures/pattern_coordinates.json").read_text())
        inputs = [{}, {"+[0,0]-[]": {"p": "red", "t": 650, "c": []}}]
        for value in [0, 1000, 1001, 1002]:
            inputs.append({pattern: {
                "p": pwd._to_play_from_packed_pattern_key(pwd._pack_pattern_key(pattern)),
                "c": [[q, r, value] for q, r in pairs],
            } for pattern, pairs in coordinates.items()})
        payload = []
        for patterns in inputs:
            bundle = pwd._build_pattern_bundle_from_index({"patterns": patterns, "pattern_count": len(patterns)})
            payload.append(base64.b64encode(bundle).decode())
        decoded = self._node("""
            const fs=require('node:fs'),{loadCodec,arrayBuffer}=require('./tests/website_codec_harness.cjs');
            const api=loadCodec('docs','patterns'),input=JSON.parse(fs.readFileSync(0,'utf8'));
            const results=input.map(bundle=>{
                const data=api.normalizeLoadedData(arrayBuffer(Buffer.from(bundle,'base64'))),patterns={};
                for (let row = 0; row < data.pattern_count; row += 1) {
                    const key = api.patternForIndex(data, row);
                    patterns[key] = api.patternEntryForLookupInData(data, api.parsePattern(key));
                }
                return patterns;
            });
            console.log(JSON.stringify(results));
        """, payload)
        self.assertEqual(decoded, inputs)

    def test_editor_transforms_match_python_symmetry(self):
        points = [[-2, 1], [0, 3], [4, -1]]
        decoded = self._node("""
            const fs=require('node:fs'),{loadCodec}=require('./tests/website_codec_harness.cjs');
            const api=loadCodec('docs','patterns'),points=JSON.parse(fs.readFileSync(0,'utf8'));
            console.log(JSON.stringify(Array.from({length:12},(_,i)=>points.map(p=>api.applyTransformAx(p,i)))));
        """, points)
        expected = [[list(apply_transform_ax(tuple(point), i)) for point in points] for i in range(12)]
        self.assertEqual(decoded, expected)
