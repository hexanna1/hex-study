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

    def test_pattern_bundle_round_trip_geometry_markers_and_lazy_rice(self):
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
            // Reverse lookup order crosses seek checkpoints independently of the key stream.
            for (const key of Object.keys(data.patterns).reverse()) out[key] = api.patternEntryForLookupInData(data,key);
            console.log(JSON.stringify({patterns:out, cached:data.entryCache.size}));
        """, {"bundle": base64.b64encode(bundle).decode()})
        self.assertEqual(decoded["patterns"], patterns)
        self.assertEqual(decoded["cached"], len(patterns))

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
            console.log(JSON.stringify({edges,continuations:Array.from(data.continuationCounts),
                root:root.candidates.map(c=>[c.move,c.prior,c.redMetric]),
                child:child.candidates.map(c=>[c.move,c.prior,c.redMetric,c.tree_mover_winrate])}));
        """, {"bundle": base64.b64encode(bundle).decode()})
        self.assertEqual(decoded, {
            "edges": [[1, 3], [2], [], [2]], "continuations": [4, 1, 0, 1],
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
            api.ensureJosekiRandomIndex(data);
            const paths=(kind,localOnly,count)=>Array.from({length:count},(_,rank)=>api.josekiLineForRandomRank(data,kind,localOnly,rank));
            console.log(JSON.stringify({nodes:data.nodeCount,
                core:paths('core',false,data.subtreeCore[0]),coreLocal:paths('core',true,data.subtreeCoreLocal[0]),
                leaf:paths('leaf',false,data.subtreeLeaves[0]),leafLocal:paths('leaf',true,data.subtreeLeavesLocal[0])}));
        """, {"bundle": base64.b64encode(bundle).decode()})
        self.assertEqual(decoded, {
            "nodes": 4,
            "core": ["A[1,1]", "A[1,1:3,1]", "A[1,1:]", "A[2,1]", "A[2,1:3,1]", "A[2,1:]"],
            "coreLocal": ["A[1,1]", "A[1,1:3,1]", "A[2,1]", "A[2,1:3,1]"],
            "leaf": ["A[1,1:3,1]", "A[1,1:]", "A[2,1:3,1]", "A[2,1:]"],
            "leafLocal": ["A[1,1:3,1]", "A[2,1:3,1]"],
        })

    def test_shared_bit_readers_and_bitmap_rank_boundaries(self):
        result = self._node("""
            const fs = require('node:fs'),assert = require('node:assert/strict');
            const window = {};
            Function('window',fs.readFileSync('docs/study_ui.js','utf8'))(window);
            const ui = window.HexStudyUI;
            const bytes = Uint8Array.from({length:90},(_,i)=>(i*73+19)&255);
            const view = new DataView(bytes.buffer);
            for(const bits of [0,1,2,10,18,19]) for(let at=0;at<50;at++) {
                let expected=0;
                for(let bit=0;bit<bits;bit++) expected+=((bytes[(at+bit)>>>3]>>>((at+bit)&7))&1)*2**bit;
                assert.equal(ui.readPackedWordAtBit(view,0,at,bits),expected);
            }
            for (const bitCount of [0,1,7,8,63,64,255,256,257,513]) {
                // Include nonzero padding and extra bytes beyond the logical bitmap.
                const bitmap=Uint8Array.from({length:Math.ceil(bitCount/8)+2},(_,i)=>(i*73+255)&255);
                for (const interval of [8,64,256]) {
                    const checkpoints=ui.bitmapRankCheckpoints(bitmap,bitCount,interval);
                    let rank=0;
                    for(let i=0;i<=bitCount;i++) {
                        assert.equal(ui.bitmapRankBefore(bitmap,checkpoints,i,interval),rank);
                        if(i<bitCount) rank+=(bitmap[i>>>3]>>>(i&7))&1;
                    }
                    assert.equal(checkpoints.at(-1),rank);
                }
            }
            console.log(JSON.stringify(true));
        """, {})
        self.assertTrue(result)

    def test_uvarints_preserve_large_unsigned_values(self):
        values = [0, 127, 128, 16383, 16384, 2**31 - 1, 2**31, 2**32 - 1, 2**33 - 2, 2**53 - 1]
        encoded = bytearray()
        for value in values:
            wbu.write_uvarint(encoded, value)
        decoded = self._node("""
            const fs=require('node:fs'),window={};
            Function('window',fs.readFileSync('docs/study_ui.js','utf8'))(window);
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
                for(const key of Object.keys(data.patterns)) patterns[key]=api.patternEntryForLookupInData(data,key);
                return patterns;
            });
            console.log(JSON.stringify(results));
        """, payload)
        self.assertEqual(decoded, inputs)

    def test_redirect_gaps_and_signed_target_deltas_round_trip(self):
        counts = [2, 0, 150, 0, 4]
        flags = [False] * sum(counts)
        for position in [0, 1, 140, 152, 155]:
            flags[position] = True
        targets = [4, 0, 1, 4, 2]
        encoded = wbu.encode_redirect_references(flags=flags, targets=targets, node_edge_counts=counts)
        decoded = self._node("""
            const fs=require('node:fs'),window={};
            Function('window',fs.readFileSync('docs/study_ui.js','utf8'))(window);
            const input=JSON.parse(fs.readFileSync(0,'utf8')),starts=[0];
            for(const count of input.counts) starts.push(starts.at(-1)+count);
            const result=window.HexStudyUI.decodeRedirectReferences(Buffer.from(input.bytes,'base64'),0,input.count,starts);
            const flags=Array.from({length:starts.at(-1)},(_,i)=>Boolean(result.bitmap[i>>>3]&(1<<(i&7))));
            console.log(JSON.stringify({flags,targets:Array.from(result.targets),offset:result.offset}));
        """, {"bytes": base64.b64encode(encoded).decode(), "counts": counts, "count": len(targets)})
        self.assertEqual(decoded, {"flags": flags, "targets": targets, "offset": len(encoded)})

    def test_editor_transforms_match_python_symmetry(self):
        points = [[-2, 1], [0, 3], [4, -1]]
        decoded = self._node("""
            const fs=require('node:fs'),{loadCodec}=require('./tests/website_codec_harness.cjs');
            const api=loadCodec('docs','patterns'),points=JSON.parse(fs.readFileSync(0,'utf8'));
            console.log(JSON.stringify(Array.from({length:12},(_,i)=>points.map(p=>api.applyTransformAx(p,i)))));
        """, points)
        expected = [[list(apply_transform_ax(tuple(point), i)) for point in points] for i in range(12)]
        self.assertEqual(decoded, expected)


class JosekiInternTests(unittest.TestCase):
    def test_interning_merges_recursive_payloads_and_preserves_incoming_move_kind(self):
        nodes = {
            0: (0, False, 0, [(1, 900), (2, 900), (3, 900)], [1, 2, 3]),
            1: (1, True, 500, [(4, 900)], [4]),
            2: (1, True, 500, [(4, 900)], [5]),
            3: (0, True, 500, [], [6]),
            4: (16, True, 800, [], []),
            5: (16, True, 800, [], []),
            6: (16, True, 800, [], []),
        }
        merged = jwd._intern_compact_nodes(nodes)
        self.assertEqual(len(merged), 5)
        self.assertEqual(merged[0][4][0], merged[0][4][1])
        local_leaf = merged[merged[0][4][0]][4][0]
        tenuki_leaf = merged[merged[0][4][2]][4][0]
        self.assertNotEqual(local_leaf, tenuki_leaf)
        self.assertEqual(merged[local_leaf], merged[tenuki_leaf])

    def test_interning_keeps_observable_payload_differences(self):
        base = (17, True, 800, [(1, 900)], [None])
        variants = [
            base,
            (1, True, 800, [(1, 900)], [None]),
            (17, True, 801, [(1, 900)], [None]),
            (17, True, 800, [(2, 900)], [None]),
            (17, True, 800, [(1, 899)], [None]),
            (17, True, 800, [(1, 900), (2, 900)], [None, None]),
            (17, True, 800, [(2, 900), (1, 900)], [None, None]),
        ]
        nodes = {0: (len(variants), False, 0, [(i, 900) for i in range(len(variants))], list(range(1, len(variants) + 1)))}
        nodes.update({i: payload for i, payload in enumerate(variants, 1)})
        self.assertEqual(len(jwd._intern_compact_nodes(nodes)), len(nodes))
