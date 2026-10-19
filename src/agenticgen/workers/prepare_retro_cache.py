"""Losslessly cache every upstream template fingerprint, avoiding duplicate int64 arrays."""

import hashlib
import json
import pickle
import sys
from pathlib import Path


def prepare(templates_path, output):
    import numpy as np
    from rdkit import rdBase
    from syntheseus.reaction_prediction.inference.simpretro import smarts_to_fingerprint

    templates_path, output = Path(templates_path), Path(output)
    if output.exists():
        raise FileExistsError("Fingerprint cache already exists")
    templates = json.loads(templates_path.read_text())
    cache, minimum, maximum = {}, 0, 0
    for index, template in enumerate(templates):
        fingerprint = smarts_to_fingerprint(template)
        compact = fingerprint.astype(np.int16)
        if not np.array_equal(fingerprint, compact):
            raise ValueError("Template fingerprint cannot be represented losslessly as int16")
        cache[template] = compact
        minimum, maximum = min(minimum, int(compact.min())), max(maximum, int(compact.max()))
        if (index + 1) % 5000 == 0:
            print(f"Cached {index + 1}/{len(templates)} template fingerprints", flush=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("wb") as stream:
        pickle.dump(cache, stream, protocol=4)
    with partial.open("rb") as stream:
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
        digest = hasher.hexdigest()
    metadata = {
        "schema_version": "agenticgen.template-cache.v1",
        "templates_sha256": hashlib.sha256(templates_path.read_bytes()).hexdigest(),
        "sha256": digest,
        "template_count": len(templates),
        "unique_templates": len(cache),
        "rdkit_version": rdBase.rdkitVersion,
        "dtype": "int16",
        "minimum": minimum,
        "maximum": maximum,
        "all_values_losslessly_checked": True,
        "bytes": partial.stat().st_size,
        "scope": "Storage dtype only; filter inputs still use upstream float32 conversion",
    }
    partial.replace(output)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    prepare(*sys.argv[1:])
