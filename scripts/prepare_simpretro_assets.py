"""Acquire the stock explicitly selected by the user and build its bounded-memory index."""

import argparse
import json
from pathlib import Path

from agenticgen.materials import (
    download_official_stock,
    fetch_reference_corpus,
    index_stock,
    prepare_template_cache,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory", type=Path, default=Path("data/inventories/simpretro_official")
    )
    parser.add_argument("--simpretro-root", type=Path, default=Path("external/SimpRetro"))
    parser.add_argument("--image", default="agenticgen-simpretro-worker:20261016")
    args = parser.parse_args()
    stock = args.directory / "emolecules.txt"
    print(json.dumps(download_official_stock(stock), indent=2), flush=True)
    print(json.dumps(index_stock(stock, args.directory / "stock.sqlite"), indent=2), flush=True)
    print(
        json.dumps(
            prepare_template_cache(
                args.simpretro_root,
                Path("data/processed/simpretro/fingerprint_base.pkl"),
                args.image,
            ),
            indent=2,
        ),
        flush=True,
    )
    print(
        json.dumps(fetch_reference_corpus(Path("data/raw/syncraft_release")), indent=2), flush=True
    )


if __name__ == "__main__":
    main()
