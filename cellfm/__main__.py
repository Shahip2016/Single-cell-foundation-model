"""
Enables running CellFM as a module: python -m cellfm

Usage:
    python -m cellfm info --config 80M
    python -m cellfm train --data cells.h5ad --epochs 10
    python -m cellfm predict --checkpoint best.pt --data test.h5ad
    python -m cellfm embed --checkpoint best.pt --data cells.h5ad
"""

from .cli import main

if __name__ == "__main__":
    main()
