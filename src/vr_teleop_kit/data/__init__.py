"""Dataset review + training-run orchestration behind the web UI's
`/data` page.

Split so the relay server can import `catalog`/`review` (parquet + JSON
only) without ever importing `lerobot` or `torch`; the heavy imports
live in the `*_runner` modules, which only ever run in a subprocess.
"""
