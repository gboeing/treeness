#!/usr/bin/env python
import multiprocessing as mp
import warnings
from os.path import getsize
from pathlib import Path

import momepy
import neatnet
import osmnx as ox

cpus = 3  # mp.cpu_count()

graphml_path = "/data/snm/models/graphml"
output_path = "/data/snm/models/neat/"
done = {fp.stem for fp in Path(output_path).glob("*/*.graphml")}
filepaths = [fp for fp in Path(graphml_path).glob("*/*.graphml") if fp.stem not in done]
filepaths = sorted(filepaths, key=getsize)  # sample(filepaths, len(filepaths))
args = ((fp,) for fp in filepaths)
print(ox.ts(), f"There are {len(filepaths):,} total GraphML files remaining.")


def neat(fp):
    G = ox.graph_to_gdfs(
        ox.convert.to_undirected(ox.projection.project_graph(ox.io.load_graphml(fp))), nodes=False
    )
    place = f"{fp.parent.stem}/{fp.stem}"
    print(ox.ts(), place, "START", len(G), flush=True)

    try:
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            Gs = momepy.gdf_to_nx(neatnet.neatify(G))
        ox.io.save_graphml(Gs, Path(output_path) / fp.parent.stem / fp.name)
        print(ox.ts(), place, "DONE", len(Gs), flush=True)
    except Exception:
        print(ox.ts(), place, "FAIL", flush=True)


if cpus > 1:
    with mp.get_context().Pool(cpus) as pool:
        pool.starmap_async(neat, args).get()
else:
    for fp in filepaths:
        neat(fp)

print(ox.ts(), "Done.")
