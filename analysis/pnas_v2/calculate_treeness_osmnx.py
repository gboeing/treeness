#!/usr/bin/env python
"""Calculate treeness after OSMnx intersection consolidation."""

import csv
import multiprocessing as mp
import traceback
from numbers import Real
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

# load source GraphML files; other paths are relative to the repository root
INPUT_ROOT = Path("/data/snm/models/graphml")
INDICATORS_PATH = Path("analysis/data/indicators-all.csv")
OUTPUT_PATH = Path("analysis/data/treeness-results_osmnx.csv")
FAILURE_LOG_PATH = Path("analysis/data/treeness-results_osmnx-failures.csv")
# set each node buffer radius in meters
TOLERANCE = 10
TOKYO_UC_ID = 5929
# use fewer workers for larger graphs to limit memory use
# process Tokyo separately after the other batches
BATCH_WORKERS = (("small", 24), ("medium", 4), ("large", 1))
RESULT_FIELDS = [
    "id",
    "node_count",
    "component_count",
    "edge_count_orig",
    "edge_count_mst",
    "edge_count_ratio",
    "length_orig",
    "length_mst",
    "length_ratio",
    "circuit_rank",
    "circuit_rank_ratio",
]
FAILURE_FIELDS = ["id", "path", "error"]


def parse_uc_id(path: Path | str) -> int:
    """Extract the urban centre ID from the last filename component.

    Parameters
    ----------
    path : pathlib.Path or str
        Graph or result path whose final stem component is the urban-centre ID.

    Returns
    -------
    int
        Urban centre ID.

    Notes
    -----
    City names can contain hyphens, so the ID is read from the final part.
    """
    return int(Path(str(path)).stem.rsplit("-", 1)[1])


def select_eligible_paths(input_root: Path, indicators_path: Path) -> list[Path]:
    """Select eligible graphs and sort them by file size.

    Parameters
    ----------
    input_root : pathlib.Path
        Root directory containing one country directory per group of GraphML
        files.
    indicators_path : pathlib.Path
        Indicator CSV containing ``uc_id`` and ``node_count`` columns.

    Returns
    -------
    list[pathlib.Path]
        Eligible paths sorted by file size, then path.

    Raises
    ------
    ValueError
        If the eligible ID count is wrong or an eligible graph is missing or
        duplicated.
    """
    indicators = pd.read_csv(indicators_path)
    # select urban centres with more than 100 source graph nodes
    eligible_ids = set(indicators.loc[indicators["node_count"] > 100, "uc_id"].astype(int))
    if len(eligible_ids) != 9_923:
        raise ValueError(f"Expected 9,923 eligible IDs, found {len(eligible_ids)}")

    paths_by_id: dict[int, Path] = {}
    # find graphs in the country subdirectories
    for path in input_root.glob("*/*.graphml"):
        uc_id = parse_uc_id(path)
        if uc_id not in eligible_ids:
            continue
        # check that each urban centre has only one graph
        if uc_id in paths_by_id:
            raise ValueError(f"Duplicate graph ID {uc_id}: {paths_by_id[uc_id]} and {path}")
        paths_by_id[uc_id] = path

    missing_ids = sorted(eligible_ids - paths_by_id.keys())
    if missing_ids:
        raise ValueError(f"Missing eligible graph IDs: {missing_ids}")
    # sort by file size, then path to break ties
    return sorted(paths_by_id.values(), key=lambda path: (path.stat().st_size, str(path)))


def select_batch(paths: list[Path], batch: str) -> list[Path]:
    """Select the graphs for one processing batch.

    Parameters
    ----------
    paths : list[pathlib.Path]
        Eligible paths sorted by file size, then path.
    batch : str
        One of ``small``, ``medium``, ``large``, or ``tokyo``.

    Returns
    -------
    list[pathlib.Path]
        Paths in the requested batch.

    Raises
    ------
    ValueError
        If the input path count or batch name is invalid.
    AssertionError
        If the batch size or Tokyo filename is incorrect.
    """
    if len(paths) != 9_923:
        raise ValueError(f"Expected 9,923 paths, found {len(paths)}")

    if batch == "small":
        # select the small graphs
        selected = paths[:9_725]
        expected_size = 9_725
    elif batch == "medium":
        # select the medium graphs
        selected = paths[9_725:9_874]
        expected_size = 149
    elif batch == "large":
        # select the large graphs and reserve Tokyo for the final batch
        selected = paths[9_874:]
        expected_size = 49
        selected = [path for path in selected if parse_uc_id(path) != TOKYO_UC_ID]
        expected_size -= 1
    elif batch == "tokyo":
        # select Tokyo by urban centre ID and check its filename
        selected = [path for path in paths if parse_uc_id(path) == TOKYO_UC_ID]
        expected_size = 1
        if selected and selected[0].name != "tokyo-5929.graphml":
            raise AssertionError(f"UC ID 5929 is not tokyo-5929.graphml: {selected[0]}")
    else:
        raise ValueError(f"Unknown batch: {batch}")

    assert len(selected) == expected_size, (
        f"Batch {batch!r} has {len(selected)} paths; expected {expected_size}"
    )
    return selected


def calculate_stats(graph: nx.MultiGraph) -> dict[str, int | float]:
    """Calculate treeness statistics for an undirected graph.

    Parameters
    ----------
    graph : networkx.MultiGraph
        Full graph with numeric, finite, positive ``length`` edge attributes.

    Returns
    -------
    dict[str, int | float]
        Node, component, edge, length, and circuit-rank statistics.

    Notes
    -----
    Disconnected graphs use a minimum spanning forest. Link redundancy is
    circuit rank divided by edge count; length redundancy is one minus
    the ratio of forest length to full graph length.
    """
    # calculate a minimum spanning forest if the graph has multiple components
    minimum_spanning_tree = nx.minimum_spanning_tree(graph, weight="length")
    node_count = graph.number_of_nodes()
    component_count = nx.number_connected_components(graph)
    edge_count_orig = graph.number_of_edges()
    edge_count_mst = minimum_spanning_tree.number_of_edges()
    # compare edge counts and total lengths with the spanning forest
    edge_count_ratio = edge_count_mst / edge_count_orig
    length_orig = sum(float(data["length"]) for _, _, _, data in graph.edges(keys=True, data=True))
    length_mst = sum(
        float(data["length"]) for _, _, _, data in minimum_spanning_tree.edges(keys=True, data=True)
    )
    length_ratio = length_mst / length_orig
    # calculate circuit rank as the edges in excess of a spanning forest
    circuit_rank = edge_count_orig - node_count + component_count
    circuit_rank_ratio = circuit_rank / edge_count_orig
    return {
        "node_count": node_count,
        "component_count": component_count,
        "edge_count_orig": edge_count_orig,
        "edge_count_mst": edge_count_mst,
        "edge_count_ratio": edge_count_ratio,
        "length_orig": length_orig,
        "length_mst": length_mst,
        "length_ratio": length_ratio,
        "circuit_rank": circuit_rank,
        "circuit_rank_ratio": circuit_rank_ratio,
    }


def validate_edge_lengths(graph: nx.MultiGraph) -> None:
    """Check that edge lengths are numeric, finite, and positive.

    Parameters
    ----------
    graph : networkx.MultiGraph
        Full graph whose edge lengths must be checked.

    Raises
    ------
    ValueError
        If an edge length is missing, nonnumeric, nonfinite, or not positive.

    Returns
    -------
    None
        No return value.
    """
    for _, _, _, data in graph.edges(keys=True, data=True):
        length = data.get("length")
        # reject strings and booleans as edge lengths
        if isinstance(length, bool) or not isinstance(length, Real):
            raise ValueError(f"Missing or nonnumeric edge length: {length!r}")
        if not np.isfinite(length) or length <= 0:
            raise ValueError(f"Edge length must be finite and positive: {length!r}")


def process_graph(
    path: Path,
) -> tuple[dict[str, object] | None, dict[str, str] | None]:
    """Consolidate intersections and calculate treeness for one graph.

    Parameters
    ----------
    path : pathlib.Path
        Directed GraphML path to load and process.

    Returns
    -------
    tuple[dict[str, object] | None, dict[str, str] | None]
        A (result, None) pair, or (None, failure) with the traceback.
    """
    try:
        # load the directed source graph
        graph = ox.io.load_graphml(path)
        # project to metric coordinates for the 10 m consolidation tolerance
        projected_graph = ox.projection.project_graph(graph)
        # merge nearby nodes and reconnect their incident edges
        consolidated_graph = ox.consolidate_intersections(
            projected_graph,
            tolerance=TOLERANCE,
            rebuild_graph=True,
            reconnect_edges=True,
            dead_ends=True,
        )
        # convert to an undirected graph to measure physical street
        # infrastructure
        graph_undirected = ox.convert.to_undirected(consolidated_graph)
        validate_edge_lengths(graph_undirected)

        stats = calculate_stats(graph_undirected)
        return {"id": path.stem, **stats}, None
    except Exception:
        return None, {"id": path.stem, "path": str(path), "error": traceback.format_exc()}


def load_completed_ids(output_path: Path) -> set[str]:
    """Read graph IDs already saved in the result CSV.

    Parameters
    ----------
    output_path : pathlib.Path
        Incremental result CSV. A missing path means no IDs are complete.

    Returns
    -------
    set[str]
        Unique result IDs already present in the CSV.

    Raises
    ------
    ValueError
        If the header is wrong, an ID is missing, or an ID is duplicated.
    """
    if not output_path.exists():
        return set()

    with output_path.open(newline="") as file:
        reader = csv.DictReader(file)
        # check the header before using existing rows to resume
        if reader.fieldnames != RESULT_FIELDS:
            raise ValueError(f"Unexpected result header: {reader.fieldnames!r}")
        completed_ids: set[str] = set()
        for row in reader:
            graph_id = row.get("id")
            if not graph_id:
                raise ValueError(f"Missing result ID in {output_path}")
            if graph_id in completed_ids:
                raise ValueError(f"Duplicate result ID in {output_path}: {graph_id}")
            completed_ids.add(graph_id)
    return completed_ids


def select_remaining_paths(paths: list[Path], completed_ids: set[str]) -> list[Path]:
    """Select graphs without saved results.

    Parameters
    ----------
    paths : list[pathlib.Path]
        Paths selected for one processing batch.
    completed_ids : set[str]
        Result IDs already written to the output CSV.

    Returns
    -------
    list[pathlib.Path]
        Paths whose result IDs are not complete.
    """
    # keep unprocessed graphs in their original batch order
    return [path for path in paths if path.stem not in completed_ids]


def append_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    """Append a CSV row and flush the file.

    Parameters
    ----------
    path : pathlib.Path
        CSV path to create or append.
    fieldnames : list[str]
        Header and column order.
    row : dict[str, object]
        Values for one row.

    Returns
    -------
    None
        No return value.
    """
    # write a header if the file is new or empty
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        # flush each row so completed results are available when resuming
        file.flush()


def validate_output(output_path: Path, indicators_path: Path) -> None:
    """Check output coverage, bounds, and graph identities.

    Parameters
    ----------
    output_path : pathlib.Path
        OSMnx result CSV.
    indicators_path : pathlib.Path
        Indicator CSV used to define eligible IDs.

    Raises
    ------
    AssertionError
        If coverage, finiteness, bounds, or graph identities fail.

    Returns
    -------
    None
        No return value.
    """
    results = pd.read_csv(output_path)
    # check the output columns and graph IDs
    assert list(results.columns) == RESULT_FIELDS
    assert results["id"].notna().all()
    assert not results["id"].duplicated().any()

    indicators = pd.read_csv(indicators_path)
    eligible_ids = set(indicators.loc[indicators["node_count"] > 100, "uc_id"].astype(int))
    result_uc_ids = {parse_uc_id(graph_id) for graph_id in results["id"]}
    assert len(results) == len(eligible_ids)
    assert len(result_uc_ids) == len(results)
    assert result_uc_ids == eligible_ids

    # check that every statistic is numeric and finite
    numeric = results[RESULT_FIELDS[1:]].apply(pd.to_numeric, errors="coerce")
    assert numeric.notna().all().all()
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()
    assert (numeric["node_count"] >= 1).all()
    assert (numeric["component_count"] >= 1).all()
    assert (numeric["edge_count_orig"] >= numeric["edge_count_mst"]).all()
    assert (numeric["edge_count_mst"] >= 1).all()
    # check the spanning forest edge count and circuit rank identities
    assert np.array_equal(
        numeric["edge_count_mst"].to_numpy(),
        (numeric["node_count"] - numeric["component_count"]).to_numpy(),
    )
    assert np.array_equal(
        numeric["circuit_rank"].to_numpy(),
        (
            numeric["edge_count_orig"] - numeric["node_count"] + numeric["component_count"]
        ).to_numpy(),
    )
    assert np.allclose(
        numeric["edge_count_ratio"] + numeric["circuit_rank_ratio"],
        1,
        atol=1e-12,
        rtol=0,
    )
    assert (numeric["circuit_rank_ratio"] >= 0).all()
    assert (numeric["circuit_rank_ratio"] < 1).all()
    assert (numeric["length_mst"] > 0).all()
    assert (numeric["length_mst"] <= numeric["length_orig"]).all()
    assert (numeric["length_ratio"] > 0).all()
    assert (numeric["length_ratio"] <= 1).all()


def main() -> None:
    """Process the graph batches, then Tokyo, and check the results.

    Returns
    -------
    None
        No return value.
    """
    eligible_paths = select_eligible_paths(INPUT_ROOT, INDICATORS_PATH)
    # select all batches before processing graphs
    all_batch_names = [batch for batch, _ in BATCH_WORKERS] + ["tokyo"]
    batch_paths = {batch: select_batch(eligible_paths, batch) for batch in all_batch_names}
    all_batch_ids = [parse_uc_id(path) for paths in batch_paths.values() for path in paths]
    # check that batches cover all eligible graphs without overlap
    assert len(all_batch_ids) == 9_923
    assert len(set(all_batch_ids)) == 9_923

    for batch, workers in BATCH_WORKERS:
        # skip graphs that already have result rows
        completed_ids = load_completed_ids(OUTPUT_PATH)
        remaining_paths = select_remaining_paths(batch_paths[batch], completed_ids)
        if remaining_paths:
            # use fork to avoid forkserver socket restrictions
            # restart each worker after one graph to release memory
            context = mp.get_context("fork")
            with context.Pool(workers, maxtasksperchild=1) as pool:
                for result, error in pool.imap_unordered(
                    process_graph, remaining_paths, chunksize=1
                ):
                    # write results from the parent process
                    if result is not None:
                        append_row(OUTPUT_PATH, RESULT_FIELDS, result)
                    else:
                        assert error is not None
                        append_row(FAILURE_LOG_PATH, FAILURE_FIELDS, error)

        completed_ids = load_completed_ids(OUTPUT_PATH)
        missing_ids = sorted(
            path.stem for path in batch_paths[batch] if path.stem not in completed_ids
        )
        if missing_ids:
            # stop if any graph in this batch has no result row
            print(f"Missing result rows after batch {batch!r}: {missing_ids}")
            raise SystemExit(1)
        print(f"Completed {len(batch_paths[batch]):,} paths for batch {batch!r}")

    completed_ids = load_completed_ids(OUTPUT_PATH)
    remaining_tokyo = select_remaining_paths(batch_paths["tokyo"], completed_ids)
    if remaining_tokyo:
        # process Tokyo without a worker pool to limit memory use
        result, error = process_graph(remaining_tokyo[0])
        if result is not None:
            append_row(OUTPUT_PATH, RESULT_FIELDS, result)
        else:
            assert error is not None
            append_row(FAILURE_LOG_PATH, FAILURE_FIELDS, error)

    completed_ids = load_completed_ids(OUTPUT_PATH)
    missing_tokyo = sorted(
        path.stem for path in batch_paths["tokyo"] if path.stem not in completed_ids
    )
    if missing_tokyo:
        # stop if Tokyo has no result row
        print(f"Missing result rows after Tokyo: {missing_tokyo}")
        raise SystemExit(1)
    print("Completed Tokyo batch")
    # check the complete output
    validate_output(OUTPUT_PATH, INDICATORS_PATH)


if __name__ == "__main__":
    main()
