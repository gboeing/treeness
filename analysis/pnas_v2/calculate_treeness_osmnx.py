#!/usr/bin/env python
"""Calculate treeness after the fixed OSMnx node-consolidation test."""

import csv
import multiprocessing as mp
import traceback
from numbers import Real
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

# The analysis uses the supplied GraphML collection rather than downloading or
# generating a different network.  The remaining paths are relative to the repository root
INPUT_ROOT = Path("/data/snm/models/graphml")
INDICATORS_PATH = Path("analysis/data/indicators-all.csv")
OUTPUT_PATH = Path("analysis/data/treeness-results_osmnx.csv")
FAILURE_LOG_PATH = Path("analysis/data/treeness-results_osmnx-failures.csv")
# Ten metres is the source data convention used as each node's buffer radius.
TOLERANCE = 10
TOKYO_UC_ID = 5929
# Fixed batch sizes limit concurrent memory use: 9,725 small graphs use 24 workers,
# 149 medium graphs use 4, and the large batch runs 48 of the final 49 paths with
# 1 worker; Tokyo runs separately last.
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
    """Parse the urban-centre ID from the final hyphen-delimited name part.

    The graph names contain descriptive text that can itself contain hyphens, so
    the ID must be read from the last stem component, not from an earlier split.

    Parameters
    ----------
    path : pathlib.Path or str
        Graph or result path whose final stem component is the urban-centre ID.

    Returns
    -------
    int
        Parsed urban-centre ID.

    Notes
    -----
    Both GraphML paths and result IDs use the same final-hyphen convention.  Using
    one parser for both lets resume and coverage checks compare the same identifier
    that the batch selector used.
    """
    return int(Path(str(path)).stem.rsplit("-", 1)[1])


def select_eligible_paths(input_root: Path, indicators_path: Path) -> list[Path]:
    """Select one size-sorted GraphML path for every eligible urban centre.

    Parameters
    ----------
    input_root : pathlib.Path
        Root directory containing one country directory per group of GraphML files.
    indicators_path : pathlib.Path
        Indicator CSV containing ``uc_id`` and ``node_count`` columns.

    Returns
    -------
    list[pathlib.Path]
        Exactly 9,923 eligible paths sorted by file size and path string.

    Raises
    ------
    ValueError
        If the eligible ID count is wrong or an eligible graph is missing or duplicated.

    Notes
    -----
    The indicator table is authoritative for eligibility because it records the
    source node count.  A dictionary keyed by UC ID detects duplicate GraphML files
    before the size sort, which is important because the prescribed batch boundaries
    depend on a complete, one-file-per-ID list.
    """
    indicators = pd.read_csv(indicators_path)
    # Apply the strict source criterion before touching GraphML paths.  The expected
    # count catches a wrong input table early instead of silently changing batches.
    eligible_ids = set(indicators.loc[indicators["node_count"] > 100, "uc_id"].astype(int))
    if len(eligible_ids) != 9_923:
        raise ValueError(f"Expected 9,923 eligible IDs, found {len(eligible_ids)}")

    paths_by_id: dict[int, Path] = {}
    # Search only one directory below the fixed root, matching the source layout and
    # preventing unrelated files elsewhere in the repository from entering a batch.
    for path in input_root.glob("*/*.graphml"):
        uc_id = parse_uc_id(path)
        if uc_id not in eligible_ids:
            continue
        # Keying by ID makes duplicate input files an explicit failure rather than a
        # nondeterministic choice of which file happens to be visited last.
        if uc_id in paths_by_id:
            raise ValueError(f"Duplicate graph ID {uc_id}: {paths_by_id[uc_id]} and {path}")
        paths_by_id[uc_id] = path

    missing_ids = sorted(eligible_ids - paths_by_id.keys())
    if missing_ids:
        raise ValueError(f"Missing eligible graph IDs: {missing_ids}")
    # File size is the prescribed workload ordering.  The path string breaks ties so
    # repeated runs submit the same graphs at the same batch boundaries.
    return sorted(paths_by_id.values(), key=lambda path: (path.stat().st_size, str(path)))


def select_batch(paths: list[Path], batch: str) -> list[Path]:
    """Select one prescribed OSMnx batch from all eligible paths.

    Parameters
    ----------
    paths : list[pathlib.Path]
        All 9,923 eligible paths sorted by the prescribed key.
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
        If the requested batch does not have its prescribed size or Tokyo name.

    Notes
    -----
    Python slices are zero-based and half-open, so ``paths[9_725:9_874]`` contains
    exactly the 149 medium paths specified by the analysis plan.  Tokyo is removed
    from the final 49-path slice and selected separately so the largest graph does
    not enter the worker pool.
    """
    if len(paths) != 9_923:
        raise ValueError(f"Expected 9,923 paths, found {len(paths)}")

    if batch == "small":
        # The first 9,725 size-sorted paths form the small batch and use 24 workers.
        selected = paths[:9_725]
        expected_size = 9_725
    elif batch == "medium":
        # This half-open slice starts immediately after small and has 149 paths.
        selected = paths[9_725:9_874]
        expected_size = 149
    elif batch == "large":
        # The final 49 paths contain Tokyo.  Exclude it here because Tokyo is run
        # directly in the parent after all other batches have durable result rows.
        selected = paths[9_874:]
        expected_size = 49
        selected = [path for path in selected if parse_uc_id(path) != TOKYO_UC_ID]
        expected_size -= 1
    elif batch == "tokyo":
        # Select by parsed ID, then verify the expected source filename so a renamed
        # or mismatched graph cannot silently be treated as Tokyo.
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
    """Calculate length-weighted treeness statistics for an undirected graph.

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
    NetworkX returns a minimum spanning forest for a disconnected graph.  That is
    required here because the analysis uses the full consolidated graph and defines
    ``circuit_rank`` with the number of connected components.  The edge and length
    ratios compare that forest with the original graph; ``circuit_rank_ratio`` is
    therefore the link-redundancy measure used as theta.
    """
    # A forest, rather than a single connected-tree calculation, preserves every
    # component in the full graph and gives the planned edge-weighted MST statistic.
    minimum_spanning_tree = nx.minimum_spanning_tree(graph, weight="length")
    node_count = graph.number_of_nodes()
    component_count = nx.number_connected_components(graph)
    edge_count_orig = graph.number_of_edges()
    edge_count_mst = minimum_spanning_tree.number_of_edges()
    # Sum the original and forest edge lengths separately.  The normalization step
    # runs before this function, so NetworkX and the sums use numeric edge weights.
    edge_count_ratio = edge_count_mst / edge_count_orig
    length_orig = sum(float(data["length"]) for _, _, _, data in graph.edges(keys=True, data=True))
    length_mst = sum(
        float(data["length"]) for _, _, _, data in minimum_spanning_tree.edges(keys=True, data=True)
    )
    length_ratio = length_mst / length_orig
    # For a graph with c components, a spanning forest has n-c edges.  The excess
    # edges are the independent circuits, so this is the same circuit-rank formula
    # used by the v1 analysis and later reported as theta.
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
    """Validate that every edge length is numeric, finite, and positive.

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
        Returns after every edge passes validation.

    Notes
    -----
    The check deliberately rejects bad data instead of converting or estimating a
    replacement length.  This preserves the consolidated graph exactly as OSMnx
    produced it before NetworkX calculates the weighted spanning forest.
    """
    for _, _, _, data in graph.edges(keys=True, data=True):
        length = data.get("length")
        # Numeric strings and booleans do not satisfy the input contract.  Reject
        # them rather than changing an edge attribute after consolidation.
        if isinstance(length, bool) or not isinstance(length, Real):
            raise ValueError(f"Missing or nonnumeric edge length: {length!r}")
        if not np.isfinite(length) or length <= 0:
            raise ValueError(f"Edge length must be finite and positive: {length!r}")


def process_graph(
    path: Path,
) -> tuple[dict[str, object] | None, dict[str, str] | None]:
    """Consolidate one directed GraphML graph and calculate its statistics.

    Parameters
    ----------
    path : pathlib.Path
        Directed GraphML path to load and process.

    Returns
    -------
    tuple[dict[str, object] | None, dict[str, str] | None]
        A result row, or a failure row containing the complete traceback.

    Notes
    -----
    The order of operations matches the prescribed representation: load the
    directed source graph, project it, consolidate intersections with the fixed
    OSMnx options, convert it to an undirected graph, normalize its edge lengths, and
    then calculate statistics.  Catching an exception here lets the parent preserve
    the traceback in the failure CSV while continuing to receive other workers'
    results.
    """
    try:
        # Load the source directed graph without rebuilding or filtering its edges.
        graph = ox.io.load_graphml(path)
        # Projection supplies the metric coordinate system required by the 10 m
        # consolidation tolerance; it does not replace the source network geometry.
        projected_graph = ox.projection.project_graph(graph)
        # These are the only consolidation options allowed by the revision analysis;
        # no additional filtering, edge correction, or neatnet step follows them.
        consolidated_graph = ox.consolidate_intersections(
            projected_graph,
            tolerance=TOLERANCE,
            rebuild_graph=True,
            reconnect_edges=True,
            dead_ends=True,
        )
        # Undirected conversion makes each physical street available to the spanning
        # forest while retaining OSMnx's consolidated edge attributes.
        graph_undirected = ox.convert.to_undirected(consolidated_graph)
        validate_edge_lengths(graph_undirected)

        stats = calculate_stats(graph_undirected)
        return {"id": path.stem, **stats}, None
    except Exception:
        return None, {"id": path.stem, "path": str(path), "error": traceback.format_exc()}


def load_completed_ids(output_path: Path) -> set[str]:
    """Load completed result IDs after validating the result header.

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

    Notes
    -----
    An absent output is a clean first run, so it returns an empty set.  Existing IDs
    are read from the durable result CSV rather than from in-memory worker state;
    this makes an interrupted run restart only the graphs that have no result row.
    """
    if not output_path.exists():
        return set()

    with output_path.open(newline="") as file:
        reader = csv.DictReader(file)
        # A matching header prevents a stale or unrelated CSV from controlling which
        # graphs are skipped during resume.
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
    """Remove completed result IDs before a batch is submitted.

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

    Notes
    -----
    Result IDs are the graph stems, so comparing ``path.stem`` avoids dependence on
    directory names and keeps retry behavior consistent with the result writer.
    """
    # Preserve the prescribed path order while removing only durable result IDs.
    return [path for path in paths if path.stem not in completed_ids]


def append_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    """Append one row to a CSV and flush it before returning.

    Parameters
    ----------
    path : pathlib.Path
        CSV path to create or append.
    fieldnames : list[str]
        Header and column order.
    row : dict[str, object]
        Values for one row.

    Notes
    -----
    The file is opened for each row because worker results arrive incrementally.  A
    single header is written only when the file is new, and ``flush`` makes a
    completed result or failure visible before the next graph is processed.

    Returns
    -------
    None
        The supplied row is appended and flushed before the function returns.
    """
    # Check both existence and size so a pre-created empty output still receives a
    # valid header before the first durable row.
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        # Immediate flushing is the resume guarantee: an interruption loses at most
        # the row currently being handled, not all prior completed rows.
        file.flush()


def validate_output(output_path: Path, indicators_path: Path) -> None:
    """Validate complete OSMnx coverage and the required numerical identities.

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
        Returns only after every final-result check passes.

    Notes
    -----
    Coverage is checked against the same source eligibility rule used to construct
    the batches.  The numerical identities then verify that every result is a valid
    spanning-forest calculation, rather than only checking that the CSV has the right
    number of rows.
    """
    results = pd.read_csv(output_path)
    # Validate the schema and IDs before converting values so a malformed partial
    # output cannot appear complete merely because it has enough rows.
    assert list(results.columns) == RESULT_FIELDS
    assert results["id"].notna().all()
    assert not results["id"].duplicated().any()

    indicators = pd.read_csv(indicators_path)
    eligible_ids = set(indicators.loc[indicators["node_count"] > 100, "uc_id"].astype(int))
    result_uc_ids = {parse_uc_id(graph_id) for graph_id in results["id"]}
    assert len(results) == len(eligible_ids)
    assert len(result_uc_ids) == len(results)
    assert result_uc_ids == eligible_ids

    # Coerce all statistic columns together.  This rejects missing or nonnumeric
    # values and gives the remaining assertions one consistent numeric frame.
    numeric = results[RESULT_FIELDS[1:]].apply(pd.to_numeric, errors="coerce")
    assert numeric.notna().all().all()
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()
    assert (numeric["node_count"] >= 1).all()
    assert (numeric["component_count"] >= 1).all()
    assert (numeric["edge_count_orig"] >= numeric["edge_count_mst"]).all()
    assert (numeric["edge_count_mst"] >= 1).all()
    # A spanning forest has one fewer edge than nodes in each component; the next
    # identity is the definition of circuit rank for the original graph.
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
    """Process non-Tokyo batches, then process Tokyo directly in the parent.

    Notes
    -----
    The workflow builds and validates all four fixed batches before processing any
    graph.  It then resumes from durable result IDs, uses worker pools only for the
    three ordinary batches, handles Tokyo in the parent, and performs final coverage
    and numerical validation only after all 9,923 results are present.

    Returns
    -------
    None
        Returns only after the complete result file passes final validation.
    """
    eligible_paths = select_eligible_paths(INPUT_ROOT, INDICATORS_PATH)
    # Build every batch up front so a boundary or Tokyo naming error stops the run
    # before it starts a long-lived pool or writes partial analysis output.
    all_batch_names = [batch for batch, _ in BATCH_WORKERS] + ["tokyo"]
    batch_paths = {batch: select_batch(eligible_paths, batch) for batch in all_batch_names}
    all_batch_ids = [parse_uc_id(path) for paths in batch_paths.values() for path in paths]
    # The batch assertions check sizes; this union check additionally catches overlap
    # or omission between the ordinary slices and the separately selected Tokyo path.
    assert len(all_batch_ids) == 9_923
    assert len(set(all_batch_ids)) == 9_923

    for batch, workers in BATCH_WORKERS:
        # Read durable result IDs before submission so a rerun processes only missing graphs.
        completed_ids = load_completed_ids(OUTPUT_PATH)
        remaining_paths = select_remaining_paths(batch_paths[batch], completed_ids)
        if remaining_paths:
            # Fork avoids the sandbox-denied forkserver socket; one task per worker
            # limits memory retained between graphs while preserving the fixed worker
            # count for this batch.
            context = mp.get_context("fork")
            with context.Pool(workers, maxtasksperchild=1) as pool:
                for result, error in pool.imap_unordered(
                    process_graph, remaining_paths, chunksize=1
                ):
                    # Only the parent writes rows, making each completion durable
                    # before the next result is handled.
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
            # Fail before advancing to the next batch.  A failure row is diagnostic
            # only; a result row is the completion marker used by resume.
            print(f"Missing result rows after batch {batch!r}: {missing_ids}")
            raise SystemExit(1)
        print(f"Completed {len(batch_paths[batch]):,} paths for batch {batch!r}")

    completed_ids = load_completed_ids(OUTPUT_PATH)
    remaining_tokyo = select_remaining_paths(batch_paths["tokyo"], completed_ids)
    if remaining_tokyo:
        # Tokyo runs last without a pool because it is the largest graph and needs
        # the parent process to avoid another worker's retained memory.
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
        # Keep the 9,922 non-Tokyo rows intact, but do not claim success while Tokyo
        # still has only a failure-log entry or no row at all.
        print(f"Missing result rows after Tokyo: {missing_tokyo}")
        raise SystemExit(1)
    print("Completed Tokyo batch")
    # This is the only success path.  It verifies coverage, finiteness, graph
    # identities, and ratio bounds before the process returns status zero.
    validate_output(OUTPUT_PATH, INDICATORS_PATH)


if __name__ == "__main__":
    main()
