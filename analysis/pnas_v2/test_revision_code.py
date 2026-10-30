"""Synthetic tests for the v2 analysis code.

These tests exercise only imported functions with in-memory, temporary, or tiny
synthetic inputs.  They deliberately do not call either production ``main``
function, read ``/data/snm``, create production outputs, or run the shell runner.
"""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx
import osmnx as ox
import pandas as pd

from analysis.pnas_v2 import calculate_revision_statistics as revision
from analysis.pnas_v2 import calculate_treeness_osmnx as osmnx_analysis


def test_uc_id_parsing_and_batch_membership() -> None:
    """Check final-component UC parsing and all fixed batch boundaries.

    Notes
    -----
    The synthetic list has the exact planned 9,923 IDs, with Tokyo placed at the
    end to exercise the prescribed final-49 exclusion and separate selection without
    touching the production GraphML directory.
    """
    # Use deterministic names and order so a failure identifies a slice-boundary
    # problem rather than a filesystem ordering difference.
    paths = []
    for uc_id in range(1, 9_924):
        filename = f"city-{uc_id}.graphml"
        country = "country"
        if uc_id == osmnx_analysis.TOKYO_UC_ID:
            filename = "tokyo-5929.graphml"
            country = "japan"
        paths.append(Path(country) / filename)
    paths = [path for path in paths if osmnx_analysis.parse_uc_id(path) != 5_929]
    paths.append(Path("japan/tokyo-5929.graphml"))

    # Hyphens in the descriptive prefix must not change the final numeric ID.
    assert osmnx_analysis.parse_uc_id(Path("country/name-with-hyphens-123.graphml")) == 123
    small = osmnx_analysis.select_batch(paths, "small")
    medium = osmnx_analysis.select_batch(paths, "medium")
    large = osmnx_analysis.select_batch(paths, "large")
    tokyo = osmnx_analysis.select_batch(paths, "tokyo")
    assert len(small) == 9_725
    assert len(medium) == 149
    assert len(large) == 48
    assert len(tokyo) == 1
    assert small == paths[:9_725]
    assert medium == paths[9_725:9_874]
    assert large == [
        path for path in paths[9_725 + 149 :] if osmnx_analysis.parse_uc_id(path) != 5_929
    ]
    assert tokyo[0].name == "tokyo-5929.graphml"
    assert (
        len(
            {
                osmnx_analysis.parse_uc_id(path)
                for batch in (small, medium, large, tokyo)
                for path in batch
            }
        )
        == 9_923
    )


def test_treeness_formulas_for_connected_and_disconnected_graphs() -> None:
    """Check MST, length, and circuit-rank formulas in both graph cases.

    Notes
    -----
    The first graph confirms that a tree has no circuit redundancy.  The second has
    two components, so its MST is a forest and its circuit rank includes the
    component correction required by the full-graph formula.
    """
    # A connected tree has exactly n-1 edges and therefore zero circuit rank.
    tree = nx.MultiGraph()
    tree.add_edge(1, 2, length=1)
    tree.add_edge(2, 3, length=2)
    tree_stats = osmnx_analysis.calculate_stats(tree)
    assert tree_stats["node_count"] == 3
    assert tree_stats["component_count"] == 1
    assert tree_stats["edge_count_mst"] == 2
    assert tree_stats["circuit_rank"] == 0
    assert tree_stats["length_ratio"] == 1

    # Add one cycle to the first component and one separate edge component.  This
    # makes both the forest edge count and component-aware circuit rank observable.
    graph = nx.MultiGraph()
    graph.add_edge(1, 2, length=1)
    graph.add_edge(2, 3, length=2)
    graph.add_edge(1, 3, length=4)
    graph.add_edge(4, 5, length=3)
    stats = osmnx_analysis.calculate_stats(graph)
    assert stats["node_count"] == 5
    assert stats["component_count"] == 2
    assert stats["edge_count_orig"] == 4
    assert stats["edge_count_mst"] == 3
    assert stats["edge_count_ratio"] == 0.75
    assert stats["length_orig"] == 10
    assert stats["length_mst"] == 6
    assert stats["length_ratio"] == 0.6
    assert stats["circuit_rank"] == 1
    assert stats["circuit_rank_ratio"] == 0.25


def test_tiny_geographic_graph_processing() -> None:
    """Process a tiny temporary geographic GraphML file through OSMnx.

    Notes
    -----
    OSMnx projection and consolidation require geographic coordinates, so this test
    creates a three-node directed chain in a temporary directory.  It verifies the
    real per-graph pipeline without loading any production graph.
    """
    # Keep both directions for each street because the production input is directed
    # before OSMnx converts the consolidated result to an undirected graph.
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    graph.graph["simplified"] = True
    nodes = {
        1: (-122.0000, 37.0000),
        2: (-121.9990, 37.0000),
        3: (-121.9980, 37.0000),
    }
    for node_id, (x, y) in nodes.items():
        graph.add_node(node_id, x=x, y=y)
    for start, end in ((1, 2), (2, 1), (2, 3), (3, 2)):
        graph.add_edge(start, end, length=100.0)

    with TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "tiny-1.graphml"
        ox.io.save_graphml(graph, filepath=path)
        result, error = osmnx_analysis.process_graph(path)

    assert error is None
    assert result is not None
    assert result["id"] == "tiny-1"
    assert result["node_count"] >= 1
    assert result["edge_count_mst"] == result["node_count"] - result["component_count"]


def test_invalid_edge_lengths_are_rejected_without_replacement() -> None:
    """Check that invalid edge lengths fail instead of being replaced.

    Notes
    -----
    Numeric strings, missing values, booleans, nonfinite values, and nonpositive
    values do not meet the fixed edge-length contract.  The validator must leave
    each rejected value unchanged so it cannot alter the consolidated graph.
    """
    for invalid_length in ("2", None, True, float("nan"), float("inf"), 0, -1):
        graph = nx.MultiGraph()
        graph.add_edge(1, 2, length=invalid_length)
        try:
            osmnx_analysis.validate_edge_lengths(graph)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid edge length was accepted: {invalid_length!r}")
        assert graph.edges[1, 2, 0]["length"] is invalid_length

    valid_graph = nx.MultiGraph()
    valid_graph.add_edge(1, 2, length=2.0)
    osmnx_analysis.validate_edge_lengths(valid_graph)
    assert valid_graph.edges[1, 2, 0]["length"] == 2.0


def test_append_header_validation_duplicate_detection_and_resume() -> None:
    """Check incremental CSV writing, duplicate rejection, and ID resumption.

    Notes
    -----
    Temporary files model an interrupted production run: completed IDs come from
    durable rows, duplicate IDs are rejected, and a wrong header cannot control a
    retry.
    """
    first_row = {
        "id": "city-1",
        "node_count": 3,
        "component_count": 1,
        "edge_count_orig": 3,
        "edge_count_mst": 2,
        "edge_count_ratio": 2 / 3,
        "length_orig": 6.0,
        "length_mst": 3.0,
        "length_ratio": 0.5,
        "circuit_rank": 1,
        "circuit_rank_ratio": 1 / 3,
    }
    with TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "results.csv"
        # Two appends must create one header and two durable result IDs.
        osmnx_analysis.append_row(output_path, osmnx_analysis.RESULT_FIELDS, first_row)
        second_row = first_row | {"id": "city-2"}
        osmnx_analysis.append_row(output_path, osmnx_analysis.RESULT_FIELDS, second_row)
        assert output_path.read_text().count("id,node_count") == 1
        assert osmnx_analysis.load_completed_ids(output_path) == {"city-1", "city-2"}
        remaining = osmnx_analysis.select_remaining_paths(
            [Path("city-1"), Path("city-3")], {"city-1"}
        )
        assert remaining == [Path("city-3")]

        duplicate_path = Path(temporary_directory) / "duplicate.csv"
        # A duplicate result would make resume ambiguous, so the reader must reject it.
        with duplicate_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=osmnx_analysis.RESULT_FIELDS)
            writer.writeheader()
            writer.writerow(first_row)
            writer.writerow(first_row)
        try:
            osmnx_analysis.load_completed_ids(duplicate_path)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate result ID was accepted")

        bad_header_path = Path(temporary_directory) / "bad-header.csv"
        # A mismatched schema is unsafe to append to or use for skip decisions.
        bad_header_path.write_text("wrong\nvalue\n")
        try:
            osmnx_analysis.load_completed_ids(bad_header_path)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid result header was accepted")


def test_final_result_invariants() -> None:
    """Check final-result coverage and numerical invariants on synthetic rows.

    Notes
    -----
    The valid rows cover one connected and one disconnected graph.  The second pass
    changes only circuit rank, proving that validation catches an identity error even
    when coverage and CSV schema remain valid.
    """
    rows = [
        {
            "id": "city-1",
            "node_count": 4,
            "component_count": 1,
            "edge_count_orig": 4,
            "edge_count_mst": 3,
            "edge_count_ratio": 0.75,
            "length_orig": 10.0,
            "length_mst": 6.0,
            "length_ratio": 0.6,
            "circuit_rank": 1,
            "circuit_rank_ratio": 0.25,
        },
        {
            "id": "city-2",
            "node_count": 4,
            "component_count": 2,
            "edge_count_orig": 3,
            "edge_count_mst": 2,
            "edge_count_ratio": 2 / 3,
            "length_orig": 9.0,
            "length_mst": 5.0,
            "length_ratio": 5 / 9,
            "circuit_rank": 1,
            "circuit_rank_ratio": 1 / 3,
        },
    ]
    with TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "results.csv"
        indicators_path = Path(temporary_directory) / "indicators.csv"
        # The indicator table defines exactly the two eligible IDs expected in output.
        with output_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=osmnx_analysis.RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        indicators_path.write_text("uc_id,node_count\n1,101\n2,102\n")
        osmnx_analysis.validate_output(output_path, indicators_path)
        # Break the circuit-rank identity while leaving all other fields unchanged.
        rows[0]["circuit_rank"] = 2
        with output_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=osmnx_analysis.RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        try:
            osmnx_analysis.validate_output(output_path, indicators_path)
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid circuit rank was accepted")


def test_derived_columns_references_and_strict_deciles() -> None:
    """Check v1-derived variables, analytical references, and strict tail masks.

    Notes
    -----
    The one-row frame makes each unit conversion visible.  The repeated decile values
    verify that observations equal to a percentile are excluded by the strict masks.
    """
    # Keep source quantities simple so every expected derived value can be checked by
    # inspection rather than by duplicating the implementation's arithmetic.
    frame = pd.DataFrame(
        {
            "length_ratio": [0.4],
            "circuit_rank_ratio": [0.25],
            "length_total": [2_000.0],
            "intersect_count": [10.0],
            "built_up_area_m2": [1_000_000.0],
            "resident_pop": [1_000.0],
            "gdp_ppp": [100_000.0],
        }
    )
    derived = revision.add_derived_columns(frame, include_cc=False)
    assert derived.loc[0, "length_redundancy"] == 0.6
    assert derived.loc[0, "link_redundancy"] == 0.25
    assert derived.loc[0, "length_total_km"] == 2
    assert derived.loc[0, "intersect_density_km2"] == 10
    assert derived.loc[0, "built_up_area_km2"] == 1
    assert derived.loc[0, "pop_density_km2_1000s"] == 1
    assert derived.loc[0, "gdp_ppp_percap"] == 100
    assert set(dict(revision.STREET_NETWORK_REFERENCES).values()) == {0, 1 / 3, 1 / 2, 2 / 3}
    assert revision.manuscript_label("zhenhai_district") == "Zhenhai District"
    assert revision.manuscript_label("mexico") == "Mexico"
    assert (
        revision.display_city(pd.Series({"core_city": "unnamed", "uc_names": "شهرک صنعتی اشتهارد"}))
        == "شهرک صنعتی اشتهارد"
    )

    contrast_frame = pd.DataFrame(
        {
            "theta": [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9],
            "bc_gini": list(range(10, 22)),
        }
    )
    contrasts = revision.calculate_bc_gini_contrasts(contrast_frame, "theta")
    assert contrasts == {"p10": 1.0, "p90": 8.0, "low_median": 10.0, "high_median": 21.0}


def test_load_representation_filters_source_node_count() -> None:
    """Check loading when the v1 result CSV has no result-side node count.

    Notes
    -----
    The source indicator row at exactly 100 nodes must be excluded before merging.
    This also protects the v1 inputs, whose result CSV does not contain a result-side
    ``node_count`` column.
    """
    indicators = pd.DataFrame(
        {
            "uc_id": [1, 2],
            "node_count": [101, 100],
            "length_total": [2_000.0, 3_000.0],
            "intersect_count": [10.0, 15.0],
            "built_up_area_m2": [1_000_000.0, 1_000_000.0],
            "resident_pop": [1_000.0, 1_000.0],
            "gdp_ppp": [100_000.0, 100_000.0],
        }
    )
    results = pd.DataFrame(
        {
            "id": ["city-1", "city-2"],
            "length_ratio": [0.4, 0.5],
            "circuit_rank_ratio": [0.25, 0.3],
        }
    )
    with TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        indicators_path = directory / "indicators.csv"
        results_path = directory / "results.csv"
        indicators.to_csv(indicators_path, index=False)
        results.to_csv(results_path, index=False)
        loaded = revision.load_representation(results_path, indicators_path, include_cc=False)

    assert loaded["uc_id"].tolist() == [1]
    assert loaded["node_count"].tolist() == [101]


def test_revision_output_schemas_and_unique_items() -> None:
    """Check the values and 36-row correlation output schemas.

    Notes
    -----
    Temporary CSVs verify the machine-readable contracts without running the
    production statistics script or requiring its large source data files.
    """
    # Build the exact robust-row cardinality: two representations times two measures
    # times nine comparison variables.
    values = [{"item": "sample_size", "value": 3}, {"item": "theta_mean", "value": 0.2}]
    correlations = []
    for representation in ("neatnet", "osmnx"):
        for redundancy in ("theta", "lambda"):
            for comparison in revision.COMPARISON_COLUMNS:
                correlations.append(
                    {
                        "representation": representation,
                        "redundancy": redundancy,
                        "comparison": comparison,
                        "r": 0.1,
                        "p": 0.2,
                        "sign_changed_from_primary": False,
                        "significance_changed_from_primary": False,
                    }
                )

    with TemporaryDirectory() as temporary_directory:
        values_path = Path(temporary_directory) / "values.csv"
        correlations_path = Path(temporary_directory) / "correlations.csv"
        revision.write_values(values_path, values)
        revision.write_correlations(correlations_path, correlations)
        values_frame = pd.read_csv(values_path)
        correlations_frame = pd.read_csv(correlations_path)

    assert list(values_frame.columns) == ["item", "value"]
    assert values_frame["item"].is_unique
    assert list(correlations_frame.columns) == revision.CORRELATION_FIELDS
    assert len(correlations_frame) == 36
    assert correlations_frame["representation"].value_counts().to_dict() == {
        "neatnet": 18,
        "osmnx": 18,
    }
