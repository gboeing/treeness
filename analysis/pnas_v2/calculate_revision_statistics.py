#!/usr/bin/env python
"""Calculate revision statistics and robustness summaries."""

import csv
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

INDICATORS_PATH = Path("analysis/data/indicators-all.csv")
GDP_PATH = Path("/data/snm/inputs/GHS_UCDB_GLOBE_R2024A_V1_1/GHS_UCDB_GLOBE_R2024A.gpkg")
ORIGINAL_PATH = Path("analysis/data/treeness-results.csv")
NEATNET_PATH = Path("analysis/data/treeness-results_neat.csv")
OSMNX_PATH = Path("analysis/data/treeness-results_osmnx.csv")
VALUES_PATH = Path("analysis/data/pnas-v2-values.csv")
CORRELATIONS_PATH = Path("analysis/data/pnas-v2-correlations.csv")
SUMMARY_PATH = Path("analysis/data/pnas-v2-summary.txt")
FIGURE_1_IDS = (10309, 11569, 9553, 11088, 7100, 5345)
CORRELATION_FIELDS = [
    "representation",
    "redundancy",
    "comparison",
    "r",
    "p",
    "sign_changed_from_primary",
    "significance_changed_from_primary",
]
COMPARISON_COLUMNS = (
    "k_avg",
    "bc_gini",
    "circuity",
    "gdp_ppp_percap_log",
    "intersect_density_km2",
    "length_total_km_log",
    "elev_std",
    "pop_density_km2_1000s",
    "built_up_area_km2_log",
)
STREET_NETWORK_REFERENCES = (
    ("Tree", 0.0),
    ("Hexagonal lattice", 1 / 3),
    ("Square lattice", 1 / 2),
    ("Triangular lattice", 2 / 3),
)


def parse_uc_id(path: Path | str) -> int:
    """Extract the urban centre ID from the last filename component.

    Parameters
    ----------
    path : pathlib.Path or str
        Result filename or path whose final stem component is the UC ID.

    Returns
    -------
    int
        Urban centre ID.

    Notes
    -----
    City names can contain hyphens, so the ID is read from the final part.
    """
    return int(Path(str(path)).stem.rsplit("-", 1)[1])


def load_gdp_inputs(gpkg_path: Path = GDP_PATH) -> pd.DataFrame:
    """Read 2020 GDP and population totals from GHSL R2024A V1.1.

    Parameters
    ----------
    gpkg_path : pathlib.Path
        V1.1 GeoPackage containing the socioeconomic and GHSL tables.

    Returns
    -------
    pandas.DataFrame
        Unique ``uc_id`` rows with ``gdp_ppp_total_2020`` (2017 international
        dollars) and ``resident_pop_2020`` (persons, retaining fractional
        values).

    Raises
    ------
    ValueError
        If IDs are null, duplicated, or unmatched, or totals are not positive
        and
        finite. Duplicate IDs raise pandas' ``MergeError``, a ``ValueError``
        subtype.

    Notes
    -----
    GDP is in 2017 international dollars (Zenodo record 10976734).
    Both totals use fixed 2025 urban centre boundaries. Population retains
    fractional values.
    """
    frames = []
    # strip the leading byte-order marks from V1.1 column names
    connection = sqlite3.connect(gpkg_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for table, source, target in (
            ("GHS_UCDB_THEME_SOCIOECONOMIC_GLOBE_R2024A", "SC_GDP_SUM_2020", "gdp_ppp_total_2020"),
            ("GHS_UCDB_THEME_GHSL_GLOBE_R2024A", "GH_POP_TOT_2020", "resident_pop_2020"),
        ):
            columns = {
                row[1].lstrip("\ufeff"): row[1]
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            }
            frame = pd.read_sql_query(
                f'SELECT "{columns["ID_UC_G0"]}" AS uc_id, '
                f'"{columns[source]}" AS "{target}" FROM "{table}"',
                connection,
            )
            if frame["uc_id"].isna().any():
                raise ValueError(f"Null GHSL urban centre IDs in {table}")
            frames.append(frame)
    finally:
        connection.close()

    # check that both tables contain the same unique urban centre IDs
    totals = frames[0].merge(
        frames[1], on="uc_id", how="outer", validate="one_to_one", indicator=True
    )
    if not totals["_merge"].eq("both").all():
        raise ValueError("GDP and population urban centre IDs do not match")
    totals = totals.drop(columns="_merge")
    values = totals[["gdp_ppp_total_2020", "resident_pop_2020"]].astype(float)
    if not (np.isfinite(values) & values.gt(0)).all().all():
        raise ValueError("2020 GDP and population totals must be positive and finite")
    return totals


def add_derived_columns(frame: pd.DataFrame, include_cc: bool) -> pd.DataFrame:
    """Calculate redundancy, densities, GDP per capita, and logs.

    Parameters
    ----------
    frame : pandas.DataFrame
        Indicator and treeness-result columns for one representation.
    include_cc : bool
        Whether the result has largest-connected-component columns.

    Returns
    -------
    pandas.DataFrame
        Copy of ``frame`` with the required derived variables.

    Notes
    -----
    GDP per capita uses 2020 GDP and population totals. Population density
    uses 2025 population. Lengths are in km, areas in km2, and population
    density in thousands of persons per km2. Logs are natural logarithms.
    """
    derived = frame.copy()
    # calculate redundancy for the full graph and largest component
    derived["length_redundancy"] = 1 - derived["length_ratio"]
    derived["link_redundancy"] = derived["circuit_rank_ratio"]
    if include_cc:
        derived["length_redundancy_cc"] = 1 - derived["length_ratio_cc"]
        derived["link_redundancy_cc"] = derived["circuit_rank_ratio_cc"]
    # convert lengths to km and areas to km2 before calculating densities
    derived["length_total_km"] = derived["length_total"] / 1_000
    derived["intersect_density_km2"] = (
        derived["intersect_count"] / derived["built_up_area_m2"] * 1_000_000
    )
    derived["built_up_area_km2"] = derived["built_up_area_m2"] / 1_000_000
    derived["pop_density_km2_1000s"] = (derived["resident_pop"] / 1_000) / derived[
        "built_up_area_km2"
    ]
    # divide 2020 GDP by 2020 population within the same urban boundary
    # keep 2025 population for the population density calculation
    values = derived[["gdp_ppp_total_2020", "resident_pop_2020"]].astype(float)
    if not (np.isfinite(values) & values.gt(0)).all().all():
        raise ValueError("2020 GDP and population totals must be positive and finite")
    derived["gdp_ppp_percap"] = derived["gdp_ppp_total_2020"] / derived["resident_pop_2020"]
    if not (np.isfinite(derived["gdp_ppp_percap"]) & derived["gdp_ppp_percap"].gt(0)).all():
        raise ValueError("2020 GDP per capita must be positive and finite")
    # take natural logs for the correlation analysis
    derived["built_up_area_km2_log"] = np.log(derived["built_up_area_km2"])
    derived["length_total_km_log"] = np.log(derived["length_total_km"])
    derived["gdp_ppp_percap_log"] = np.log(derived["gdp_ppp_percap"])
    return derived


def calculate_bc_gini_contrasts(
    frame: pd.DataFrame,
    redundancy_column: str,
) -> dict[str, float]:
    """Calculate median betweenness Gini in the redundancy deciles.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame containing the redundancy and ``bc_gini`` columns.
    redundancy_column : str
        Redundancy column used to define the two tails.

    Returns
    -------
    dict[str, float]
        The two thresholds and the two tail medians.

    Notes
    -----
    Values exactly on the 10th or 90th percentile are excluded.
    """
    # exclude values exactly on either decile threshold
    lower = frame[redundancy_column].quantile(0.1)
    upper = frame[redundancy_column].quantile(0.9)
    lower_mask = frame[redundancy_column] < lower
    upper_mask = frame[redundancy_column] > upper
    return {
        "p10": float(lower),
        "p90": float(upper),
        "low_median": float(frame.loc[lower_mask, "bc_gini"].median()),
        "high_median": float(frame.loc[upper_mask, "bc_gini"].median()),
    }


def load_representation(
    results_path: Path,
    indicators_path: Path,
    include_cc: bool,
    gdp_path: Path = GDP_PATH,
) -> pd.DataFrame:
    """Load and prepare one network representation.

    Parameters
    ----------
    results_path : pathlib.Path
        Treeness result CSV for the representation.
    indicators_path : pathlib.Path
        Source indicator CSV.
    include_cc : bool
        Whether the result contains largest-connected-component columns.
    gdp_path : pathlib.Path
        GHSL R2024A V1.1 GeoPackage supplying 2020 GDP and population totals.

    Returns
    -------
    pandas.DataFrame
        Matched results with ``node_count > 100`` and derived variables.

    Raises
    ------
    ValueError
        If result IDs are duplicated or indicator IDs lack valid V1.1 GDP
        inputs.
    """
    indicators = pd.read_csv(indicators_path)
    indicators = indicators.merge(
        load_gdp_inputs(gdp_path), on="uc_id", how="left", validate="one_to_one", indicator=True
    )
    if not indicators["_merge"].eq("both").all():
        raise ValueError("Missing V1.1 GDP inputs for indicator urban centre IDs")
    indicators = indicators.drop(columns="_merge")
    # filter on source node count before merging with OSMnx node counts
    indicators = indicators[indicators["node_count"] > 100].copy()
    results = pd.read_csv(results_path)
    # extract urban centre IDs from result filenames
    results["uc_id"] = results["id"].map(parse_uc_id)
    if results["uc_id"].duplicated().any():
        raise ValueError(f"Duplicate UC IDs in {results_path}")
    # match eligible indicators to available results by urban centre ID
    merged = indicators.merge(
        results,
        on="uc_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_indicator", "_result"),
    )
    return add_derived_columns(merged, include_cc)


def calculate_correlations(
    representation: str,
    frame: pd.DataFrame,
) -> list[dict[str, object]]:
    """Calculate Pearson correlations for both redundancy indicators.

    Parameters
    ----------
    representation : str
        Name written to the correlation output.
    frame : pandas.DataFrame
        Representation frame containing the two redundancy variables and
        comparisons.

    Returns
    -------
    list[dict[str, object]]
        Eighteen correlation rows with comparison flags initially set to
        ``False``.

    Notes
    -----
    Each p-value is two-sided and uses complete pairs of observations.
    """
    rows: list[dict[str, object]] = []
    for redundancy, redundancy_column in (
        ("theta", "link_redundancy"),
        ("lambda", "length_redundancy"),
    ):
        for comparison in COMPARISON_COLUMNS:
            # drop missing values separately for each pair of variables
            values = frame[[redundancy_column, comparison]].dropna()
            result = linregress(values[comparison], values[redundancy_column])
            rows.append(
                {
                    "representation": representation,
                    "redundancy": redundancy,
                    "comparison": comparison,
                    "r": result.rvalue,
                    "p": result.pvalue,
                    "sign_changed_from_primary": False,
                    "significance_changed_from_primary": False,
                }
            )
    return rows


def compare_correlations(
    rows: list[dict[str, object]],
    primary_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Compare correlation signs and significance with the original.

    Parameters
    ----------
    rows : list[dict[str, object]]
        Correlations for neatnet or OSMnx.
    primary_rows : list[dict[str, object]]
        Correlations for the original representation.

    Returns
    -------
    list[dict[str, object]]
        Copies of ``rows`` with both comparison flags filled in.
    """
    # match each correlation to the same pair in the original representation
    primary_by_key = {(str(row["redundancy"]), str(row["comparison"])): row for row in primary_rows}
    compared: list[dict[str, object]] = []
    for row in rows:
        key = (str(row["redundancy"]), str(row["comparison"]))
        primary = primary_by_key[key]
        current = row.copy()
        # compare correlation signs and significance at p < 0.05 separately
        current["sign_changed_from_primary"] = np.sign(float(current["r"])) != np.sign(
            float(primary["r"])
        )
        current["significance_changed_from_primary"] = (float(current["p"]) < 0.05) != (
            float(primary["p"]) < 0.05
        )
        compared.append(current)
    return compared


def normalized_name(value: object) -> str:
    """Normalize case, accents, and separators in a location name.

    Parameters
    ----------
    value : object
        Location value from a source dataframe.

    Returns
    -------
    str
        Case-folded alphanumeric name suitable for exact comparison.
    """
    # remove accents, punctuation, and separators for name comparisons
    decomposed = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(character for character in decomposed if character.isalnum())


def manuscript_label(value: object) -> str:
    """Format a source identifier as a readable label.

    Parameters
    ----------
    value : object
        Source label that may use underscores or hyphens as separators.

    Returns
    -------
    str
        Title-cased label with word separators replaced by spaces.
    """
    # replace identifier separators with spaces for display
    label = str(value).replace("_", " ").replace("-", " ")
    return " ".join(label.split()).title()


def display_city(row: pd.Series) -> str:
    """Select a city name for display.

    Parameters
    ----------
    row : pandas.Series
        One merged representation row.

    Returns
    -------
    str
        Manuscript-ready city label for the revision outputs.
    """
    city = str(row["core_city"]).strip()
    uc_names_value = row["uc_names"]
    uc_names = "" if pd.isna(uc_names_value) else str(uc_names_value).strip()
    # use the urban centre name if the core city is unnamed
    if normalized_name(city) == "unnamed" and uc_names:
        return uc_names
    # use a single urban centre name, or fall back to the core city
    if uc_names and ";" not in uc_names:
        return uc_names
    # format the core city identifier as a readable label
    return manuscript_label(city)


def validate_existing_values(original: pd.DataFrame, neatnet: pd.DataFrame) -> None:
    """Check original and neatnet results against reference values.

    Parameters
    ----------
    original : pandas.DataFrame
        Prepared original representation frame.
    neatnet : pandas.DataFrame
        Prepared neatnet representation frame.

    Raises
    ------
    AssertionError
        If a stated sample, value, location, or correlation target differs.

    Returns
    -------
    None
        No return value.
    """
    # check sample sizes and mean redundancy
    assert len(original) == 9_923
    assert original["uc_id"].nunique() == 9_923
    assert len(neatnet) == 9_921
    assert neatnet["uc_id"].nunique() == 9_921
    assert np.isclose(original["link_redundancy"].mean(), 0.281, atol=0.0005)
    assert np.isclose(original["length_redundancy"].mean(), 0.442, atol=0.0005)
    assert np.isclose(neatnet["link_redundancy"].mean(), 0.256, atol=0.0005)
    assert np.isclose(neatnet["length_redundancy"].mean(), 0.407, atol=0.0005)

    # check the locations and values of the original sample extrema
    extrema = {
        ("theta", "min"): (
            original["link_redundancy"].idxmin(),
            6970,
            "comilla",
            "bangladesh",
            0.030612,
        ),
        ("theta", "max"): (
            original["link_redundancy"].idxmax(),
            1462,
            "san luis rio colorado",
            "mexico",
            0.457933,
        ),
        ("lambda", "min"): (
            original["length_redundancy"].idxmin(),
            6970,
            "comilla",
            "bangladesh",
            0.023763,
        ),
        ("lambda", "max"): (
            original["length_redundancy"].idxmax(),
            4612,
            "شهرک صنعتی اشتهارد",
            "iran",
            0.706681,
        ),
    }
    for (redundancy, direction), (index, uc_id, city, country, value) in extrema.items():
        row = original.loc[index]
        column = "link_redundancy" if redundancy == "theta" else "length_redundancy"
        assert int(row["uc_id"]) == uc_id
        assert normalized_name(display_city(row)) == normalized_name(city)
        assert normalized_name(row["country"]) == normalized_name(country)
        assert np.isclose(float(row[column]), value, atol=1e-6)
        assert direction in {"min", "max"}

    # check the full-network values used in the example figure caption
    examples = {4443: (0.414606, 0.554694), 2315: (0.130825, 0.290192)}
    for uc_id, (theta, length) in examples.items():
        row = original.loc[original["uc_id"] == uc_id].iloc[0]
        assert np.isclose(row["link_redundancy"], theta, atol=1e-6)
        assert np.isclose(row["length_redundancy"], length, atol=1e-6)

    # check betweenness Gini medians in the upper and lower deciles
    theta_contrasts = calculate_bc_gini_contrasts(original, "link_redundancy")
    lambda_contrasts = calculate_bc_gini_contrasts(original, "length_redundancy")
    assert np.isclose(theta_contrasts["p10"], 0.186365, atol=1e-6)
    assert np.isclose(theta_contrasts["p90"], 0.363803, atol=1e-6)
    assert np.isclose(lambda_contrasts["p10"], 0.317129, atol=1e-6)
    assert np.isclose(lambda_contrasts["p90"], 0.549653, atol=1e-6)
    assert np.isclose(theta_contrasts["low_median"], 0.766992, atol=1e-6)
    assert np.isclose(theta_contrasts["high_median"], 0.578241, atol=1e-6)
    assert np.isclose(lambda_contrasts["low_median"], 0.739050, atol=1e-6)
    assert np.isclose(lambda_contrasts["high_median"], 0.628061, atol=1e-6)

    # check largest-component values in Figure 1 order
    for uc_id, city, country, theta, length in (
        (10309, "xiangtan", "china", 0.397, 0.652),
        (11569, "zhenhai district", "china", 0.411, 0.641),
        (9553, "wudi", "china", 0.403, 0.623),
        (11088, "baisi", "india", 0.080, 0.174),
        (7100, "akhaura", "bangladesh", 0.112, 0.153),
        (5345, "sukasari", "indonesia", 0.073, 0.132),
    ):
        row = original.loc[original["uc_id"] == uc_id].iloc[0]
        assert normalized_name(row["core_city"]) == normalized_name(city)
        assert normalized_name(row["country"]) == normalized_name(country)
        assert np.isclose(row["link_redundancy_cc"], theta, atol=0.0005)
        assert np.isclose(row["length_redundancy_cc"], length, atol=0.0005)

    # check correlation changes between the original and neatnet results
    primary = calculate_correlations("original", original)
    robust = compare_correlations(calculate_correlations("neatnet", neatnet), primary)
    primary_by_key = {(row["redundancy"], row["comparison"]): row for row in primary}
    robust_by_key = {(row["redundancy"], row["comparison"]): row for row in robust}
    for key, row in robust_by_key.items():
        if key == ("lambda", "elev_std"):
            assert np.isclose(primary_by_key[key]["r"], -0.020, atol=0.0005)
            assert np.isclose(primary_by_key[key]["p"], 0.045, atol=0.0005)
            assert np.isclose(row["r"], 0.012, atol=0.0005)
            assert np.isclose(row["p"], 0.218, atol=0.0005)
            assert row["sign_changed_from_primary"]
            assert row["significance_changed_from_primary"]
        else:
            assert not row["sign_changed_from_primary"]
            assert not row["significance_changed_from_primary"]


def build_values(
    original: pd.DataFrame,
    neatnet: pd.DataFrame,
    osmnx: pd.DataFrame,
) -> list[dict[str, object]]:
    """Collect reference values and results for the revision.

    Parameters
    ----------
    original : pandas.DataFrame
        Prepared original representation frame.
    neatnet : pandas.DataFrame
        Prepared neatnet representation frame.
    osmnx : pandas.DataFrame
        Prepared OSMnx representation frame.

    Returns
    -------
    list[dict[str, object]]
        Unique ``item`` and ``value`` rows for the v2 values CSV.
    """
    rows: list[dict[str, object]] = []
    # store lattice references and representation summaries
    for item, value in (
        ("reference_tree_theta", 0),
        ("reference_tree_lambda", 0),
        ("reference_hexagonal_theta", 1 / 3),
        ("reference_hexagonal_lambda", 1 / 3),
        ("reference_square_theta", 1 / 2),
        ("reference_square_lambda", 1 / 2),
        ("reference_triangular_theta", 2 / 3),
        ("reference_triangular_lambda", 2 / 3),
        ("original_sample_size", len(original)),
        ("neatnet_sample_size", len(neatnet)),
        ("osmnx_sample_size", len(osmnx)),
        ("original_theta_mean", original["link_redundancy"].mean()),
        ("original_lambda_mean", original["length_redundancy"].mean()),
        ("neatnet_theta_mean", neatnet["link_redundancy"].mean()),
        ("neatnet_lambda_mean", neatnet["length_redundancy"].mean()),
        ("osmnx_theta_mean", osmnx["link_redundancy"].mean()),
        ("osmnx_lambda_mean", osmnx["length_redundancy"].mean()),
        (
            "neatnet_theta_mean_change",
            neatnet["link_redundancy"].mean() - original["link_redundancy"].mean(),
        ),
        (
            "neatnet_lambda_mean_change",
            neatnet["length_redundancy"].mean() - original["length_redundancy"].mean(),
        ),
        (
            "osmnx_theta_mean_change",
            osmnx["link_redundancy"].mean() - original["link_redundancy"].mean(),
        ),
        (
            "osmnx_lambda_mean_change",
            osmnx["length_redundancy"].mean() - original["length_redundancy"].mean(),
        ),
    ):
        rows.append({"item": item, "value": value})

    # calculate decile thresholds and betweenness Gini medians
    theta_contrasts = calculate_bc_gini_contrasts(original, "link_redundancy")
    lambda_contrasts = calculate_bc_gini_contrasts(original, "length_redundancy")
    for item, value in (
        ("theta_p10", theta_contrasts["p10"]),
        ("theta_p90", theta_contrasts["p90"]),
        ("theta_low_bc_gini_median", theta_contrasts["low_median"]),
        ("theta_high_bc_gini_median", theta_contrasts["high_median"]),
        ("lambda_p10", lambda_contrasts["p10"]),
        ("lambda_p90", lambda_contrasts["p90"]),
        ("lambda_low_bc_gini_median", lambda_contrasts["low_median"]),
        ("lambda_high_bc_gini_median", lambda_contrasts["high_median"]),
    ):
        rows.append({"item": item, "value": value})

    # select the two example cities by urban centre ID
    examples = (("buenos_aires", 4443), ("bangkok", 2315))
    for label, uc_id in examples:
        row = original.loc[original["uc_id"] == uc_id].iloc[0]
        rows.extend(
            [
                {"item": f"{label}_theta", "value": row["link_redundancy"]},
                {"item": f"{label}_lambda", "value": row["length_redundancy"]},
            ]
        )

    # record the extrema and their city labels
    extrema = (
        ("theta_min", "link_redundancy", "min"),
        ("theta_max", "link_redundancy", "max"),
        ("lambda_min", "length_redundancy", "min"),
        ("lambda_max", "length_redundancy", "max"),
    )
    for label, column, direction in extrema:
        index = original[column].idxmin() if direction == "min" else original[column].idxmax()
        row = original.loc[index]
        rows.extend(
            [
                {"item": f"{label}_uc_id", "value": int(row["uc_id"])},
                {"item": f"{label}_city", "value": display_city(row)},
                {"item": f"{label}_country", "value": manuscript_label(row["country"])},
                {"item": f"{label}_value", "value": row[column]},
            ]
        )

    # record largest-component values in Figure 1 order
    for uc_id in FIGURE_1_IDS:
        row = original.loc[original["uc_id"] == uc_id].iloc[0]
        rows.extend(
            [
                {"item": f"figure_1_{uc_id}_city", "value": display_city(row)},
                {"item": f"figure_1_{uc_id}_country", "value": manuscript_label(row["country"])},
                {"item": f"figure_1_{uc_id}_theta_cc", "value": row["link_redundancy_cc"]},
                {"item": f"figure_1_{uc_id}_lambda_cc", "value": row["length_redundancy_cc"]},
            ]
        )
    # check that each output item has a unique name
    assert len({str(row["item"]) for row in rows}) == len(rows)
    return rows


def write_values(path: Path, rows: list[dict[str, object]]) -> None:
    """Write revision values to a two-column CSV.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    rows : list[dict[str, object]]
        Rows containing unique ``item`` and ``value`` keys.

    Returns
    -------
    None
        The validated rows are written to ``path``.
    """
    # check column order and unique item names before writing
    fieldnames = ["item", "value"]
    assert all(list(row) == fieldnames for row in rows)
    assert len({str(row["item"]) for row in rows}) == len(rows)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_correlations(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the 36 robustness correlations to a CSV.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    rows : list[dict[str, object]]
        Eighteen neatnet and eighteen OSMnx rows.

    Returns
    -------
    None
        The validated 36 rows are written to ``path``.
    """
    # check row counts and column order before writing
    assert len(rows) == 36
    assert sum(row["representation"] == "neatnet" for row in rows) == 18
    assert sum(row["representation"] == "osmnx" for row in rows) == 18
    assert all(list(row) == CORRELATION_FIELDS for row in rows)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CORRELATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    original: pd.DataFrame,
    neatnet: pd.DataFrame,
    osmnx: pd.DataFrame,
    correlations: list[dict[str, object]],
) -> str:
    """Summarize reference values, results, and robustness checks.

    Parameters
    ----------
    original : pandas.DataFrame
        Prepared original representation frame.
    neatnet : pandas.DataFrame
        Prepared neatnet representation frame.
    osmnx : pandas.DataFrame
        Prepared OSMnx representation frame.
    correlations : list[dict[str, object]]
        Compared neatnet and OSMnx correlation rows.

    Returns
    -------
    str
        Labeled summary text ending with a newline.
    """
    # state the equal-edge-length assumption for lattice lambda values
    lines = [
        "Street-network references",
        "Analytical lattice lambda values assume unit-length edges.",
    ]
    for label, value in STREET_NETWORK_REFERENCES:
        # report both indicators for each reference network
        lines.append(f"{label}: theta={value:.6f}; lambda={value:.6f} (unit-length edges)")

    lines.append("")
    lines.append("Figure 1 cases")
    for uc_id in FIGURE_1_IDS:
        # report largest-component values in Figure 1 order
        row = original.loc[original["uc_id"] == uc_id].iloc[0]
        lines.append(
            f"UC {uc_id}: {display_city(row)}, {manuscript_label(row['country'])}; "
            f"theta_CC={row['link_redundancy_cc']:.3f}; "
            f"lambda_CC={row['length_redundancy_cc']:.3f}"
        )

    lines.append("")
    lines.append("Full-network examples")
    # report full-network values for the two example cities
    for label, uc_id in (("Buenos Aires", 4443), ("Bangkok", 2315)):
        row = original.loc[original["uc_id"] == uc_id].iloc[0]
        lines.append(
            f"{label} (UC {uc_id}): theta={row['link_redundancy']:.6f}; "
            f"lambda={row['length_redundancy']:.6f}"
        )

    lines.append("")
    lines.append("Empirical extrema")
    # report extrema with city labels and urban centre IDs
    for label, column in (
        ("theta minimum", "link_redundancy"),
        ("theta maximum", "link_redundancy"),
        ("lambda minimum", "length_redundancy"),
        ("lambda maximum", "length_redundancy"),
    ):
        index = original[column].idxmin() if "minimum" in label else original[column].idxmax()
        row = original.loc[index]
        lines.append(
            f"{label}: {display_city(row)}, {manuscript_label(row['country'])}, "
            f"UC {int(row['uc_id'])}; "
            f"value={row[column]:.6f}"
        )

    lines.append("")
    lines.append("bc_gini contrasts")
    # report decile thresholds and betweenness Gini medians
    for label, column in (("theta", "link_redundancy"), ("lambda", "length_redundancy")):
        contrasts = calculate_bc_gini_contrasts(original, column)
        lines.append(
            f"{label}: p10={contrasts['p10']:.6f}; p90={contrasts['p90']:.6f}; "
            f"low_median={contrasts['low_median']:.6f}; "
            f"high_median={contrasts['high_median']:.6f}"
        )

    lines.append("")
    lines.append("Representation summaries")
    # report sample sizes, means, and changes from the original representation
    for label, frame in (("original", original), ("neatnet", neatnet), ("osmnx", osmnx)):
        summary = (
            f"{label}: n={len(frame)}; theta_mean={frame['link_redundancy'].mean():.6f}; "
            f"lambda_mean={frame['length_redundancy'].mean():.6f}"
        )
        if label != "original":
            theta_change = frame["link_redundancy"].mean() - original["link_redundancy"].mean()
            lambda_change = frame["length_redundancy"].mean() - original["length_redundancy"].mean()
            summary += (
                f"; theta_mean_change={theta_change:.6f}; lambda_mean_change={lambda_change:.6f}"
            )
        lines.append(summary)

    lines.append("")
    lines.append("Correlation sign or significance changes")
    for representation in ("neatnet", "osmnx"):
        # report correlations whose sign or significance changed
        changed = [
            row
            for row in correlations
            if row["representation"] == representation
            and (row["sign_changed_from_primary"] or row["significance_changed_from_primary"])
        ]
        if not changed:
            lines.append(f"{representation}: none")
        else:
            for row in changed:
                lines.append(
                    f"{representation} {row['redundancy']} vs {row['comparison']}: "
                    f"r={float(row['r']):.6f}; p={float(row['p']):.6f}; "
                    f"sign_changed={row['sign_changed_from_primary']}; "
                    f"significance_changed={row['significance_changed_from_primary']}"
                )
    return "\n".join(lines) + "\n"


def write_summary(path: Path, summary: str) -> None:
    """Write the text summary.

    Parameters
    ----------
    path : pathlib.Path
        Summary text path.
    summary : str
        Labeled summary content.

    Returns
    -------
    None
        The prepared summary is written to ``path``.
    """
    path.write_text(summary)


def main() -> None:
    """Calculate and save revision statistics.

    Returns
    -------
    None
        No return value.
    """
    # load and prepare each representation
    original = load_representation(ORIGINAL_PATH, INDICATORS_PATH, include_cc=True)
    neatnet = load_representation(NEATNET_PATH, INDICATORS_PATH, include_cc=True)
    osmnx = load_representation(OSMNX_PATH, INDICATORS_PATH, include_cc=False)
    # check the source results before writing summaries
    validate_existing_values(original, neatnet)
    assert len(osmnx) == 9_923
    assert osmnx["uc_id"].nunique() == 9_923

    # compare both alternative representations with the original
    primary = calculate_correlations("original", original)
    correlations = compare_correlations(calculate_correlations("neatnet", neatnet), primary)
    correlations.extend(compare_correlations(calculate_correlations("osmnx", osmnx), primary))
    values = build_values(original, neatnet, osmnx)
    # write values, correlations, and the text summary
    write_values(VALUES_PATH, values)
    write_correlations(CORRELATIONS_PATH, correlations)
    write_summary(SUMMARY_PATH, build_summary(original, neatnet, osmnx, correlations))


if __name__ == "__main__":
    main()
