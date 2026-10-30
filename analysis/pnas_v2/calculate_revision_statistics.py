#!/usr/bin/env python
"""Calculate the v2 revision values and robustness summaries."""

import csv
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

INDICATORS_PATH = Path("analysis/data/indicators-all.csv")
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
    """Parse a UC ID from the final hyphen-delimited filename component.

    Result filenames include descriptive prefixes that may contain hyphens.  The
    final stem component is therefore the only stable location for the numeric ID.

    Parameters
    ----------
    path : pathlib.Path or str
        Result filename or path whose final stem component is the UC ID.

    Returns
    -------
    int
        Parsed UC ID.

    Notes
    -----
    The same convention is used by the OSMnx result writer.  Parsing it here allows
    all representations to merge on the source indicator table without relying on
    directory names or the spelling of a city.
    """
    return int(Path(str(path)).stem.rsplit("-", 1)[1])


def add_derived_columns(frame: pd.DataFrame, include_cc: bool) -> pd.DataFrame:
    """Add the v1-derived analysis variables to a copy of a merged frame.

    The formulas are kept explicit so a reviewer can compare each unit conversion
    and redundancy definition with the v1 analysis and with the revision plan.

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
    Lengths are converted from metres to kilometres, area from square metres to
    square kilometres, and population density is reported in thousands per square
    kilometre.  Log columns are derived only after those physical quantities are
    formed, which keeps the units visible and consistent.
    """
    derived = frame.copy()
    # Lambda is the complement of the MST length ratio; theta is the normalized
    # circuit rank.  The connected-component versions exist only in v1/neatnet.
    derived["length_redundancy"] = 1 - derived["length_ratio"]
    derived["link_redundancy"] = derived["circuit_rank_ratio"]
    if include_cc:
        derived["length_redundancy_cc"] = 1 - derived["length_ratio_cc"]
        derived["link_redundancy_cc"] = derived["circuit_rank_ratio_cc"]
    # Convert source totals before calculating densities or logarithms so downstream
    # correlation names describe their actual units.
    derived["length_total_km"] = derived["length_total"] / 1_000
    derived["intersect_density_km2"] = (
        derived["intersect_count"] / derived["built_up_area_m2"] * 1_000_000
    )
    derived["built_up_area_km2"] = derived["built_up_area_m2"] / 1_000_000
    derived["pop_density_km2_1000s"] = (derived["resident_pop"] / 1_000) / derived[
        "built_up_area_km2"
    ]
    derived["gdp_ppp_percap"] = (
        derived["gdp_ppp"] * derived["built_up_area_km2"] / derived["resident_pop"]
    )
    # The three logs match the variables used in the existing correlation analysis;
    # no epsilon or imputation is added because that would change the analysis.
    derived["built_up_area_km2_log"] = np.log(derived["built_up_area_km2"])
    derived["length_total_km_log"] = np.log(derived["length_total_km"])
    derived["gdp_ppp_percap_log"] = np.log(derived["gdp_ppp_percap"])
    return derived


def calculate_bc_gini_contrasts(
    frame: pd.DataFrame,
    redundancy_column: str,
) -> dict[str, float]:
    """Calculate strict lower and upper decile ``bc_gini`` contrasts.

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
    Pandas' 10th and 90th percentiles define the thresholds.  Strict comparisons
    exclude observations exactly on a threshold, as required for the practical
    low- and high-redundancy contrasts.
    """
    # Keep threshold observations out of both tails.  This differs from a rank-based
    # split when repeated redundancy values occur at a decile boundary.
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
) -> pd.DataFrame:
    """Load, merge, filter, and derive variables for one representation.

    Parameters
    ----------
    results_path : pathlib.Path
        Treeness result CSV for the representation.
    indicators_path : pathlib.Path
        Source indicator CSV.
    include_cc : bool
        Whether the result contains largest-connected-component columns.

    Returns
    -------
    pandas.DataFrame
        Merged rows with ``node_count > 100`` and v1-derived variables.

    Raises
    ------
    ValueError
        If result IDs are duplicated.

    Notes
    -----
    Eligibility is filtered on the indicator-side node count before merging.  The
    OSMnx result also has a ``node_count`` column, so filtering after the merge would
    create suffixed columns and could select the wrong measure.  The one-to-one
    merge keeps each eligible UC represented at most once.
    """
    indicators = pd.read_csv(indicators_path)
    # Filter on the source graph size before merging because only OSMnx results
    # also contain a result-side ``node_count`` column.
    indicators = indicators[indicators["node_count"] > 100].copy()
    results = pd.read_csv(results_path)
    # Parse the final filename component before merging because result IDs are names,
    # while the indicator table uses integer UC IDs.
    results["uc_id"] = results["id"].map(parse_uc_id)
    if results["uc_id"].duplicated().any():
        raise ValueError(f"Duplicate UC IDs in {results_path}")
    # An inner, one-to-one merge removes ineligible or missing result rows without
    # inventing observations and catches duplicate indicator IDs if they exist.
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
    """Calculate the 18 prescribed Pearson correlations.

    Parameters
    ----------
    representation : str
        Name written to the correlation output.
    frame : pandas.DataFrame
        Representation frame containing the two redundancy variables and comparisons.

    Returns
    -------
    list[dict[str, object]]
        Eighteen correlation rows with comparison flags initially set to ``False``.

    Notes
    -----
    The original frame is also calculated even though only neatnet and OSMnx rows
    are written.  It supplies the primary correlations needed for the sign and
    significance comparisons.  ``linregress`` returns the Pearson ``r`` and its
    two-sided p-value for each complete pair of columns.
    """
    rows: list[dict[str, object]] = []
    for redundancy, redundancy_column in (
        ("theta", "link_redundancy"),
        ("lambda", "length_redundancy"),
    ):
        for comparison in COMPARISON_COLUMNS:
            # Pairwise dropping preserves the existing analysis for variables with
            # different missing-value patterns instead of dropping whole rows once.
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
    """Add sign and significance changes relative to primary correlations.

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

    Notes
    -----
    Rows are matched by redundancy and comparison name, not by list position.  This
    makes the comparison explicit and prevents a changed column order from attaching
    an original p-value to the wrong robustness result.
    """
    # Index the primary rows once because each robust row must be compared with the
    # same indicator/comparison pair from the original representation.
    primary_by_key = {(str(row["redundancy"]), str(row["comparison"])): row for row in primary_rows}
    compared: list[dict[str, object]] = []
    for row in rows:
        key = (str(row["redundancy"]), str(row["comparison"]))
        primary = primary_by_key[key]
        current = row.copy()
        # A sign change and a p<0.05 change are separate states, so report both
        # rather than treating significance as a proxy for the direction of r.
        current["sign_changed_from_primary"] = np.sign(float(current["r"])) != np.sign(
            float(primary["r"])
        )
        current["significance_changed_from_primary"] = (float(current["p"]) < 0.05) != (
            float(primary["p"]) < 0.05
        )
        compared.append(current)
    return compared


def normalized_name(value: object) -> str:
    """Normalize a location name for case, accents, and separator differences.

    Parameters
    ----------
    value : object
        Location value from a source dataframe.

    Returns
    -------
    str
        Case-folded alphanumeric name suitable for exact comparison.

    Notes
    -----
    This helper is used only for validating known source labels.  Removing accents
    and separators lets a source identifier such as ``san_luis_rio_colorado`` match
    its manuscript spelling without changing the label written to an output.
    """
    # Decompose accents first, then retain letters and numbers so case, punctuation,
    # underscores, and diacritics do not affect a validation comparison.
    decomposed = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(character for character in decomposed if character.isalnum())


def manuscript_label(value: object) -> str:
    """Format a source identifier as a readable manuscript label.

    Parameters
    ----------
    value : object
        Source label that may use underscores or hyphens as separators.

    Returns
    -------
    str
        Title-cased label with word separators replaced by spaces.

    Notes
    -----
    Source tables use identifier-style names in some rows.  Replacing separators and
    title-casing here keeps output labels readable while leaving non-Latin names
    unchanged apart from whitespace normalization.
    """
    # Normalize only presentation spelling; the source value remains unchanged in
    # the input frame and continues to support exact data validation.
    label = str(value).replace("_", " ").replace("-", " ")
    return " ".join(label.split()).title()


def display_city(row: pd.Series) -> str:
    """Choose and format a readable city label from one source row.

    Parameters
    ----------
    row : pandas.Series
        One merged representation row.

    Returns
    -------
    str
        Manuscript-ready city label for the revision outputs.

    Notes
    -----
    ``uc_names`` is the preferred human label when it contains one name, and it is
    required when ``core_city`` is ``unnamed``.  Rows with semicolon-separated UC
    names describe multiple places, so the single core-city identifier is clearer in
    the compact revision output.
    """
    city = str(row["core_city"]).strip()
    uc_names_value = row["uc_names"]
    uc_names = "" if pd.isna(uc_names_value) else str(uc_names_value).strip()
    # The source core-city field is not informative for this row; retain the supplied
    # human-readable UC name, including its script, instead of inventing a translation.
    if normalized_name(city) == "unnamed" and uc_names:
        return uc_names
    # A single UC name is already a manuscript label.  Semicolon-separated names are
    # a list, so use the compact core-city label instead of printing the full list.
    if uc_names and ";" not in uc_names:
        return uc_names
    # This fallback handles missing UC names and converts identifier syntax to prose.
    return manuscript_label(city)


def validate_existing_values(original: pd.DataFrame, neatnet: pd.DataFrame) -> None:
    """Validate the fixed values from the existing original and neatnet CSVs.

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
        Returns only after all stated existing-data checks pass.

    Notes
    -----
    These assertions are guards for the author's production run, not estimates or
    corrections.  They compare the existing v1/neatnet source values with the
    revision plan before any new summary is written.
    """
    # Counts and means catch an incorrect merge or eligibility filter before more
    # specific checks could pass on a subtly different sample.
    assert len(original) == 9_923
    assert original["uc_id"].nunique() == 9_923
    assert len(neatnet) == 9_921
    assert neatnet["uc_id"].nunique() == 9_921
    assert np.isclose(original["link_redundancy"].mean(), 0.281, atol=0.0005)
    assert np.isclose(original["length_redundancy"].mean(), 0.442, atol=0.0005)
    assert np.isclose(neatnet["link_redundancy"].mean(), 0.256, atol=0.0005)
    assert np.isclose(neatnet["length_redundancy"].mean(), 0.407, atol=0.0005)

    # Store each target with its expected source row, indicator, label, and precision.
    # Selecting extrema from the frame keeps these checks tied to the data rather than
    # rebuilding or reselecting any graph.
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

    # Check the two full-network examples used in the revision text by UC ID, not by
    # a city string that may have multiple source spellings.
    examples = {4443: (0.414606, 0.554694), 2315: (0.130825, 0.290192)}
    for uc_id, (theta, length) in examples.items():
        row = original.loc[original["uc_id"] == uc_id].iloc[0]
        assert np.isclose(row["link_redundancy"], theta, atol=1e-6)
        assert np.isclose(row["length_redundancy"], length, atol=1e-6)

    # Recompute the strict-tail contrasts from the prepared original frame so the
    # summary reports the same values that were validated here.
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

    # Connected-component values are checked in the exact Figure 1 order because the
    # later summary and caption use that order without reordering the cases.
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

    # Calculate primary correlations only to establish the comparison baseline; the
    # output CSV intentionally contains robust (neatnet and OSMnx) rows only.
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
    """Build the revision value rows from prepared representation frames.

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

    Notes
    -----
    Calculations remain at full floating-point precision in this machine-readable
    output.  The rows are grouped by references, sample summaries, decile contrasts,
    examples, extrema, and Figure 1 cases so later manuscript edits can use one
    stable lookup table without parsing the text summary.
    """
    rows: list[dict[str, object]] = []
    # Store both measures for each analytical reference.  The text summary adds the
    # unit-edge qualification for lambda because the CSV has no prose context.
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

    # Reuse the strict-tail calculation so the values CSV and caption-ready summary
    # cannot disagree about the decile definition.
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

    # These examples are selected by their fixed UC IDs from the existing data; no
    # graph is rebuilt and no city is reselected by a name match.
    examples = (("buenos_aires", 4443), ("bangkok", 2315))
    for label, uc_id in examples:
        row = original.loc[original["uc_id"] == uc_id].iloc[0]
        rows.extend(
            [
                {"item": f"{label}_theta", "value": row["link_redundancy"]},
                {"item": f"{label}_lambda", "value": row["length_redundancy"]},
            ]
        )

    # Select empirical extrema from the full original sample and retain the readable
    # city/country labels needed for later prose revisions.
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

    # Preserve the prescribed Figure 1 order.  The connected-component columns are
    # available only for the original and neatnet representations.
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
    # Duplicate item names would make a later lookup ambiguous, so reject them
    # before the writer creates the values file.
    assert len({str(row["item"]) for row in rows}) == len(rows)
    return rows


def write_values(path: Path, rows: list[dict[str, object]]) -> None:
    """Write unique revision values with the required two-column schema.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    rows : list[dict[str, object]]
        Rows containing unique ``item`` and ``value`` keys.

    Notes
    -----
    The schema assertion runs before opening the output.  This keeps malformed rows
    from producing a partially valid values file and preserves the exact two-column
    contract used by downstream manuscript work.

    Returns
    -------
    None
        The validated rows are written to ``path``.
    """
    # Validate every row before creating the file, so a malformed list cannot leave a
    # plausible but incomplete output behind.
    fieldnames = ["item", "value"]
    assert all(list(row) == fieldnames for row in rows)
    assert len({str(row["item"]) for row in rows}) == len(rows)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_correlations(path: Path, rows: list[dict[str, object]]) -> None:
    """Write exactly 36 robustness correlation rows.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    rows : list[dict[str, object]]
        Eighteen neatnet and eighteen OSMnx rows.

    Notes
    -----
    The primary/original correlations are used only as an in-memory comparison
    baseline.  Exactly 18 rows per robust representation are written, with flags
    that state whether sign or p<0.05 status changed from that baseline.

    Returns
    -------
    None
        The validated 36 rows are written to ``path``.
    """
    # Validate cardinality and column order before creating the output so a bad
    # correlation list cannot leave a plausible but incomplete CSV.
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
    """Build a short caption-ready text summary without result interpretation.

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

    Notes
    -----
    The summary uses stated display precision only; the values CSV retains full
    precision.  Sections contain only the quantities requested for the later
    revision, and labels are formatted for prose instead of copied from identifiers.
    Lattice lambda values are explicitly marked as assuming unit-length edges.
    """
    # Start with the one qualification that applies to the analytical lattice
    # comparison.  Empirical lambda values below use the observed edge lengths.
    lines = [
        "Street-network references",
        "Analytical lattice lambda values assume unit-length edges.",
    ]
    for label, value in STREET_NETWORK_REFERENCES:
        # Keep theta and lambda side by side because the analytical references are
        # limits for the same redundancy definitions.
        lines.append(f"{label}: theta={value:.6f}; lambda={value:.6f} (unit-length edges)")

    lines.append("")
    lines.append("Figure 1 cases")
    for uc_id in FIGURE_1_IDS:
        # Use the fixed order and manuscript labels so this block can be pasted into
        # a caption without exposing source identifiers such as ``zhenhai_district``.
        row = original.loc[original["uc_id"] == uc_id].iloc[0]
        lines.append(
            f"UC {uc_id}: {display_city(row)}, {manuscript_label(row['country'])}; "
            f"theta_CC={row['link_redundancy_cc']:.3f}; "
            f"lambda_CC={row['length_redundancy_cc']:.3f}"
        )

    lines.append("")
    lines.append("Full-network examples")
    # These examples use full-network theta and lambda, not the connected-component
    # values reported for Figure 1.
    for label, uc_id in (("Buenos Aires", 4443), ("Bangkok", 2315)):
        row = original.loc[original["uc_id"] == uc_id].iloc[0]
        lines.append(
            f"{label} (UC {uc_id}): theta={row['link_redundancy']:.6f}; "
            f"lambda={row['length_redundancy']:.6f}"
        )

    lines.append("")
    lines.append("Empirical extrema")
    # Select extrema from the already prepared original sample and format the source
    # labels for prose while retaining each UC ID for unambiguous lookup.
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
    # Recompute both requested contrasts from the same original frame used for
    # validation; strict percentile masks are applied inside the helper.
    for label, column in (("theta", "link_redundancy"), ("lambda", "length_redundancy")):
        contrasts = calculate_bc_gini_contrasts(original, column)
        lines.append(
            f"{label}: p10={contrasts['p10']:.6f}; p90={contrasts['p90']:.6f}; "
            f"low_median={contrasts['low_median']:.6f}; "
            f"high_median={contrasts['high_median']:.6f}"
        )

    lines.append("")
    lines.append("Representation summaries")
    # Display sample sizes, global means, and the two required robustness changes;
    # no result interpretation belongs in this machine-generated summary.
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
        # Report only changed states, as requested, rather than interpreting the
        # direction or practical meaning of any correlation.
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
    """Write the caption-ready summary text.

    Parameters
    ----------
    path : pathlib.Path
        Summary text path.
    summary : str
        Labeled summary content.

    Notes
    -----
    ``build_summary`` supplies the final newline.  This function only writes that
    prepared text so it cannot alter formatting or add an unplanned section.

    Returns
    -------
    None
        The prepared summary is written to ``path``.
    """
    path.write_text(summary)


def main() -> None:
    """Calculate and write the v2 values, correlations, and summary.

    Notes
    -----
    The source frames are loaded and derived in the same order as the three
    representations.  Existing-value validation runs before any output write, then
    the original frame supplies the comparison baseline and only robust rows are
    written to the 36-row correlation file.

    Returns
    -------
    None
        Returns after the three requested output artifacts are written.
    """
    # Loading through one function keeps eligibility filtering, ID parsing, merging,
    # and derived-variable definitions identical across representations.
    original = load_representation(ORIGINAL_PATH, INDICATORS_PATH, include_cc=True)
    neatnet = load_representation(NEATNET_PATH, INDICATORS_PATH, include_cc=True)
    osmnx = load_representation(OSMNX_PATH, INDICATORS_PATH, include_cc=False)
    # Stop before writing any v2 output if the existing source data do not match the
    # stated revision targets.
    validate_existing_values(original, neatnet)
    assert len(osmnx) == 9_923
    assert osmnx["uc_id"].nunique() == 9_923

    # The original correlations remain in memory only as the primary comparison;
    # compare both robustness representations against that same baseline.
    primary = calculate_correlations("original", original)
    correlations = compare_correlations(calculate_correlations("neatnet", neatnet), primary)
    correlations.extend(compare_correlations(calculate_correlations("osmnx", osmnx), primary))
    values = build_values(original, neatnet, osmnx)
    # These are the only three output artifacts permitted by the revision plan.
    write_values(VALUES_PATH, values)
    write_correlations(CORRELATIONS_PATH, correlations)
    write_summary(SUMMARY_PATH, build_summary(original, neatnet, osmnx, correlations))


if __name__ == "__main__":
    main()
