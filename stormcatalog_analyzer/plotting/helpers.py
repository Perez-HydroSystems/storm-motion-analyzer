"""Shared plotting helpers.

`_plot_geographic_outline` replaces the previous inline
``gdf.plot(facecolor='none', ...)`` calls. It was referenced in
``diagnostic_plots.py`` but never defined, which raised ``NameError`` in every
diagnostic plot. Defined here once and imported by the plotting modules.
"""


def _plot_geographic_outline(
    gdf,
    ax,
    label=None,
    edgecolor="black",
    facecolor="none",
    linewidth=1.0,
    linestyle="-",
    zorder=1,
    **kwargs,
):
    """Draw a GeoDataFrame boundary outline on a (possibly Cartopy) axis.

    The geometries are expected in geographic coordinates (lat/lon), matching the
    PlateCarree axes used by the diagnostic plots. No-op when ``gdf`` is None or
    empty. Returns ``ax``.
    """
    if gdf is None:
        return ax
    try:
        if hasattr(gdf, "empty") and gdf.empty:
            return ax
    except Exception:
        pass

    gdf.plot(
        ax=ax,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        zorder=zorder,
        **kwargs,
    )
    return ax


def _annotate_conus_grid(ax, grid, font_size=0):
    """Annotate the CONUS 5x5 reference grid with 1..12 (columns) and A..E (rows).

    Paper-specific: expects a GeoDataFrame with ``row_index``/``col_index``/
    ``left``/``right``/``top``/``bottom`` columns and the original ``grid.loc[26]``
    layout. Callers should only invoke this when such a grid is provided.
    """
    import cartopy.crs as ccrs

    for i in range(1, 13):
        cell = grid[(grid["row_index"] == 2.0) & (grid["col_index"] == float(i - 1))]
        x = (cell.right.values[0] + cell.left.values[0]) / 2
        y = cell.top.values[0] + 0.5
        ax.text(x, y, str(i), fontsize=10 + font_size, fontweight="bold",
                ha="center", va="center", transform=ccrs.PlateCarree())
    for i, letter in enumerate(["A", "B", "C", "D", "E"]):
        if i < 3:
            cell = grid[(grid["row_index"] == float(i + 2)) & (grid["col_index"] == 0.0)]
        elif i == 3:
            cell = grid[(grid["row_index"] == float(i + 2)) & (grid["col_index"] == 1.0)]
        else:
            cell = grid[(grid["row_index"] == float(i + 2)) & (grid["col_index"] == 5.0)]
        x = grid.loc[26].left - 0.9
        y = (cell.top.values[0] + cell.bottom.values[0]) / 2
        ax.text(x, y, letter, fontsize=10 + font_size, fontweight="bold",
                ha="center", va="center", transform=ccrs.PlateCarree())
