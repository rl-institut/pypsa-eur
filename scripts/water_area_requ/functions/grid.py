import geopandas as gpd
from shapely.geometry import LineString
from shapely.validation import make_valid
from shapely.wkt import loads


def wkt_loads(x):
    try:
        return loads(x)
    except Exception:
        return None


def build_lines(line_gdf, network):
    line_gdf["geometry"] = None

    for idx in line_gdf.index:
        bus0, bus1 = line_gdf.bus0.loc[idx], line_gdf.bus1.loc[idx]
        x0, x1 = network.buses.loc[bus0, "x"], network.buses.loc[bus1, "x"]
        y0, y1 = network.buses.loc[bus0, "y"], network.buses.loc[bus1, "y"]
        line_gdf.loc[idx, "geometry"] = LineString([(x0, y0), (x1, y1)])

    return line_gdf


def spatial_join_and_length_calculation(df, geo):
    # A `GeoDataFrame` is created out of the `DataFrame` with `geometry`.
    gdf = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry='geometry').to_crs('EPSG:3035')

    # A `GeoDataFrame` with the considered regions (`POLYGON`) is created.
    regions_on_gdf = geo.to_crs('EPSG:3035')
    regions_on_gdf['geometry'] = regions_on_gdf.apply(lambda row: make_valid(row.geometry)
    if not row.geometry.is_valid else row.geometry, axis=1)

    # In the following step, the `overlay()` function from the geopandas library is used to perform
    # a spatial join between the `lines_gdf` and `regions_on_gdf` dataframes. This function identifies
    # the spatial intersection between the two dataframes and creates a new `GeoDataFrame` with the combined attributes of both dataframes.
    # The `how` parameter specifies the type of spatial join to perform (in this case, an intersection).
    # The `keep_geom_type` parameter is set to `False` to ensure that the output `GeoDataFrame`
    # only contains the geometries that are relevant to the intersection (i.e., it drops any empty geometries that do not intersect).
    lines_gdf = gpd.overlay(gdf, regions_on_gdf, how='intersection', keep_geom_type=False)

    # After performing the spatial join, the `duplicated()` method is used to identify which line segments have been cut by
    # the intersection with the regions. The `length` attribute of these cut segments is then updated to reflect the actual
    # length of the cut segment.
    lines_gdf['cut'] = lines_gdf.duplicated(
        subset=[val for val in lines_gdf.columns if not val in ['name', 'geometry']], keep=False)
    lines_gdf.loc[lines_gdf['cut'] == True, 'length'] = lines_gdf['geometry'].length / 1000

    lines_gdf['country'] = lines_gdf.name.str[:2]
    regions_on_gdf['country'] = regions_on_gdf.name.str[:2]

    # In the following step, the `dissolve()` method from the geopandas library is used to group `lines_c` by
    # the region (`name`) and calculate the total length of lines per country.
    # The `aggfunc` parameter specifies the aggregation functions to apply to each group. In this case, the `length` column is summed,
    # and the `name` column is concatenated into a single string using the `lambda` function. The resulting
    # `GeoDataFrame` contains a single row for each country with the total length of lines and the concatenated names of the line segments.

    lines_per_region = lines_gdf.dissolve(by='name', aggfunc={"length": 'sum', "power_capacity": 'sum',
                                                              'name': lambda x: ', '.join(x.unique())})
    regions = regions_on_gdf.dissolve(by='name', aggfunc={'name': lambda x: ', '.join(x.unique())})

    return lines_per_region, regions, lines_gdf