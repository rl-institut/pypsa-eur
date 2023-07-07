import pandas as pd
import geopandas as gpd

def spatial_join_existing_powerplants(ppl_path, geo_path):
    # credits: sergiotom for RLI

    # The information about the power plants is obtained from JRC Open Power Plants Database
    # (JRC-PPDB-OPEN) (https://zenodo.org/record/3574566). The `.zip` file has been already
    # opened and the `.csv` file has been saved in this folder. A pandas `DataFrame` is created.
    df_ppl = pd.read_csv(ppl_path)

    # Onyl the commissioned (ca. 5000) and the reserve (ca. 20) power plants are considered for
    # further calculations.
    df_ppl = df_ppl[df_ppl["status_g"].isin(["COMMISSIONED", "RESERVE"])]

    # Attention: "_g" stands for the single unit of a power plant, "_p" stands for whole power
    # plant. For simplicity, "_g" is used (the number of number plants is/seems to be higher).
    # Some typos are corrected, long names are abbreviated according to the PyPSA nomenclature
    # and the differentiation of uninteresting technologies is neglected according to the PyPSA
    # technologies. Therefore, 'Hydro Pumped Storage', 'Wind Offshore', 'Wind Onshore', 'Waste',
    # 'Solar', 'Fossil Peat', 'Marine', 'Other' are not considered since they are not relevant
    # for the puropose of this analyze or present in the PyPSA `Network`.
    df_ppl.replace("Fossil Hard coal", "coal", inplace=True)
    df_ppl.replace("Fossil Brown coal/Lignite", "lignite", inplace=True)
    df_ppl.replace("Fossil gas", "CCGT", inplace=True)  # OCGT is counted later in the data frame from PyPSA
    df_ppl.replace("Fossil Gas", "CCGT", inplace=True)  # OCGT is counted later in the data frame from PyPSA
    df_ppl.replace("Fossil Coal-derived gas", "CCGT",
                   inplace=True)  # coal gas is counted as gas since it is burned in gas turbine like natural gas
    df_ppl.replace("Fossil Oil", "oil", inplace=True)
    df_ppl.replace("Fossil Oil Shale", "oil", inplace=True)
    df_ppl.replace("Nuclear", "nuclear", inplace=True)
    df_ppl.replace("Biomass", "biomass", inplace=True)
    df_ppl.replace("Hydro Water Reservoir", "ror", inplace=True) # ror or hydro?
    df_ppl.replace("Hydro Run-of-river and poundage", "ror", inplace=True)
    df_ppl = df_ppl[df_ppl["type_g"].isin(["coal", "lignite", "CCGT", "hydro", "ror", "nuclear", "biomass", "oil"])]

    # A `GeoDataFrame` is created using the coordinates of the the power plants in `df_ppl` as
    # its geometry.
    geometry = gpd.points_from_xy(df_ppl.lon, df_ppl.lat)
    gdf_ppl = gpd.GeoDataFrame(df_ppl, geometry=geometry, crs=4326)

    # now import the onshore region geometries
    network_geom = gpd.read_file(geo_path)

    # Every power plant is now associated to a region of the considered PyPSA `Network` according
    # to its coordinates. By doing so, a new column (called `name`) with the `buses` is added to
    # the same `GeoDataFrame`.
    gdf_ppl = gpd.sjoin(gdf_ppl, network_geom, how='inner', predicate='within')

    # In order to fit with the PyPSA nomenclature, the column `name` is turned into `bus` and
    # `type_g` into `carrier`.
    gdf_ppl = gdf_ppl.rename(columns={'name': 'bus', 'type_g': 'carrier', 'capacity_g': 'p_nom'})

    # The nominal power capacities (`capacity_g`) are calculated for each region and each technology according
    # which may differ from the optimal power plant capacities from the PyPSA model
    gdf_ppl = gdf_ppl.groupby(["bus", "carrier"]).agg({"p_nom": "sum"})
    gdf_ppl = gdf_ppl.reset_index(level='carrier')
    gdf_ppl['cty'] = gdf_ppl.index.str[:2]

    return gdf_ppl