import pandas as pd
import geopandas as gpd


def create_powerplants_df_wui(df_wui, network_geom):
    # The information about the power plants is obtained from JRC Open Power Plants Database
    # (JRC-PPDB-OPEN) (https://zenodo.org/record/3574566). The `.zip` file has been already
    # opened and the `.csv` file has been saved in this folder. A pandas `DataFrame` is created.
    df_ppl = pd.read_csv("../../data/scenario/JRC_OPEN_UNITS.csv")

    # Onyl the commissioned (ca. 5000) power plants are considered for
    # further calculations.
    df_ppl = df_ppl[df_ppl["status_g"].isin(["COMMISSIONED"])]

    # Attention: "_g" stands for the single unit of a power plant, "_p" stands for whole power
    # plant. For simplicity, "_g" is used (the number of number plants is/seems to be higher).
    # Some typos are corrected, long names are abbreviated according to the PyPSA nomenclature
    # and the differentiation of uninteresting technologies is neglected according to the PyPSA
    # technologies. Therefore, 'Hydro Pumped Storage', 'Wind Offshore', 'Wind Onshore', 'Waste',
    # 'Solar', 'Fossil Peat', 'Marine', 'Other' are not considered since they are not relevant
    # for the puropose of this analyze or present in the PyPSA `Network`.
    df_ppl.replace("Fossil Hard coal", "coal", inplace=True)
    df_ppl.replace("Fossil Brown coal/Lignite", "lignite", inplace=True)
    df_ppl.replace("Fossil gas", "gas", inplace=True)  # OCGT is counted later in the data frame from PyPSA
    df_ppl.replace("Fossil Gas", "gas", inplace=True)  # OCGT is counted later in the data frame from PyPSA
    df_ppl.replace("Fossil Coal-derived gas", "gas",
                   inplace=True)  # coal gas is counted as gas since it is burned in gas turbine like natural gas
    df_ppl.replace("Fossil Oil", "oil", inplace=True)
    df_ppl.replace("Fossil Oil Shale", "oil", inplace=True)
    df_ppl.replace("Nuclear", "nuclear", inplace=True)
    df_ppl.replace("Biomass", "biomass", inplace=True)
    df_ppl.replace("Hydro Water Reservoir", "hydro", inplace=True)
    df_ppl.replace("Hydro Run-of-river and poundage", "hydro", inplace=True)
    df_ppl = df_ppl[df_ppl["type_g"].isin(["coal", "lignite", "gas", "hydro", "nuclear", "biomass", "oil"])]

    # A `GeoDataFrame` is created using the coordinates of the the power plants in `df_ppl` as
    # its geometry.
    geometry = gpd.points_from_xy(df_ppl.lon, df_ppl.lat)
    gdf_ppl = gpd.GeoDataFrame(df_ppl, geometry=geometry, crs=4326)
    network_geom = network_geom.to_crs(4326)

    # Every power plant is now associated to a region of the considered PyPSA `Network` according
    # to its coordinates. By doing so, a new column (called `name`) with the `buses` is added to
    # the same `GeoDataFrame`.
    gdf_ppl = gpd.sjoin(gdf_ppl, network_geom, how='inner', predicate='within')

    # In order to fit with the PyPSA nomenclature, the column `name` is turned into `bus` and
    # `type_g` into `carrier`.
    if 'index_right' in gdf_ppl.columns:
        gdf_ppl = gdf_ppl.rename(columns={'index_right': 'bus', 'type_g': 'carrier'}) # depend on gpd version
    elif 'name' in gdf_ppl.columns:
        gdf_ppl = gdf_ppl.rename(columns={'name': 'bus', 'type_g': 'carrier'}) # depend on gpd version

    # The source for the WUI factors do not distinguish between mechanical / natural Draft tower.
    # So, these cooling types are simplified:
    gdf_ppl.replace("Natural Draught Tower", "Draft Tower", inplace=True)
    gdf_ppl.replace("Mechanical Draught Tower", "Draft Tower", inplace=True)

    # It is evident that hydropower plants do not have a cooling system and they would be cut
    # off when `gdf_ppl` is aggregated according to the cooling system. Therefore, all the
    # hydropower plants get a fictive `cooling_type` called `hydro`.
    gdf_ppl.loc[gdf_ppl["carrier"] == "ror", "cooling_type"] = "hydro"
    gdf_ppl.loc[gdf_ppl["carrier"] == "hydro", "cooling_type"] = "hydro"

    # columns `water_consumption` and `water_withdrawal` in JRC Open Power Plants Database represent
    # water use intensity factors introduced in Lohrmann et al. Since these factors are not available
    # for hydropower in JRC database, Lohrmann's values are added manually
    hydro_ppl = gdf_ppl[gdf_ppl.carrier == "hydro"]
    gdf_ppl.loc[hydro_ppl.index, 'water_consumption'] = df_wui.loc['Hydropower', "Median"][0]
    gdf_ppl.loc[hydro_ppl.index, 'water_withdrawal'] = df_wui.loc['Hydropower', "Median"][0]

    # A new column with the WUI factors is added to the `GeoDataFrame`. WUI factors for the
    # "Once-Through" and "Draft Tower" are read directly from the source. For "No Cooling" and
    # "Air Cooling" the WUI factors are known only for CCGT, OCGT and oil. For all the other
    # technologies where these cooling systems are used (biomass, coal, lignite), the WUI factors
    # are taken from the European average values of each technology. This means that these power
    # plants have a WUI factor between the values of "Once-Through" and "Draft Tower".
    # Different columns for OCGT and CCGT are created in order to consider this difference
    # correctly in the next steps.

    '''
    # coal (Draft Tower, Once-Through)
    for i in gdf_ppl.query("carrier == 'coal' & cooling_type == 'Once-through'").index:  # Once-Through
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc[('Coal', 'Once-through', 'Generic'), "Median"]
    for i in gdf_ppl.query("carrier == 'coal' & cooling_type == 'Draft Tower'").index:  # Draft Tower
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc[('Coal', 'Tower', 'Generic'), "Median"]

        # lignite (Draft Tower, Once-Through, Air Cooling and No Cooling)
    for i in gdf_ppl.query("carrier == 'lignite' & cooling_type == 'Once-through'").index:  # Once-Through
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc[('Coal', 'Once-through', 'Generic'), "Median"]
    for i in gdf_ppl.query("carrier == 'lignite' & cooling_type == 'Draft Tower'").index:  # Draft Tower
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc[('Coal', 'Tower', 'Generic'), "Median"]

    # nuclear (Draft Tower, Once-Through)
    for i in gdf_ppl.query("carrier == 'nuclear' & cooling_type == 'Once-through'").index:  # Once-Through
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc[('Nuclear', 'Once-through', 'Generic'), "Median"]
    for i in gdf_ppl.query("carrier == 'nuclear' & cooling_type == 'Draft Tower'").index:  # Draft Tower
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc[('Nuclear', 'Tower', 'Generic'), "Median"]

    # CCGT (Draft Tower, Once-Through, Air Cooling and No Cooling)
    for i in gdf_ppl.query("carrier == 'CCGT' & cooling_type == 'Once-through'").index:  # Once-Through
        gdf_ppl.loc[i, "WUI cooling type CCGT"] = df_wui.loc[
            ('Natural Gas and Oil', 'Once-through', 'Combined cycle'), "Median"]
    for i in gdf_ppl.query("carrier == 'CCGT' & cooling_type == 'Draft Tower'").index:  # Draft Tower
        gdf_ppl.loc[i, "WUI cooling type CCGT"] = df_wui.loc[
            ('Natural Gas and Oil', 'Tower', 'Combined cycle'), "Median"]
    for i in gdf_ppl.query(
            "carrier == 'CCGT' & (cooling_type == 'Air Cooling' | cooling_type == 'No Cooling')").index:  # Air Cooling | No Cooling
        gdf_ppl.loc[i, "WUI cooling type CCGT"] = df_wui.loc[('Natural Gas and Oil', 'Dry', 'Combined cycle'), "Median"]

    # OCGT (Draft Tower, Once-Through, Air Cooling and No Cooling)
    for i in gdf_ppl.query("carrier == 'CCGT' & cooling_type == 'Once-through'").index:  # Once-Through
        gdf_ppl.loc[i, "WUI cooling type OCGT"] = df_wui.loc[
            ('Natural Gas and Oil', 'Once-through', 'Combined cycle'), "Min"]
    for i in gdf_ppl.query("carrier == 'CCGT' & cooling_type == 'Draft Tower'").index:  # Draft Tower
        gdf_ppl.loc[i, "WUI cooling type OCGT"] = df_wui.loc[('Natural Gas and Oil', 'Tower', 'Combined cycle'), "Min"]
    for i in gdf_ppl.query(
            "carrier == 'CCGT' & (cooling_type == 'Air Cooling' | cooling_type == 'No Cooling')").index:  # Air Cooling | No Cooling
        gdf_ppl.loc[i, "WUI cooling type OCGT"] = df_wui.loc[('Natural Gas and Oil', 'Dry', 'Combined cycle'), "Min"]

    # oil (Draft Tower, Once-Through, Air Cooling and No Cooling)
    for i in gdf_ppl.query("carrier == 'oil' & cooling_type == 'Once-through'").index:
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc[
            ('Natural Gas and Oil', 'Once-through', 'Combined cycle'), "Min"]  # Once-Through
    for i in gdf_ppl.query("carrier == 'oil' & cooling_type == 'Draft Tower'").index:
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc[
            ('Natural Gas and Oil', 'Tower', 'Combined cycle'), "Min"]  # Draft Tower
    for i in gdf_ppl.query("carrier == 'oil' & cooling_type == 'Draft Tower'").index:
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc[
            ('Natural Gas and Oil', 'Dry', 'Combined cycle'), "Min"]  # Air Cooling | No Cooling

    # biomass (Draft Tower, Once-Through, Air Cooling and No Cooling)
    for i in gdf_ppl.query("carrier == 'biomass' & cooling_type == 'Once-through'").index:  # Once-Through
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc[('Biomass', 'Once-through', 'Steam'), "Median"]
    for i in gdf_ppl.query("carrier == 'biomass' & cooling_type == 'Draft Tower'").index:  # Draft Tower
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc[('Biomass', 'Tower', 'Steam'), "Median"]

    # hydro
    for i in gdf_ppl.query("carrier == 'hydro'").index:
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc['Hydropower', "Median"].values

    # ror
    for i in gdf_ppl.query("carrier == 'ror'").index:
        gdf_ppl.loc[i, "WUI cooling type"] = df_wui.loc['Hydropower', "Median"].values

    # coal (NaN, Air Cooling and No Cooling)
    for i in gdf_ppl.query(
            "carrier == 'coal' & (cooling_type.isna() | cooling_type == 'Air Cooling' | cooling_type == 'No Cooling')").index:
        gdf_ppl.loc[i, "WUI cooling type"] = gdf_ppl.groupby("carrier").mean(numeric_only=True).loc[
            "coal", "WUI cooling type"]

    # lignite (NaN, Air Cooling and No Cooling)
    for i in gdf_ppl.query(
            "carrier == 'lignite' & (cooling_type.isna() | cooling_type == 'Air Cooling' | cooling_type == 'No Cooling')").index:
        gdf_ppl.loc[i, "WUI cooling type"] = gdf_ppl.groupby("carrier").mean(numeric_only=True).loc[
            "lignite", "WUI cooling type"]

    # nuclear (NaN, Air Cooling and No Cooling)
    for i in gdf_ppl.query(
            "carrier == 'nuclear' & (cooling_type.isna() | cooling_type == 'Air Cooling' | cooling_type == 'No Cooling')").index:
        gdf_ppl.loc[i, "WUI cooling type"] = gdf_ppl.groupby("carrier").mean(numeric_only=True).loc[
            "nuclear", "WUI cooling type"]

    # biomass (NaN, Air Cooling and No Cooling)
    for i in gdf_ppl.query(
            "carrier == 'biomass' & (cooling_type.isna() | cooling_type == 'Air Cooling' | cooling_type == 'No Cooling')").index:
        gdf_ppl.loc[i, "WUI cooling type"] = gdf_ppl.groupby("carrier").mean(numeric_only=True).loc[
            "biomass", "WUI cooling type"]

    # CCGT (NaN)
    for i in gdf_ppl.query("carrier == 'CCGT' & cooling_type.isna()").index:
        gdf_ppl.loc[i, "WUI cooling type CCGT"] = gdf_ppl.groupby("carrier").mean(numeric_only=True).loc[
            "CCGT", "WUI cooling type CCGT"]

    # OCGT (NaN)
    for i in gdf_ppl.query("carrier == 'CCGT' & cooling_type.isna()").index:
        gdf_ppl.loc[i, "WUI cooling type OCGT"] = gdf_ppl.groupby("carrier").mean(numeric_only=True).loc[
            "CCGT", "WUI cooling type OCGT"]

    # oil (NaN)
    for i in gdf_ppl.query("carrier == 'oil' & cooling_type.isna()").index:
        gdf_ppl.loc[i, "WUI cooling type CCGT"] = gdf_ppl.groupby("carrier").mean(numeric_only=True).loc[
            "oil", "WUI cooling type"]
    '''

    # The share of each cooling system is calculated for each region and each technology according
    # to the nominal power capacities (`capacity_g`) which may differ from the optimal power plant
    # capacities from the PyPSA model (**strong assumption!**):
    gdf_ppl.loc[gdf_ppl[gdf_ppl["cooling_type"].isnull()].index, 'cooling_type'] = "not_available"
    df_ppl_grouped = gdf_ppl.groupby(["bus", "carrier", "cooling_type"]).agg({"capacity_g": "sum",
                                                                              "water_consumption": "first",
                                                                              "water_withdrawal": "first"})

    for region in gdf_ppl["bus"].unique():
        for tech in gdf_ppl["carrier"].unique():
            for cool in gdf_ppl["cooling_type"].unique():
                group_key = (region, tech, cool)
                if group_key in df_ppl_grouped.index:

                    df_ppl_grouped.loc[group_key, "share"] = df_ppl_grouped.loc[group_key, "capacity_g"] / \
                                                             gdf_ppl.groupby(["bus", "carrier"]).sum(
                                                                 numeric_only=True).loc[(region, tech), "capacity_g"]

                    df_ppl_grouped.loc[group_key, "wui_c"] = df_ppl_grouped.loc[group_key, "share"] * \
                                                                    df_ppl_grouped.loc[
                                                                        group_key, "water_consumption"]

                    df_ppl_grouped.loc[group_key, "wui_w"] = df_ppl_grouped.loc[group_key, "share"] * \
                                                             df_ppl_grouped.loc[
                                                                 group_key, "water_withdrawal"]

                        # Since the cooling technologies are not relevant, the data is now aggregated into a new
    # `DataFrame`.
    df_ppl_WUI = df_ppl_grouped.reset_index().groupby(["bus", "carrier"]).agg({"wui_c": "sum", "wui_w": "sum"})

    # This `DataFrame` has the WUI factors of te different technologies in each region. This
    # `DataFrame` is now saved in a `.csv` file.
    df_ppl_WUI.to_csv("inputs/wui_per_region_technology.csv")