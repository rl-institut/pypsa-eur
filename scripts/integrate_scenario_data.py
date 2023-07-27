import pandas as pd
import geopandas as gpd
import numpy as np

sce_name = 'Global Ambition'

def import_sce_data(file_path, sheet_name):

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    df = df.query(
        f'''
        `Scenario` == '{sce_name}' & \
        `Output_type` == 'Energy_demand' & \
        `Country` != ['CY','MT'] & \
        `Sector` == ['Residential','Tertiary','Agriculture','Transport','Industry']
        ''').drop(['Scenario', 'Output_type'], axis=1).replace({'Country': {'UK': 'GB'}}) \
        .apply(lambda x: x.str.lower() if x.name in ['Sector', 'Subsector', 'Energy_carrier'] else x) \
        .set_index('Country').sort_index()

    return df


def get_sce_data_by_sector(df, sector):

    if sector == 'buildings':

        df_buil = df.query(
            '''
            `Sector` == ['residential','tertiary'] & \
            `Energy_carrier` != 'all'
            ''').drop(['Sector_viz_platform'], axis=1) \
            .replace({'Sector': {'tertiary': 'services'}})

        return df_buil

    elif sector == 'transport':

        df_tra = df.query(
            '''
            `Sector` == 'transport' & \
            `Energy_carrier` != 'all'
            ''').drop(['Sector_viz_platform'], axis=1)

        return df_tra

    elif sector == 'industry':

        df_ind = df.query(
            '''
                `Sector` == 'industry'
                ''').drop(['Sector_viz_platform'], axis=1)

        return df_ind

    elif sector == 'agriculture':

        df_agr = df.query(
            '''
            `Sector` == 'agriculture'
            ''').drop(['Sector_viz_platform'], axis=1)

        return df_agr

    elif sector == 'all':

        return df


def get_heat_demand_by_use(df_buil, subsector, use, year):

    df = df_buil.query(
        f'''
        `Sector` == '{subsector}' & \
        `Subsector` == '{use}_heating' & \
        `Energy_carrier` == ['hydrogen','methane','liquids','solids','biomass','others'] & \
        `Year` == {year}
        ''').drop(['Sector', 'Subsector', 'Energy_carrier', 'Year'], axis=1) \
        .rename(columns={"Value": f"{subsector} {use}"}) \
        .groupby('Country').sum()

    return df

def get_electricity_demand_by_use(df_buil, subsector, year):

    df = df_buil.query(
        f'''
        `Sector` == '{subsector}' & \
        `Energy_carrier` == ['electricity'] & \
        `Year` == {year}
        ''').drop(['Sector','Subsector','Energy_carrier','Year'], axis=1) \
        .rename(columns={"Value": f"{subsector}"}) \
        .groupby('Country').sum()

    return df

# TODO: discuss how to deal with electricity rail
def get_transport_demand_by_subsector(df_tra, subsector, carrier, year):

    if subsector == 'land_transport':
        subsector = ['2_wheelers', 'passenger_cars', 'busses', 'rail', 'heavy_trucks', 'light_trucks']
    else:
        subsector = [f'{subsector}']

    if carrier == 'all':
        carrier = ['liquids', 'methane', 'electricity', 'hydrogen']
    else:
        carrier = [f'{carrier}']

    df = df_tra.query(
        f'''
        `Subsector` == {subsector} & \
        `Energy_carrier` == {carrier} & \
        `Year` == {year}
        ''').drop(['Sector', 'Subsector', 'Energy_carrier', 'Year'], axis=1) \
        .groupby('Country').sum()["Value"]

    return df


# TODO: discuss how to deal with LT heat
def get_industrial_demand_by_carrier(df_ind, year):

    df = df_ind.query(
        f'''
        `Energy_carrier` == ['hydrogen', 'electricity', 'biomass', 'liquids', 'methane'] & \
        `Subsector` != 'low_enthalpy_heat' & \
        `Year` == {year}
        ''').drop(['Sector', 'Subsector', 'Year'], axis=1)

    df = df.reset_index().pivot_table(index='Country', columns='Energy_carrier', values='Value', aggfunc='sum')

    df_leh = df_ind.query(
            f'''
            `Subsector` == 'low_enthalpy_heat' & \
            `Energy_carrier` == 'all' & \
            `Year` == {year}
            ''').drop(['Sector', 'Subsector', 'Energy_carrier', 'Year'], axis=1) \
        .rename(columns={"Value": "low_enthalpy_heat"})

    df_cel = df_ind.query(
            f'''
            `Energy_carrier` == 'electricity' & \
            `Subsector` != 'low_enthalpy_heat' & \
            `Year` == 2015
            ''').drop(['Sector', 'Subsector', 'Year', 'Energy_carrier'], axis=1) \
            .groupby('Country').sum() \
            .rename(columns={"Value": "current electricity"})

    df = pd.concat([df, df_leh, df_cel], axis=1)

    return df


def get_agricultural_demand_by_carrier(df_agr, carrier, year):

    df = df_agr.query(
        f'''
        `Energy_carrier` == '{carrier}' & \
        `Year` == {year}
        ''').drop(['Sector', 'Subsector', 'Energy_carrier', 'Year'], axis=1)['Value']

    return df

def distribute_sce_demand_by_pes_layout(sce_demand_nat, pes_demand_reg, pop_layout):

    pes_demand_reg['cty'] = pes_demand_reg.index.str[:2]
    pes_demand_reg['fraction'] = pes_demand_reg.groupby("cty").transform(lambda x: x / x.sum())

    # in case pes demand is zero, but sce demand not - weight based on population
    if pes_demand_reg.fraction.isnull().any():
        pes_demand_reg['fraction'] = pes_demand_reg.fraction.fillna(pop_layout.fraction)

    sce_demand_reg = pes_demand_reg.cty.map(sce_demand_nat).mul(pes_demand_reg.fraction)

    return sce_demand_reg

def scale_district_heating_dem(n, dh_share, year):

    # Code from branch https://github.com/PyPSA/pypsa-eur-sec/tree/PAC

    # get scenario share and pes demands
    share = pd.read_csv(dh_share, index_col=0)
    heat_total = n.loads_t.p_set.loc[:, n.loads.carrier.str.contains("heat")].sum().sum()
    heat_decentral = n.loads_t.p_set.loc[:, (n.loads.carrier.str.contains("heat")) &
                                             ~((n.loads.carrier.str.contains("urban central heat")))].sum().sum()
    heat_dh_total = share.loc[float(year), sce_name] * heat_total

    # calculate scaling factors
    scale_factor_not_dh = (heat_total - heat_dh_total) / heat_decentral
    scale_factor_dh = heat_dh_total / n.loads_t.p_set.loc[:, n.loads.carrier=="urban central heat"].sum().sum()

    # scale demands
    n.loads_t.p_set.loc[:, (n.loads.carrier.str.contains("heat")) &
                            ~((n.loads.carrier.str.contains("urban central heat")))] *= scale_factor_not_dh
    n.loads_t.p_set.loc[:, n.loads.carrier=="urban central heat"] *= scale_factor_dh

    print("district heating share is scaled up by; ", scale_factor_dh)


def build_sce_caps_and_prods(input_path_cap, input_path_h2, output_path):

    carrier_mapping = {'wind onshore': 'onwind',
                       'onshore wind_stand alone': 'onwind',
                       'wind offshore': 'offwind',
                       'offshore wind_stand alone': 'offwind',
                       'solar_stand alone': 'solar'}

    eu28_str = 'AT|BE|BG|CZ|DE|DK|EE|ES|FI|FR|GR|HR|HU|IE|IT|LT|LU|LV|NL|PL|PT|RO|SE|SI|SK|UK'  # 'CY','MT'

    # PPL Capacities

    sheet_name = "Capacity & Dispatch"
    df = pd.read_excel(input_path_cap, sheet_name)

    df = df.query(
        f'''
        `Scenario` == '{sce_name}' & \
        `Climate Year` == 'CY 2009'
        ''').apply(lambda x: x.str.lower() if x.name in ['Fuel'] else x) \
        .rename(columns={"Fuel": "carrier"}) \
        .replace({'carrier': carrier_mapping})

    # add country column
    df = df[df.Node.str.contains(eu28_str)]
    df["country"] = df.Node.str[:2]
    df["country"] = df["country"].replace({'UK': "GB"})

    df_cap = df.query(
        '''
        `Parameter` == 'Capacity (MW)' & \
        `carrier` != ['other res', 'other non res']
        ''')

    df_agg_ppl_caps = pd.pivot_table(df_cap, values='Value', index=['country', 'carrier'],
                                     columns=['Year'], aggfunc=np.sum)

    df_gen = df.query(
        '''
        `Parameter` == 'Dispatch (GWh)' & \
        `carrier` == ['hydro', 'gas', 'oil', 'coal & lignite', 'nuclear']
        ''')

    # Electrolyser Generation

    sheet_name = "H2 Generation"
    df_gen_h2 = pd.read_excel(input_path_h2, sheet_name)

    df_gen_h2 = df_gen_h2.query(
        f'''
        `Scenario` == '{sce_name}' & \
        `Parameter` == 'Generation (GWh)' & \
        `Climate Year` == '2009-01-01'
        ''').apply(lambda x: x.str.lower() if x.name in ['Fuel'] else x) \
        .rename(columns={"Fuel": "carrier", "Node 1": "Node"})

    # add country column
    df_gen_h2 = df_gen_h2[df_gen_h2.Node.str.contains(eu28_str)]
    df_gen_h2["country"] = df_gen_h2.Node.str[:2]
    df_gen_h2["country"] = df_gen_h2["country"].replace({'UK': "GB"})
    df_gen_h2["carrier"] = "electrolyser"

    df_gen = pd.concat([df_gen, df_gen_h2])

    df_agg_ppl_gens = pd.pivot_table(df_gen, values='Value', index=['country', 'carrier'],
                                     columns=['Year'], aggfunc=np.sum)

    # Electrolyser Capacity

    df_ele_res = df[df["Node/Line"].str.contains("H2R4")]

    sheet_name = "H2 Capacity"
    df = pd.read_excel(input_path_h2, sheet_name)

    df = df.query(
        f'''
        `Scenario` == '{sce_name}' & \
        `Parameter` == 'Capacity (MW)' & \
        `Fuel` == 'Electrolyser' & \
        `Climate Year` == '2009-01-01'
        ''').apply(lambda x: x.str.lower() if x.name in ['Fuel'] else x) \
        .rename(columns={"Fuel": "carrier", "Node 1": "Node"})

    # add country column
    df = df[df.Node.str.contains(eu28_str)]
    df["country"] = df.Node.str[:2]
    df["country"] = df["country"].replace({'UK': "GB"})
    df_ele_mar = df

    # concat market/hybrid RES + dedicated PES electrolyser caps
    df_sup_ele = pd.concat([df_ele_res, df_ele_mar])
    df_sup_ele["carrier"] = "electrolyser"

    df_agg_ele_caps = pd.pivot_table(df_sup_ele, values='Value', index=['country', 'carrier'],
                                     columns=['Year'], aggfunc=np.sum)

    # merge ppl caps and electrolyser caps
    df_agg_caps = pd.concat([df_agg_ppl_caps, df_agg_ele_caps]).sort_index()


    # add NaN rows for carrier not used in country
    carr = list(df_agg_caps.index.get_level_values("carrier").unique())
    ctys = list(df_agg_caps.index.get_level_values("country").unique())
    multi_idx = pd.MultiIndex.from_product([ctys, carr], names=['country', 'carrier'])
    df_agg_caps = df_agg_caps.reindex(multi_idx).fillna(0)

    # carr = list(df_agg_ppl_gens.index.get_level_values("carrier").unique())
    # ctys = list(df_agg_ppl_gens.index.get_level_values("country").unique())
    # multi_idx = pd.MultiIndex.from_product([ctys, carr], names=['country', 'carrier'])
    # df_agg_ppl_gens = df_agg_ppl_gens.reindex(multi_idx)

    # export
    df_agg_caps.to_csv(output_path + "/agg_p_nom_sce.csv", index=True)
    df_agg_ppl_gens.to_csv(output_path + "/agg_e_gen_sce.csv", index=True)


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
