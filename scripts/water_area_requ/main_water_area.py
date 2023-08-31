import os
import json
import yaml
import pypsa
import pandas as pd
import geopandas as gpd
pd.options.mode.chained_assignment = None
from functions.network import create_powerplants_df_wui
from functions.grid import wkt_loads, build_lines, spatial_join_and_length_calculation
from functions.landuse import calculate_landuse_areas

############################### CHANGE DATA HERE #####################################

with open("../../config/config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

# scenario specs
sce_name = config["run"]["name"]
clusters = config["scenario"]["clusters"][0]
target_years = config["scenario"]["planning_horizons"]
ll = config["scenario"]["ll"][0]
opts = config["scenario"]["sector_opts"][0]

# directory
unopt_network_name = f"elec_s_{clusters}"
opt_network_name = f"{unopt_network_name}_l{ll}__{opts}"

######################################################################################

# pypsa-eur files
n_base = pypsa.Network(f"../../resources/{sce_name}/networks/base.nc")
n_unopt = pypsa.Network(f"../../resources/{sce_name}/networks/{unopt_network_name}.nc")
geo_on = gpd.read_file(f"../../resources/{sce_name}/regions_onshore_{unopt_network_name}.geojson", crs='EPSG:4326')
geo_off = gpd.read_file(f"../../resources/{sce_name}/regions_offshore_{unopt_network_name}.geojson", crs='EPSG:4326')

# area specs
res_area_mapping = {'onwind': 10.42, 'solar': 152.46 * 0.33, 'solar rooftop': 152.46, 'offwind': 10.42}
spec_electrolyser = 170  # m²/MW # area - hydrogen electrolyzer (from IRENA 2020)
spec_h2storages = 10  # m²/MWh (assumed) # area - hydrogen storages ( ASSUMED! )
cor_width = 0.07  # km # typical corridor width of transmission grid

# water specs
H2O_elec_kg = 12.015  # kgH2O/kgH2 (from Cetinkaya)
smr = True
H20_SMR_kg = 18.8
H2O_elec_MWh = H2O_elec_kg / 0.03332222
H2O_SMR_MWh = H2O_elec_kg / 0.0333222
df_wui = pd.read_csv("inputs/wui.csv", index_col=[0, 1, 2]) # water usage intensity factors [m³/MWh]

# conversion factors
olympic_pool_volume = (50 * 25 * 2) # m3
olympic_stadium_area = (105 * 68) # m2

#### AREA CALC ####

# Transmission Grid
# prepare unoptimized / base network
lines_base = pd.concat([n_base.lines, n_base.links])
lines_base['geometry'] = lines_base['geometry'].apply(wkt_loads)
lines_base.dropna(subset=['geometry'], inplace=True)
lines_base["power_capacity"] = lines_base["s_nom"]
lines_per_region_base, regions_base, lines_gdf_base = spatial_join_and_length_calculation(lines_base, geo_on)

lines_df_unopt = pd.concat([n_unopt.lines, n_unopt.links])
unopt = lines_df_unopt[["bus0", "bus1", "s_nom", "p_nom"]].fillna(0)
unopt["power_capacity"] = unopt["p_nom"] + unopt["s_nom"]
unopt.drop(columns={"s_nom", "p_nom"}, inplace=True)
unopt = build_lines(unopt, n_unopt)
lines_per_region_unopt, regions_unopt, lines_gdf_unopt = spatial_join_and_length_calculation(unopt, geo_on)

# land use
# calculate area of urban / industrial sites as well as natural protection areas using atlite
if not os.path.isfile("inputs/landuse_on_" + unopt_network_name + ".csv"):
    calculate_landuse_areas(unopt_network_name, geo_on, "onshore")

area_tys = []

for ty in target_years:

    # import optimized network per target year
    n = pypsa.Network(f"../../results/{sce_name}/postnetworks/{opt_network_name}_{ty}.nc")

    # all RES (solar / onshore / offshore)
    # filter - rename - group - calc - reindex
    gens = n.generators[
        n.generators.carrier.str.contains("solar|onwind|offwind") & ~n.generators.carrier.str.contains("rural|urban")]
    gens.bus = gens.bus.str.split(expand=True)[[0, 1]].agg(' '.join, axis=1)
    gens.loc[gens.carrier.str.contains("-ac|-dc"), "carrier"] = "offwind"
    gens = gens[['bus', 'carrier', 'p_nom_opt']].groupby(['bus', 'carrier'], as_index=False).sum()
    gens["area_km2"] = gens.p_nom_opt / gens.carrier.map(res_area_mapping)
    gens = gens.set_index(['bus', 'carrier']).drop(['p_nom_opt'], axis=1)

    # H2 Electrolyser
    # filter - rename - group - calc - reindex
    links = n.links[n.links.carrier == "H2 Electrolysis"]
    links = links[['bus0', 'carrier', 'p_nom_opt']].groupby(['bus0', 'carrier'], as_index=False).sum()
    links["area_km2"] = links.p_nom_opt * spec_electrolyser / 1e6
    links = links.set_index(['bus0', 'carrier']).drop(['p_nom_opt'], axis=1)

    # H2 Storages
    # filter - rename - group - calc - reindex
    stores = n.stores[n.stores.carrier == "H2"]
    stores.bus = stores.bus.str.split(expand=True)[[0, 1]].agg(' '.join, axis=1)
    stores.carrier = "H2 Store"
    stores = stores[['bus', 'carrier', 'e_nom_opt']].groupby(['bus', 'carrier'], as_index=False).sum()
    stores["area_km2"] = stores.e_nom_opt * spec_h2storages / 1e6
    stores = stores.set_index(['bus', 'carrier']).drop(['e_nom_opt'], axis=1)

    # Transmission Grid
    # prepare
    n_opt = n
    lines_df_opt = pd.concat([n_opt.lines, n_opt.links]).iloc[:len(lines_df_unopt)]
    opt = lines_df_opt[["bus0", "bus1", "s_nom_opt", "p_nom_opt"]].fillna(0)
    opt["power_capacity"] = opt["p_nom_opt"] + opt["s_nom_opt"]
    opt.drop(columns={"s_nom_opt", "p_nom_opt"}, inplace=True)
    opt = build_lines(opt, n_opt)
    lines_per_region_opt, regions_opt, lines_gdf_opt = spatial_join_and_length_calculation(opt, geo_on)
    # evaluate using unoptimized / base network
    lines = pd.DataFrame(index=lines_per_region_opt.index)  # comparison dataframe
    lines["len_km"] = lines_per_region_base["length"]  # km
    lines["s_nom_exp_mw"] = lines_per_region_opt["power_capacity"] - lines_per_region_unopt["power_capacity"]
    lines["s_nom_exp_rel"] = (lines_per_region_opt["power_capacity"] - lines_per_region_unopt["power_capacity"]) / \
                             lines_per_region_unopt["power_capacity"] * 100  # %
    lines["len_exp_km"] = lines["s_nom_exp_rel"] * lines["len_km"] / 100  # km
    lines["area_km2"] = lines["len_exp_km"] * cor_width  # km²
    # reindex
    lines["type"] = "grid"
    lines = lines.set_index([lines.index, 'type'])[["area_km2"]]

    # import land-use previously claculated
    lands = pd.read_csv("inputs/landuse_on_" + unopt_network_name + ".csv", index_col=["bus", "type"])

    # join all relevant frames
    area_ty = pd.concat([gens, links, stores, lines, lands], axis=0)
    area_ty["sce_name"] = sce_name
    area_ty["target_year"] = ty
    area_ty = area_ty.sort_index().set_index(["sce_name", "target_year"], append=True).reorder_levels([2, 3, 0, 1])
    area_ty.index.names = ["sce_name", "target_year", "bus", "type"]

    area_tys.append(area_ty)

# join all technologies for area calc
area_joined = pd.concat(area_tys)
area_joined["oly_field"] = (area_joined["area_km2"] * 1e6 / olympic_stadium_area).round(0)

# relative area requirement calc
# one exception: offshore wind is related to offshore area, else onshore area
geo_on = geo_on.to_crs('EPSG:3035').set_index("name")
geo_on["area"] = geo_on.geometry.area
geo_off = geo_off.to_crs('EPSG:3035').set_index("name")
geo_off["area"] = geo_off.geometry.area

area_on = area_joined[area_joined.index.get_level_values("type") != "offwind"]
area_off = area_joined[area_joined.index.get_level_values("type") == "offwind"]
area_joined.loc[area_on.index, 'rel'] = area_on.area_km2 / area_on.index.get_level_values("bus").map(geo_on.area) * 100 * 1e6
area_joined.loc[area_off.index, 'rel'] = area_off.area_km2 / area_off.index.get_level_values("bus").map(geo_off.area) * 100 * 1e6

# handle offshore regions
area_joined_off = area_joined.loc[area_off.index, :]

if not os.path.isfile("inputs/landuse_off_" + unopt_network_name + ".csv"):
    calculate_landuse_areas(unopt_network_name, geo_off, "offshore")
lands_off = pd.read_csv("inputs/landuse_off_" + unopt_network_name + ".csv").drop("type", axis=1)

lands_off["rel"] = lands_off.area_km2 / lands_off.bus.map(geo_off.area) * 100 * 1e6
area_joined_off = pd.concat([area_joined_off,
                             lands_off.set_index(
               pd.MultiIndex.from_product([[sce_name], [2030], list(geo_off.index), ["protected area"]])),
                             lands_off.set_index(
               pd.MultiIndex.from_product([[sce_name], [2040], list(geo_off.index), ["protected area"]])),
                             lands_off.set_index(
               pd.MultiIndex.from_product([[sce_name], [2050], list(geo_off.index), ["protected area"]]))])\
           .drop("bus", axis=1)

area_joined_off["oly_field"] = (area_joined_off["area_km2"] * 1e6 / olympic_stadium_area).sort_index()
#geo_off = geo_off.rename(index=lambda idx: idx + ' offshore')


#### WATER CALC ####

time_step = n.snapshot_weightings.iloc[0,0]  # in hours

# build or import powerplants allocated to pypsa-eur regions using european dataset w/ cooling technologies
if os.path.isfile("inputs/wui_per_region_technology.csv"):
    df_ppl = pd.read_csv("inputs/wui_per_region_technology.csv", index_col=[0, 1])
else:
    create_powerplants_df_wui(df_wui, geo_on)
    df_ppl = pd.read_csv("inputs/wui_per_region_technology.csv", index_col=[0, 1])

water_tys = []

for ty in target_years:

    # import optimized network per target year
    n = pypsa.Network(f"../../results/{sce_name}/postnetworks/{opt_network_name}_{ty}.nc") #

    # filter relevant techs
    gens_t = n.generators_t.p.filter(like='ror')
    links_t = n.links_t.p1.filter(regex=" oil|OCGT|CCGT|lignite|coal|nuclear").abs()
    links_t = links_t.drop(links_t.filter(regex='rural|urban').columns, axis=1)
    links_t_chp = n.links_t.p1.filter(regex="urban central solid biomass CHP-|urban central gas CHP-").abs() ##
    stores_t = n.storage_units_t.p_dispatch.filter(regex="PHS|hydro").abs()
    gens_t = pd.concat([gens_t, links_t, links_t_chp, stores_t], axis=1).sum()

    # get p_nom_opt to calc capacity factor later
    gens = n.generators[n.generators.carrier == "ror"][["p_nom_opt"]]
    links = n.links[
        n.links.index.str.contains(" oil|OCGT|CCGT|lignite|coal|nuclear") & \
                                    ~n.links.index.str.contains("rural|urban")][["p_nom_opt"]]
    links_chp = n.links[n.links.carrier.str.contains("CHP") & ~n.links.carrier.str.contains("CC")][["p_nom_opt"]] ##
    stores = n.storage_units[(n.storage_units.carrier.str.contains("PHS|hydro"))][["p_nom_opt"]]
    gens = pd.concat([gens, links, links_chp, stores]) ##

    if list(gens_t.index) != list(gens.index):
        gens.set_index(gens_t.index, inplace=True)
        print(f"set {ty} index identical")

    gens.index = gens.index.str.replace('urban central solid biomass CHP', 'biomass', regex=True) ##
    gens.index = gens.index.str.replace('urban central gas CHP', 'gas', regex=True) ##
    multi_idx = [[(" ").join(idx[:2]), idx[2].split("-")[0]] for idx in gens.index.str.split(' ')]
    gens.index = pd.MultiIndex.from_tuples(multi_idx, names=["bus", "carrier"])
    gens["e_nom_opt"] = (gens_t * 1e-3 * time_step).values

    gens = gens.rename({"OCGT": "gas", "CCGT": "gas", "ror": "hydro", "PHS": "hydro"})
    gens = gens.groupby(level=("bus", "carrier")).sum()
    gens["cap_factor"] = gens["e_nom_opt"] / (gens["p_nom_opt"] * 8760e-3) * 100

    gens = gens.join(df_ppl, how='left')

    agg_wuis = df_ppl.groupby("carrier").mean(numeric_only=True)
    for idx in gens[gens.wui_c.isnull() | gens.wui_w.isnull()].index:
        gens.loc[idx, "wui_c"] = agg_wuis.loc[idx[1], "wui_c"]
        gens.loc[idx, "wui_w"] = agg_wuis.loc[idx[1], "wui_w"]

    # derive water usage
    gens["water_miom3"] = gens["wui_c"] * gens["e_nom_opt"] * 1e-3  # e_nom_opt GWh

    # electrolysis calc is done using static values
    links_t = n.links_t.p1.filter(like='Electrolysis').abs()
    links_t = (links_t.mean() * 365 * 24 / time_step)  # h2 production in MWh
    multi_idx = [[(" ").join(idx[:2]), (" ").join(idx[2:]).split("-")[0]] for idx in links_t.index.str.split(' ')]
    links = pd.DataFrame(index=pd.MultiIndex.from_tuples(multi_idx, names=['bus', 'carrier']))
    links["water_miom3"] = links_t.values * H2O_elec_MWh * 1e-9
    links_ele = links.groupby(level=("bus", "carrier")).sum()

    if smr:
        # smr calc is done using static values
        links_t = n.links_t.p1.filter(like='SMR-').abs()
        links_t = (links_t.mean() * 365 * 24 / time_step)  # h2 production in MWh
        multi_idx = [[(" ").join(idx[:2]), (" ").join(idx[2:]).split("-")[0]] for idx in links_t.index.str.split(' ')]
        links = pd.DataFrame(index=pd.MultiIndex.from_tuples(multi_idx, names=['bus', 'carrier']))
        links["water_miom3"] = links_t.values * H2O_SMR_MWh * 1e-9
        links_smr = links.groupby(level=("bus", "carrier")).sum()
        links_ele = pd.concat([links_ele, links_smr], axis=0)
        # links_ele = links_ele.rename({"H2 Electrolysis": "H2 Production",
        #                              "SMR": "H2 Production"}).groupby(level=("bus", "carrier")).sum()

    # join frames
    water_ty = pd.concat([gens[["water_miom3"]], links_ele], axis=0)

    water_ty["sce_name"] = sce_name
    water_ty["target_year"] = ty
    water_ty = water_ty.sort_index().set_index(["sce_name", "target_year"], append=True).reorder_levels([2, 3, 0, 1])
    water_ty.index.names = ["sce_name", "target_year", "bus", "type"]

    water_tys.append(water_ty)

# join techs relevant for water usage
water_joined = pd.concat(water_tys)
water_joined["oly_pool"] = (water_joined["water_miom3"] * 1e6 / olympic_pool_volume).round(0)
water_joined["onshore"] = True

area_joined["onshore"] = True
area_joined_off["onshore"] = False
area_joined = pd.concat([area_joined, area_joined_off])

if "pac" in sce_name and unopt_network_name == "elec_s_44":

    with open("inputs/regions_onshore_elec_s_50.geojson") as f:
        json50 = json.load(f)
    with open("../../resources/{sce_name}/regions_onshore_elec_s_44.geojson") as f:
        json44 = json.load(f)

    rename_dict = {}
    for f50 in json50["features"]:
        for f44 in json44["features"]:
            if f50["geometry"] == f44["geometry"]:
                rename_dict[f44["properties"]["name"]] = f50["properties"]["name"]

    area_joined = area_joined.rename(index=rename_dict)
    water_joined = water_joined.rename(index=rename_dict)


            # export
area_joined.to_csv(f"outputs/{sce_name}_area_joined.csv")
water_joined.to_csv(f"outputs/{sce_name}_water_joined.csv")