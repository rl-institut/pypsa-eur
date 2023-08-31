#!/usr/bin/env python
# coding: utf-8

# ## **0)** Import packages

import atlite
import pandas as pd
from atlite.gis import shape_availability, ExclusionContainer

import warnings
warnings.filterwarnings("ignore")


# ## **1)** Analysis

def calculate_landuse_areas(unopt_network_name, network_geom, mode="onshore"):

    regions = network_geom
    # bounds = regions.cascaded_union.buffer(1).bounds
    # cutout = atlite.Cutout("europe", module="era5", bounds=bounds, time="2013")

    CORINE = '../../data/bundle/corine/g100_clc12_V18_5.tif'
    NATURA = '../../resources/{sce_name}/natura.tiff'

    grid_codes = pd.read_csv('../../data/bundle/corine/clc_legend.csv')[["GRID_CODE", "LABEL3"]]

    if mode == "onshore":

        share_urban = pd.DataFrame(index=regions.name)
        share_urban["area_km2"] = None

        for i in range(len(regions)):

            excluder = ExclusionContainer()
            excluder.add_raster(CORINE, codes=[1,2,3,5,6,9,10,11])

            reg = regions.loc[[i]].geometry.to_crs(excluder.crs)
            masked, transform= shape_availability(reg, excluder)

            share_urban.loc[regions.loc[i, "name"], "area_km2"] = \
                (1 - masked.sum() * excluder.res**2 / reg.geometry.item().area) * reg.geometry.item().area * 1e-6

    else:
        regions = regions.reset_index().rename(columns={regions.index.name:'name'})

    share_protected = pd.DataFrame(index=regions.name)
    share_protected["area_km2"] = None

    for i in range(len(regions)):

        excluder = ExclusionContainer()
        excluder.add_raster(NATURA, nodata=0, allow_no_overlap=True)

        reg = regions.loc[[i]].geometry.to_crs(excluder.crs)
        masked, transform= shape_availability(reg, excluder)

        share_protected.loc[regions.loc[i, "name"], "area_km2"] = \
            (1 - masked.sum() * excluder.res**2 / reg.geometry.item().area) * reg.geometry.item().area * 1e-6

    share_protected["type"] = "protected area"

    if mode == "onshore":
        share_urban["type"] = "urban area"
        land_use = pd.concat([share_urban, share_protected])
        land_use = land_use.reset_index().rename(columns={"name": "bus"})
        land_use.to_csv("inputs/landuse_on_" + unopt_network_name + ".csv", index=False)
    else:
        land_use = share_protected
        land_use = land_use.reset_index().rename(columns={"name": "bus"})
        land_use.to_csv("inputs/landuse_off_" + unopt_network_name + ".csv", index=False)





