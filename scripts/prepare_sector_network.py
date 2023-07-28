# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Adds all sector-coupling components to the network, including demand and supply
technologies for the buildings, transport and industry sectors.
"""

import logging
import os
import re
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import (
    generate_periodic_profiles,
    override_component_attrs,
    update_config_with_sector_opts,
)
from build_energy_totals import build_co2_totals, build_eea_co2, build_eurostat_co2
from networkx.algorithms import complement
from networkx.algorithms.connectivity.edge_augmentation import k_edge_augmentation
from pypsa.geo import haversine_pts
from pypsa.io import import_components_from_dataframe
from scipy.stats import beta
from vresutils.costdata import annuity

logger = logging.getLogger(__name__)

from types import SimpleNamespace

spatial = SimpleNamespace()

from packaging.version import Version, parse

pd_version = parse(pd.__version__)
agg_group_kwargs = dict(numeric_only=False) if pd_version >= Version("1.3") else {}


def define_spatial(nodes, options):
    """
    Namespace for spatial.

    Parameters
    ----------
    nodes : list-like
    """
    global spatial

    spatial.nodes = nodes

    # biomass

    spatial.biomass = SimpleNamespace()

    if options.get("biomass_spatial", options["biomass_transport"]):
        spatial.biomass.nodes = nodes + " solid biomass"
        spatial.biomass.locations = nodes
        spatial.biomass.industry = nodes + " solid biomass for industry"
        spatial.biomass.industry_cc = nodes + " solid biomass for industry CC"
    else:
        spatial.biomass.nodes = ["EU solid biomass"]
        spatial.biomass.locations = ["EU"]
        spatial.biomass.industry = ["solid biomass for industry"]
        spatial.biomass.industry_cc = ["solid biomass for industry CC"]

    spatial.biomass.df = pd.DataFrame(vars(spatial.biomass), index=nodes)

    # co2

    spatial.co2 = SimpleNamespace()

    if options["co2_spatial"]:
        spatial.co2.nodes = nodes + " co2 stored"
        spatial.co2.locations = nodes
        spatial.co2.vents = nodes + " co2 vent"
        spatial.co2.process_emissions = nodes + " process emissions"
    else:
        spatial.co2.nodes = ["co2 stored"]
        spatial.co2.locations = ["EU"]
        spatial.co2.vents = ["co2 vent"]
        spatial.co2.process_emissions = ["process emissions"]

    spatial.co2.df = pd.DataFrame(vars(spatial.co2), index=nodes)

    # gas

    spatial.gas = SimpleNamespace()

    if options["gas_network"]:
        spatial.gas.nodes = nodes + " gas"
        spatial.gas.locations = nodes
        spatial.gas.biogas = nodes + " biogas"
        spatial.gas.industry = nodes + " gas for industry"
        spatial.gas.industry_cc = nodes + " gas for industry CC"
        spatial.gas.biogas_to_gas = nodes + " biogas to gas"
    else:
        spatial.gas.nodes = ["EU gas"]
        spatial.gas.locations = ["EU"]
        spatial.gas.biogas = ["EU biogas"]
        spatial.gas.industry = ["gas for industry"]
        spatial.gas.biogas_to_gas = ["EU biogas to gas"]
        if options.get("co2_spatial", options["co2network"]):
            spatial.gas.industry_cc = nodes + " gas for industry CC"
        else:
            spatial.gas.industry_cc = ["gas for industry CC"]

    spatial.gas.df = pd.DataFrame(vars(spatial.gas), index=nodes)

    # ammonia

    if options.get("ammonia"):
        spatial.ammonia = SimpleNamespace()
        if options.get("ammonia") == "regional":
            spatial.ammonia.nodes = nodes + " NH3"
            spatial.ammonia.locations = nodes
        else:
            spatial.ammonia.nodes = ["EU NH3"]
            spatial.ammonia.locations = ["EU"]

        spatial.ammonia.df = pd.DataFrame(vars(spatial.ammonia), index=nodes)

    # hydrogen
    spatial.h2 = SimpleNamespace()
    spatial.h2.nodes = nodes + " H2"
    spatial.h2.locations = nodes

    # methanol
    spatial.methanol = SimpleNamespace()
    spatial.methanol.nodes = ["EU methanol"]
    spatial.methanol.locations = ["EU"]

    # oil
    spatial.oil = SimpleNamespace()
    spatial.oil.nodes = ["EU oil"]
    spatial.oil.locations = ["EU"]

    # uranium
    spatial.uranium = SimpleNamespace()
    spatial.uranium.nodes = ["EU uranium"]
    spatial.uranium.locations = ["EU"]

    # coal
    spatial.coal = SimpleNamespace()
    spatial.coal.nodes = ["EU coal"]
    spatial.coal.locations = ["EU"]

    # lignite
    spatial.lignite = SimpleNamespace()
    spatial.lignite.nodes = ["EU lignite"]
    spatial.lignite.locations = ["EU"]

    return spatial


from types import SimpleNamespace

spatial = SimpleNamespace()


def emission_sectors_from_opts(opts):
    sectors = ["electricity"]
    if "T" in opts:
        sectors += ["rail non-elec", "road non-elec"]
    if "H" in opts:
        sectors += ["residential non-elec", "services non-elec"]
    if "I" in opts:
        sectors += [
            "industrial non-elec",
            "industrial processes",
            "domestic aviation",
            "international aviation",
            "domestic navigation",
            "international navigation",
        ]
    if "A" in opts:
        sectors += ["agriculture"]

    return sectors


def get(item, investment_year=None):
    """
    Check whether item depends on investment year.
    """
    if isinstance(item, dict):
        return item[investment_year]
    else:
        return item


def co2_emissions_year(
    countries, input_eurostat, opts, emissions_scope, report_year, year
):
    """
    Calculate CO2 emissions in one specific year (e.g. 1990 or 2018).
    """
    emissions_scope = snakemake.config["energy"]["emissions"]
    eea_co2 = build_eea_co2(snakemake.input.co2, year, emissions_scope)

    # TODO: read Eurostat data from year > 2014
    # this only affects the estimation of CO2 emissions for BA, RS, AL, ME, MK
    report_year = snakemake.config["energy"]["eurostat_report_year"]
    if year > 2014:
        eurostat_co2 = build_eurostat_co2(
            input_eurostat, countries, report_year, year=2014
        )
    else:
        eurostat_co2 = build_eurostat_co2(input_eurostat, countries, report_year, year)

    co2_totals = build_co2_totals(countries, eea_co2, eurostat_co2)

    sectors = emission_sectors_from_opts(opts)

    co2_emissions = co2_totals.loc[countries, sectors].sum().sum()

    # convert MtCO2 to GtCO2
    co2_emissions *= 0.001

    return co2_emissions


# TODO: move to own rule with sector-opts wildcard?
def build_carbon_budget(o, input_eurostat, fn, emissions_scope, report_year):
    """
    Distribute carbon budget following beta or exponential transition path.
    """
    # opts?

    if "be" in o:
        # beta decay
        carbon_budget = float(o[o.find("cb") + 2 : o.find("be")])
        be = float(o[o.find("be") + 2 :])
    if "ex" in o:
        # exponential decay
        carbon_budget = float(o[o.find("cb") + 2 : o.find("ex")])
        r = float(o[o.find("ex") + 2 :])

    countries = snakemake.config["countries"]

    e_1990 = co2_emissions_year(
        countries, input_eurostat, opts, emissions_scope, report_year, year=1990
    )

    # emissions at the beginning of the path (last year available 2018)
    e_0 = co2_emissions_year(
        countries, input_eurostat, opts, emissions_scope, report_year, year=2018
    )

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    t_0 = planning_horizons[0]

    if "be" in o:
        # final year in the path
        t_f = t_0 + (2 * carbon_budget / e_0).round(0)

        def beta_decay(t):
            cdf_term = (t - t_0) / (t_f - t_0)
            return (e_0 / e_1990) * (1 - beta.cdf(cdf_term, be, be))

        # emissions (relative to 1990)
        co2_cap = pd.Series({t: beta_decay(t) for t in planning_horizons}, name=o)

    if "ex" in o:
        T = carbon_budget / e_0
        m = (1 + np.sqrt(1 + r * T)) / T

        def exponential_decay(t):
            return (e_0 / e_1990) * (1 + (m + r) * (t - t_0)) * np.exp(-m * (t - t_0))

        co2_cap = pd.Series(
            {t: exponential_decay(t) for t in planning_horizons}, name=o
        )

    # TODO log in Snakefile
    csvs_folder = fn.rsplit("/", 1)[0]
    if not os.path.exists(csvs_folder):
        os.makedirs(csvs_folder)
    co2_cap.to_csv(fn, float_format="%.3f")


def add_lifetime_wind_solar(n, costs):
    """
    Add lifetime for solar and wind generators.
    """
    for carrier in ["solar", "onwind", "offwind"]:
        gen_i = n.generators.index.str.contains(carrier)
        n.generators.loc[gen_i, "lifetime"] = costs.at[carrier, "lifetime"]
        n.generators.loc[gen_i, "p_nom_min"] = 0


def haversine(p):
    coord0 = n.buses.loc[p.bus0, ["x", "y"]].values
    coord1 = n.buses.loc[p.bus1, ["x", "y"]].values
    return 1.5 * haversine_pts(coord0, coord1)


def create_network_topology(
    n, prefix, carriers=["DC"], connector=" -> ", bidirectional=True
):
    """
    Create a network topology from transmission lines and link carrier
    selection.

    Parameters
    ----------
    n : pypsa.Network
    prefix : str
    carriers : list-like
    connector : str
    bidirectional : bool, default True
        True: one link for each connection
        False: one link for each connection and direction (back and forth)

    Returns
    -------
    pd.DataFrame with columns bus0, bus1, length, underwater_fraction
    """

    ln_attrs = ["bus0", "bus1", "length"]
    lk_attrs = ["bus0", "bus1", "length", "underwater_fraction"]
    lk_attrs = n.links.columns.intersection(lk_attrs)

    candidates = pd.concat(
        [n.lines[ln_attrs], n.links.loc[n.links.carrier.isin(carriers), lk_attrs]]
    ).fillna(0)

    # base network topology purely on location not carrier
    candidates["bus0"] = candidates.bus0.map(n.buses.location)
    candidates["bus1"] = candidates.bus1.map(n.buses.location)

    positive_order = candidates.bus0 < candidates.bus1
    candidates_p = candidates[positive_order]
    swap_buses = {"bus0": "bus1", "bus1": "bus0"}
    candidates_n = candidates[~positive_order].rename(columns=swap_buses)
    candidates = pd.concat([candidates_p, candidates_n])

    def make_index(c):
        return prefix + c.bus0 + connector + c.bus1

    topo = candidates.groupby(["bus0", "bus1"], as_index=False).mean()
    topo.index = topo.apply(make_index, axis=1)

    if not bidirectional:
        topo_reverse = topo.copy()
        topo_reverse.rename(columns=swap_buses, inplace=True)
        topo_reverse.index = topo_reverse.apply(make_index, axis=1)
        topo = pd.concat([topo, topo_reverse])

    return topo


# TODO merge issue with PyPSA-Eur
def update_wind_solar_costs(n, costs):
    """
    Update costs for wind and solar generators added with pypsa-eur to those
    cost in the planning year.
    """
    # NB: solar costs are also manipulated for rooftop
    # when distribution grid is inserted
    n.generators.loc[n.generators.carrier == "solar", "capital_cost"] = costs.at[
        "solar-utility", "fixed"
    ]

    n.generators.loc[n.generators.carrier == "onwind", "capital_cost"] = costs.at[
        "onwind", "fixed"
    ]

    # for offshore wind, need to calculated connection costs

    # assign clustered bus
    # map initial network -> simplified network
    busmap_s = pd.read_csv(snakemake.input.busmap_s, index_col=0).squeeze()
    busmap_s.index = busmap_s.index.astype(str)
    busmap_s = busmap_s.astype(str)
    # map simplified network -> clustered network
    busmap = pd.read_csv(snakemake.input.busmap, index_col=0).squeeze()
    busmap.index = busmap.index.astype(str)
    busmap = busmap.astype(str)
    # map initial network -> clustered network
    clustermaps = busmap_s.map(busmap)

    # code adapted from pypsa-eur/scripts/add_electricity.py
    for connection in ["dc", "ac"]:
        tech = "offwind-" + connection
        profile = snakemake.input["profile_offwind_" + connection]
        with xr.open_dataset(profile) as ds:
            underwater_fraction = ds["underwater_fraction"].to_pandas()
            connection_cost = (
                snakemake.config["lines"]["length_factor"]
                * ds["average_distance"].to_pandas()
                * (
                    underwater_fraction
                    * costs.at[tech + "-connection-submarine", "fixed"]
                    + (1.0 - underwater_fraction)
                    * costs.at[tech + "-connection-underground", "fixed"]
                )
            )

            # convert to aggregated clusters with weighting
            weight = ds["weight"].to_pandas()

            # e.g. clusters == 37m means that VRE generators are left
            # at clustering of simplified network, but that they are
            # connected to 37-node network
            if snakemake.wildcards.clusters[-1:] == "m":
                genmap = busmap_s
            else:
                genmap = clustermaps

            connection_cost = (connection_cost * weight).groupby(
                genmap
            ).sum() / weight.groupby(genmap).sum()

            capital_cost = (
                costs.at["offwind", "fixed"]
                + costs.at[tech + "-station", "fixed"]
                + connection_cost
            )

            logger.info(
                "Added connection cost of {:0.0f}-{:0.0f} Eur/MW/a to {}".format(
                    connection_cost[0].min(), connection_cost[0].max(), tech
                )
            )

            n.generators.loc[
                n.generators.carrier == tech, "capital_cost"
            ] = capital_cost.rename(index=lambda node: node + " " + tech)


def add_carrier_buses(n, carrier, nodes=None):
    """
    Add buses to connect e.g. coal, nuclear and oil plants.
    """
    if nodes is None:
        nodes = vars(spatial)[carrier].nodes
    location = vars(spatial)[carrier].locations

    # skip if carrier already exists
    if carrier in n.carriers.index:
        return

    if not isinstance(nodes, pd.Index):
        nodes = pd.Index(nodes)

    n.add("Carrier", carrier)

    unit = "MWh_LHV" if carrier == "gas" else "MWh_th"

    n.madd("Bus", nodes, location=location, carrier=carrier, unit=unit)

    # capital cost could be corrected to e.g. 0.2 EUR/kWh * annuity and O&M
    n.madd(
        "Store",
        nodes + " Store",
        bus=nodes,
        e_nom_extendable=True,
        e_cyclic=True,
        carrier=carrier,
        capital_cost=0.2
        * costs.at[carrier, "discount rate"],  # preliminary value to avoid zeros
    )

    n.madd(
        "Generator",
        nodes,
        bus=nodes,
        p_nom_extendable=True,
        carrier=carrier,
        marginal_cost=costs.at[carrier, "fuel"],
    )


# TODO: PyPSA-Eur merge issue
def remove_elec_base_techs(n):
    """
    Remove conventional generators (e.g. OCGT) and storage units (e.g.
    batteries and H2) from base electricity-only network, since they're added
    here differently using links.
    """
    for c in n.iterate_components(snakemake.config["pypsa_eur"]):
        to_keep = snakemake.config["pypsa_eur"][c.name]
        to_remove = pd.Index(c.df.carrier.unique()).symmetric_difference(to_keep)
        if to_remove.empty:
            continue
        logger.info(f"Removing {c.list_name} with carrier {list(to_remove)}")
        names = c.df.index[c.df.carrier.isin(to_remove)]
        n.mremove(c.name, names)
        n.carriers.drop(to_remove, inplace=True, errors="ignore")


# TODO: PyPSA-Eur merge issue
def remove_non_electric_buses(n):
    """
    Remove buses from pypsa-eur with carriers which are not AC buses.
    """
    to_drop = list(n.buses.query("carrier not in ['AC', 'DC']").carrier.unique())
    if to_drop:
        logger.info(f"Drop buses from PyPSA-Eur with carrier: {to_drop}")
        n.buses = n.buses[n.buses.carrier.isin(["AC", "DC"])]


def patch_electricity_network(n):
    remove_elec_base_techs(n)
    remove_non_electric_buses(n)
    update_wind_solar_costs(n, costs)
    n.loads["carrier"] = "electricity"
    n.buses["location"] = n.buses.index
    n.buses["unit"] = "MWh_el"
    # remove trailing white space of load index until new PyPSA version after v0.18.
    n.loads.rename(lambda x: x.strip(), inplace=True)
    n.loads_t.p_set.rename(lambda x: x.strip(), axis=1, inplace=True)


def add_co2_tracking(n, options):
    # minus sign because opposite to how fossil fuels used:
    # CH4 burning puts CH4 down, atmosphere up
    n.add("Carrier", "co2", co2_emissions=-1.0)

    # this tracks CO2 in the atmosphere
    n.add("Bus", "co2 atmosphere", location="EU", carrier="co2", unit="t_co2")

    # can also be negative
    n.add(
        "Store",
        "co2 atmosphere",
        e_nom_extendable=True,
        e_min_pu=-1,
        carrier="co2",
        bus="co2 atmosphere",
    )

    # this tracks CO2 stored, e.g. underground
    n.madd(
        "Bus",
        spatial.co2.nodes,
        location=spatial.co2.locations,
        carrier="co2 stored",
        unit="t_co2",
    )

    if options["regional_co2_sequestration_potential"]["enable"]:
        upper_limit = (
            options["regional_co2_sequestration_potential"]["max_size"] * 1e3
        )  # Mt
        annualiser = options["regional_co2_sequestration_potential"]["years_of_storage"]
        e_nom_max = pd.read_csv(
            snakemake.input.sequestration_potential, index_col=0
        ).squeeze()
        e_nom_max = (
            e_nom_max.reindex(spatial.co2.locations)
            .fillna(0.0)
            .clip(upper=upper_limit)
            .mul(1e6)
            / annualiser
        )  # t
        e_nom_max = e_nom_max.rename(index=lambda x: x + " co2 stored")
    else:
        e_nom_max = np.inf

    n.madd(
        "Store",
        spatial.co2.nodes,
        e_nom_extendable=True,
        e_nom_max=e_nom_max,
        capital_cost=options["co2_sequestration_cost"],
        carrier="co2 stored",
        bus=spatial.co2.nodes,
    )

    n.add("Carrier", "co2 stored")

    if options["co2_vent"]:
        n.madd(
            "Link",
            spatial.co2.vents,
            bus0=spatial.co2.nodes,
            bus1="co2 atmosphere",
            carrier="co2 vent",
            efficiency=1.0,
            p_nom_extendable=True,
        )


def add_co2_network(n, costs):
    logger.info("Adding CO2 network.")
    co2_links = create_network_topology(n, "CO2 pipeline ")

    cost_onshore = (
        (1 - co2_links.underwater_fraction)
        * costs.at["CO2 pipeline", "fixed"]
        * co2_links.length
    )
    cost_submarine = (
        co2_links.underwater_fraction
        * costs.at["CO2 submarine pipeline", "fixed"]
        * co2_links.length
    )
    capital_cost = cost_onshore + cost_submarine

    n.madd(
        "Link",
        co2_links.index,
        bus0=co2_links.bus0.values + " co2 stored",
        bus1=co2_links.bus1.values + " co2 stored",
        p_min_pu=-1,
        p_nom_extendable=True,
        length=co2_links.length.values,
        capital_cost=capital_cost.values,
        carrier="CO2 pipeline",
        lifetime=costs.at["CO2 pipeline", "lifetime"],
    )


def add_allam(n, costs):
    logger.info("Adding Allam cycle gas power plants.")

    nodes = pop_layout.index

    n.madd(
        "Link",
        nodes,
        suffix=" allam",
        bus0=spatial.gas.df.loc[nodes, "nodes"].values,
        bus1=nodes,
        bus2=spatial.co2.df.loc[nodes, "nodes"].values,
        carrier="allam",
        p_nom_extendable=True,
        # TODO: add costs to technology-data
        capital_cost=0.6 * 1.5e6 * 0.1,  # efficiency * EUR/MW * annuity
        marginal_cost=2,
        efficiency=0.6,
        efficiency2=costs.at["gas", "CO2 intensity"],
        lifetime=30.0,
    )


def add_dac(n, costs):
    heat_carriers = ["urban central heat", "services urban decentral heat"]
    heat_buses = n.buses.index[n.buses.carrier.isin(heat_carriers)]
    locations = n.buses.location[heat_buses]

    efficiency2 = -(
        costs.at["direct air capture", "electricity-input"]
        + costs.at["direct air capture", "compression-electricity-input"]
    )
    efficiency3 = -(
        costs.at["direct air capture", "heat-input"]
        - costs.at["direct air capture", "compression-heat-output"]
    )

    n.madd(
        "Link",
        heat_buses.str.replace(" heat", " DAC"),
        bus0="co2 atmosphere",
        bus1=spatial.co2.df.loc[locations, "nodes"].values,
        bus2=locations.values,
        bus3=heat_buses,
        carrier="DAC",
        capital_cost=costs.at["direct air capture", "fixed"],
        efficiency=1.0,
        efficiency2=efficiency2,
        efficiency3=efficiency3,
        p_nom_extendable=True,
        lifetime=costs.at["direct air capture", "lifetime"],
    )


def add_co2limit(n, nyears=1.0, limit=0.0):
    logger.info(f"Adding CO2 budget limit as per unit of 1990 levels of {limit}")

    countries = snakemake.config["countries"]

    sectors = emission_sectors_from_opts(opts)

    # convert Mt to tCO2
    co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

    co2_limit = co2_totals.loc[countries, sectors].sum().sum()

    co2_limit *= limit * nyears

    n.add(
        "GlobalConstraint",
        "CO2Limit",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2_limit,
    )


# TODO PyPSA-Eur merge issue
def average_every_nhours(n, offset):
    logger.info(f"Resampling the network to {offset}")
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                if c.list_name == "stores" and k == "e_max_pu":
                    pnl[k] = df.resample(offset).min()
                elif c.list_name == "stores" and k == "e_min_pu":
                    pnl[k] = df.resample(offset).max()
                else:
                    pnl[k] = df.resample(offset).mean()

    return m


def cycling_shift(df, steps=1):
    """
    Cyclic shift on index of pd.Series|pd.DataFrame by number of steps.
    """
    df = df.copy()
    new_index = np.roll(df.index, steps)
    df.values[:] = df.reindex(index=new_index).values
    return df


def prepare_costs(cost_file, config, nyears):
    # set all asset costs and other parameters
    costs = pd.read_csv(cost_file, index_col=[0, 1]).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3

    # min_count=1 is important to generate NaNs which are then filled by fillna
    costs = (
        costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    )

    costs = costs.fillna(config["fill_values"])

    def annuity_factor(v):
        return annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100

    costs["fixed"] = [
        annuity_factor(v) * v["investment"] * nyears for i, v in costs.iterrows()
    ]

    return costs


def add_generation(n, costs):
    logger.info("Adding electricity generation")

    nodes = pop_layout.index

    fallback = {"OCGT": "gas"}
    conventionals = options.get("conventional_generation", fallback)

    for generator, carrier in conventionals.items():
        carrier_nodes = vars(spatial)[carrier].nodes

        add_carrier_buses(n, carrier, carrier_nodes)

        n.madd(
            "Link",
            nodes + " " + generator,
            bus0=carrier_nodes,
            bus1=nodes,
            bus2="co2 atmosphere",
            marginal_cost=costs.at[generator, "efficiency"]
            * costs.at[generator, "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at[generator, "efficiency"]
            * costs.at[generator, "fixed"],  # NB: fixed cost is per MWel
            p_nom_extendable=True,
            carrier=generator,
            efficiency=costs.at[generator, "efficiency"],
            efficiency2=costs.at[carrier, "CO2 intensity"],
            lifetime=costs.at[generator, "lifetime"],
        )


def add_ammonia(n, costs):
    logger.info("Adding ammonia carrier with synthesis, cracking and storage")

    nodes = pop_layout.index

    cf_industry = snakemake.config["industry"]

    n.add("Carrier", "NH3")

    n.madd(
        "Bus", spatial.ammonia.nodes, location=spatial.ammonia.locations, carrier="NH3"
    )

    n.madd(
        "Link",
        nodes,
        suffix=" Haber-Bosch",
        bus0=nodes,
        bus1=spatial.ammonia.nodes,
        bus2=nodes + " H2",
        p_nom_extendable=True,
        carrier="Haber-Bosch",
        efficiency=1
        / (
            cf_industry["MWh_elec_per_tNH3_electrolysis"]
            / cf_industry["MWh_NH3_per_tNH3"]
        ),  # output: MW_NH3 per MW_elec
        efficiency2=-cf_industry["MWh_H2_per_tNH3_electrolysis"]
        / cf_industry["MWh_elec_per_tNH3_electrolysis"],  # input: MW_H2 per MW_elec
        capital_cost=costs.at["Haber-Bosch", "fixed"],
        lifetime=costs.at["Haber-Bosch", "lifetime"],
    )

    n.madd(
        "Link",
        nodes,
        suffix=" ammonia cracker",
        bus0=spatial.ammonia.nodes,
        bus1=nodes + " H2",
        p_nom_extendable=True,
        carrier="ammonia cracker",
        efficiency=1 / cf_industry["MWh_NH3_per_MWh_H2_cracker"],
        capital_cost=costs.at["Ammonia cracker", "fixed"]
        / cf_industry["MWh_NH3_per_MWh_H2_cracker"],  # given per MW_H2
        lifetime=costs.at["Ammonia cracker", "lifetime"],
    )

    # Ammonia Storage
    n.madd(
        "Store",
        spatial.ammonia.nodes,
        suffix=" ammonia store",
        bus=spatial.ammonia.nodes,
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="ammonia store",
        capital_cost=costs.at["NH3 (l) storage tank incl. liquefaction", "fixed"],
        lifetime=costs.at["NH3 (l) storage tank incl. liquefaction", "lifetime"],
    )


def add_wave(n, wave_cost_factor):
    # TODO: handle in Snakefile
    wave_fn = "data/WindWaveWEC_GLTB.xlsx"

    # in kW
    capacity = pd.Series({"Attenuator": 750, "F2HB": 1000, "MultiPA": 600})

    # in EUR/MW
    annuity_factor = annuity(25, 0.07) + 0.03
    costs = (
        1e6
        * wave_cost_factor
        * annuity_factor
        * pd.Series({"Attenuator": 2.5, "F2HB": 2, "MultiPA": 1.5})
    )

    sheets = pd.read_excel(
        wave_fn,
        sheet_name=["FirthForth", "Hebrides"],
        usecols=["Attenuator", "F2HB", "MultiPA"],
        index_col=0,
        skiprows=[0],
        parse_dates=True,
    )

    wave = pd.concat(
        [sheets[l].divide(capacity, axis=1) for l in locations], keys=locations, axis=1
    )

    for wave_type in costs.index:
        n.add(
            "Generator",
            "Hebrides " + wave_type,
            bus="GB4 0",  # TODO this location is hardcoded
            p_nom_extendable=True,
            carrier="wave",
            capital_cost=costs[wave_type],
            p_max_pu=wave["Hebrides", wave_type],
        )


def insert_electricity_distribution_grid(n, costs):
    # TODO pop_layout?
    # TODO options?

    cost_factor = options["electricity_distribution_grid_cost_factor"]

    logger.info(
        f"Inserting electricity distribution grid with investment cost factor of {cost_factor:.2f}"
    )

    nodes = pop_layout.index

    n.madd(
        "Bus",
        nodes + " low voltage",
        location=nodes,
        carrier="low voltage",
        unit="MWh_el",
    )

    n.madd(
        "Link",
        nodes + " electricity distribution grid",
        bus0=nodes,
        bus1=nodes + " low voltage",
        p_nom_extendable=True,
        p_min_pu=-1,
        carrier="electricity distribution grid",
        efficiency=1,
        lifetime=costs.at["electricity distribution grid", "lifetime"],
        capital_cost=costs.at["electricity distribution grid", "fixed"] * cost_factor,
    )

    # this catches regular electricity load and "industry electricity" and
    # "agriculture machinery electric" and "agriculture electricity"
    loads = n.loads.index[n.loads.carrier.str.contains("electric")]
    n.loads.loc[loads, "bus"] += " low voltage"

    bevs = n.links.index[n.links.carrier == "BEV charger"]
    n.links.loc[bevs, "bus0"] += " low voltage"

    v2gs = n.links.index[n.links.carrier == "V2G"]
    n.links.loc[v2gs, "bus1"] += " low voltage"

    hps = n.links.index[n.links.carrier.str.contains("heat pump")]
    n.links.loc[hps, "bus0"] += " low voltage"

    rh = n.links.index[n.links.carrier.str.contains("resistive heater")]
    n.links.loc[rh, "bus0"] += " low voltage"

    mchp = n.links.index[n.links.carrier.str.contains("micro gas")]
    n.links.loc[mchp, "bus1"] += " low voltage"

    # set existing solar to cost of utility cost rather the 50-50 rooftop-utility
    solar = n.generators.index[n.generators.carrier == "solar"]
    n.generators.loc[solar, "capital_cost"] = costs.at["solar-utility", "fixed"]
    if snakemake.wildcards.clusters[-1:] == "m":
        simplified_pop_layout = pd.read_csv(
            snakemake.input.simplified_pop_layout, index_col=0
        )
        pop_solar = simplified_pop_layout.total.rename(index=lambda x: x + " solar")
    else:
        pop_solar = pop_layout.total.rename(index=lambda x: x + " solar")

    # add max solar rooftop potential assuming 0.1 kW/m2 and 10 m2/person,
    # i.e. 1 kW/person (population data is in thousands of people) so we get MW
    potential = 0.1 * 10 * pop_solar

    n.madd(
        "Generator",
        solar,
        suffix=" rooftop",
        bus=n.generators.loc[solar, "bus"] + " low voltage",
        carrier="solar rooftop",
        p_nom_extendable=True,
        p_nom_max=potential,
        marginal_cost=n.generators.loc[solar, "marginal_cost"],
        capital_cost=costs.at["solar-rooftop", "fixed"],
        efficiency=n.generators.loc[solar, "efficiency"],
        p_max_pu=n.generators_t.p_max_pu[solar],
        lifetime=costs.at["solar-rooftop", "lifetime"],
    )

    n.add("Carrier", "home battery")

    n.madd(
        "Bus",
        nodes + " home battery",
        location=nodes,
        carrier="home battery",
        unit="MWh_el",
    )

    n.madd(
        "Store",
        nodes + " home battery",
        bus=nodes + " home battery",
        e_cyclic=True,
        e_nom_extendable=True,
        carrier="home battery",
        capital_cost=costs.at["home battery storage", "fixed"],
        lifetime=costs.at["battery storage", "lifetime"],
    )

    n.madd(
        "Link",
        nodes + " home battery charger",
        bus0=nodes + " low voltage",
        bus1=nodes + " home battery",
        carrier="home battery charger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        capital_cost=costs.at["home battery inverter", "fixed"],
        p_nom_extendable=True,
        lifetime=costs.at["battery inverter", "lifetime"],
    )

    n.madd(
        "Link",
        nodes + " home battery discharger",
        bus0=nodes + " home battery",
        bus1=nodes + " low voltage",
        carrier="home battery discharger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        marginal_cost=options["marginal_cost_storage"],
        p_nom_extendable=True,
        lifetime=costs.at["battery inverter", "lifetime"],
    )


def insert_gas_distribution_costs(n, costs):
    # TODO options?

    f_costs = options["gas_distribution_grid_cost_factor"]

    logger.info(
        f"Inserting gas distribution grid with investment cost factor of {f_costs}"
    )

    capital_cost = costs.loc["electricity distribution grid"]["fixed"] * f_costs

    # gas boilers
    gas_b = n.links.index[
        n.links.carrier.str.contains("gas boiler")
        & (~n.links.carrier.str.contains("urban central"))
    ]
    n.links.loc[gas_b, "capital_cost"] += capital_cost

    # micro CHPs
    mchp = n.links.index[n.links.carrier.str.contains("micro gas")]
    n.links.loc[mchp, "capital_cost"] += capital_cost


def add_electricity_grid_connection(n, costs):
    carriers = ["onwind", "solar"]

    gens = n.generators.index[n.generators.carrier.isin(carriers)]

    n.generators.loc[gens, "capital_cost"] += costs.at[
        "electricity grid connection", "fixed"
    ]


def add_storage_and_grids(n, costs):
    logger.info("Add hydrogen storage")

    nodes = pop_layout.index

    n.add("Carrier", "H2")

    n.madd("Bus", nodes + " H2", location=nodes, carrier="H2", unit="MWh_LHV")

    n.madd(
        "Link",
        nodes + " H2 Electrolysis",
        bus1=nodes + " H2",
        bus0=nodes,
        p_nom_extendable=True,
        carrier="H2 Electrolysis",
        efficiency=costs.at["electrolysis", "efficiency"],
        capital_cost=costs.at["electrolysis", "fixed"],
        lifetime=costs.at["electrolysis", "lifetime"],
    )

    n.madd(
        "Link",
        nodes + " H2 Fuel Cell",
        bus0=nodes + " H2",
        bus1=nodes,
        p_nom_extendable=True,
        carrier="H2 Fuel Cell",
        efficiency=costs.at["fuel cell", "efficiency"],
        capital_cost=costs.at["fuel cell", "fixed"]
        * costs.at["fuel cell", "efficiency"],  # NB: fixed cost is per MWel
        lifetime=costs.at["fuel cell", "lifetime"],
    )

    cavern_types = snakemake.config["sector"]["hydrogen_underground_storage_locations"]
    h2_caverns = pd.read_csv(snakemake.input.h2_cavern, index_col=0)

    if not h2_caverns.empty and options["hydrogen_underground_storage"]:
        h2_caverns = h2_caverns[cavern_types].sum(axis=1)

        # only use sites with at least 2 TWh potential
        h2_caverns = h2_caverns[h2_caverns > 2]

        # convert TWh to MWh
        h2_caverns = h2_caverns * 1e6

        # clip at 1000 TWh for one location
        h2_caverns.clip(upper=1e9, inplace=True)

        logger.info("Add hydrogen underground storage")

        h2_capital_cost = costs.at["hydrogen storage underground", "fixed"]

        n.madd(
            "Store",
            h2_caverns.index + " H2 Store",
            bus=h2_caverns.index + " H2",
            e_nom_extendable=True,
            e_nom_max=h2_caverns.values,
            e_cyclic=True,
            carrier="H2 Store",
            capital_cost=h2_capital_cost,
            lifetime=costs.at["hydrogen storage underground", "lifetime"],
        )

    # hydrogen stored overground (where not already underground)
    h2_capital_cost = costs.at[
        "hydrogen storage tank type 1 including compressor", "fixed"
    ]
    nodes_overground = h2_caverns.index.symmetric_difference(nodes)

    n.madd(
        "Store",
        nodes_overground + " H2 Store",
        bus=nodes_overground + " H2",
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="H2 Store",
        capital_cost=h2_capital_cost,
    )

    if options["gas_network"] or options["H2_retrofit"]:
        fn = snakemake.input.clustered_gas_network
        gas_pipes = pd.read_csv(fn, index_col=0)

    if options["gas_network"]:
        logger.info(
            "Add natural gas infrastructure, incl. LNG terminals, production and entry-points."
        )

        if options["H2_retrofit"]:
            gas_pipes["p_nom_max"] = gas_pipes.p_nom
            gas_pipes["p_nom_min"] = 0.0
            # 0.1 EUR/MWkm/a to prefer decommissioning to address degeneracy
            gas_pipes["capital_cost"] = 0.1 * gas_pipes.length
        else:
            gas_pipes["p_nom_max"] = np.inf
            gas_pipes["p_nom_min"] = gas_pipes.p_nom
            gas_pipes["capital_cost"] = (
                gas_pipes.length * costs.at["CH4 (g) pipeline", "fixed"]
            )

        n.madd(
            "Link",
            gas_pipes.index,
            bus0=gas_pipes.bus0 + " gas",
            bus1=gas_pipes.bus1 + " gas",
            p_min_pu=gas_pipes.p_min_pu,
            p_nom=gas_pipes.p_nom,
            p_nom_extendable=True,
            p_nom_max=gas_pipes.p_nom_max,
            p_nom_min=gas_pipes.p_nom_min,
            length=gas_pipes.length,
            capital_cost=gas_pipes.capital_cost,
            tags=gas_pipes.name,
            carrier="gas pipeline",
            lifetime=costs.at["CH4 (g) pipeline", "lifetime"],
        )

        # remove fossil generators where there is neither
        # production, LNG terminal, nor entry-point beyond system scope

        fn = snakemake.input.gas_input_nodes_simplified
        gas_input_nodes = pd.read_csv(fn, index_col=0)

        unique = gas_input_nodes.index.unique()
        gas_i = n.generators.carrier == "gas"
        internal_i = ~n.generators.bus.map(n.buses.location).isin(unique)

        remove_i = n.generators[gas_i & internal_i].index
        n.generators.drop(remove_i, inplace=True)

        p_nom = gas_input_nodes.sum(axis=1).rename(lambda x: x + " gas")
        n.generators.loc[gas_i, "p_nom_extendable"] = False
        n.generators.loc[gas_i, "p_nom"] = p_nom

        # add candidates for new gas pipelines to achieve full connectivity

        G = nx.Graph()

        gas_buses = n.buses.loc[n.buses.carrier == "gas", "location"]
        G.add_nodes_from(np.unique(gas_buses.values))

        sel = gas_pipes.p_nom > 1500
        attrs = ["bus0", "bus1", "length"]
        G.add_weighted_edges_from(gas_pipes.loc[sel, attrs].values)

        # find all complement edges
        complement_edges = pd.DataFrame(complement(G).edges, columns=["bus0", "bus1"])
        complement_edges["length"] = complement_edges.apply(haversine, axis=1)

        # apply k_edge_augmentation weighted by length of complement edges
        k_edge = options.get("gas_network_connectivity_upgrade", 3)
        augmentation = list(
            k_edge_augmentation(G, k_edge, avail=complement_edges.values)
        )

        if augmentation:
            new_gas_pipes = pd.DataFrame(augmentation, columns=["bus0", "bus1"])
            new_gas_pipes["length"] = new_gas_pipes.apply(haversine, axis=1)

            new_gas_pipes.index = new_gas_pipes.apply(
                lambda x: f"gas pipeline new {x.bus0} <-> {x.bus1}", axis=1
            )

            n.madd(
                "Link",
                new_gas_pipes.index,
                bus0=new_gas_pipes.bus0 + " gas",
                bus1=new_gas_pipes.bus1 + " gas",
                p_min_pu=-1,  # new gas pipes are bidirectional
                p_nom_extendable=True,
                length=new_gas_pipes.length,
                capital_cost=new_gas_pipes.length
                * costs.at["CH4 (g) pipeline", "fixed"],
                carrier="gas pipeline new",
                lifetime=costs.at["CH4 (g) pipeline", "lifetime"],
            )

    if options["H2_retrofit"]:
        logger.info("Add retrofitting options of existing CH4 pipes to H2 pipes.")

        fr = "gas pipeline"
        to = "H2 pipeline retrofitted"
        h2_pipes = gas_pipes.rename(index=lambda x: x.replace(fr, to))

        n.madd(
            "Link",
            h2_pipes.index,
            bus0=h2_pipes.bus0 + " H2",
            bus1=h2_pipes.bus1 + " H2",
            p_min_pu=-1.0,  # allow that all H2 retrofit pipelines can be used in both directions
            p_nom_max=h2_pipes.p_nom * options["H2_retrofit_capacity_per_CH4"],
            p_nom_extendable=True,
            length=h2_pipes.length,
            capital_cost=costs.at["H2 (g) pipeline repurposed", "fixed"]
            * h2_pipes.length,
            tags=h2_pipes.name,
            carrier="H2 pipeline retrofitted",
            lifetime=costs.at["H2 (g) pipeline repurposed", "lifetime"],
        )

    if options.get("H2_network", True):
        logger.info("Add options for new hydrogen pipelines.")

        h2_pipes = create_network_topology(
            n, "H2 pipeline ", carriers=["DC", "gas pipeline"]
        )

        # TODO Add efficiency losses
        n.madd(
            "Link",
            h2_pipes.index,
            bus0=h2_pipes.bus0.values + " H2",
            bus1=h2_pipes.bus1.values + " H2",
            p_min_pu=-1,
            p_nom_extendable=True,
            length=h2_pipes.length.values,
            capital_cost=costs.at["H2 (g) pipeline", "fixed"] * h2_pipes.length.values,
            carrier="H2 pipeline",
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
        )

    n.add("Carrier", "battery")

    n.madd("Bus", nodes + " battery", location=nodes, carrier="battery", unit="MWh_el")

    n.madd(
        "Store",
        nodes + " battery",
        bus=nodes + " battery",
        e_cyclic=True,
        e_nom_extendable=True,
        carrier="battery",
        capital_cost=costs.at["battery storage", "fixed"],
        lifetime=costs.at["battery storage", "lifetime"],
    )

    n.madd(
        "Link",
        nodes + " battery charger",
        bus0=nodes,
        bus1=nodes + " battery",
        carrier="battery charger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        capital_cost=costs.at["battery inverter", "fixed"],
        p_nom_extendable=True,
        lifetime=costs.at["battery inverter", "lifetime"],
    )

    n.madd(
        "Link",
        nodes + " battery discharger",
        bus0=nodes + " battery",
        bus1=nodes,
        carrier="battery discharger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        marginal_cost=options["marginal_cost_storage"],
        p_nom_extendable=True,
        lifetime=costs.at["battery inverter", "lifetime"],
    )

    if options["methanation"]:
        n.madd(
            "Link",
            spatial.nodes,
            suffix=" Sabatier",
            bus0=nodes + " H2",
            bus1=spatial.gas.nodes,
            bus2=spatial.co2.nodes,
            p_nom_extendable=True,
            carrier="Sabatier",
            efficiency=costs.at["methanation", "efficiency"],
            efficiency2=-costs.at["methanation", "efficiency"]
            * costs.at["gas", "CO2 intensity"],
            capital_cost=costs.at["methanation", "fixed"]
            * costs.at["methanation", "efficiency"],  # costs given per kW_gas
            lifetime=costs.at["methanation", "lifetime"],
        )

    if options["helmeth"]:
        n.madd(
            "Link",
            spatial.nodes,
            suffix=" helmeth",
            bus0=nodes,
            bus1=spatial.gas.nodes,
            bus2=spatial.co2.nodes,
            carrier="helmeth",
            p_nom_extendable=True,
            efficiency=costs.at["helmeth", "efficiency"],
            efficiency2=-costs.at["helmeth", "efficiency"]
            * costs.at["gas", "CO2 intensity"],
            capital_cost=costs.at["helmeth", "fixed"],
            lifetime=costs.at["helmeth", "lifetime"],
        )

    if options.get("coal_cc"):
        n.madd(
            "Link",
            spatial.nodes,
            suffix=" coal CC",
            bus0=spatial.coal.nodes,
            bus1=spatial.nodes,
            bus2="co2 atmosphere",
            bus3=spatial.co2.nodes,
            marginal_cost=costs.at["coal", "efficiency"]
            * costs.at["coal", "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at["coal", "efficiency"] * costs.at["coal", "fixed"]
            + costs.at["biomass CHP capture", "fixed"]
            * costs.at["coal", "CO2 intensity"],  # NB: fixed cost is per MWel
            p_nom_extendable=True,
            carrier="coal",
            efficiency=costs.at["coal", "efficiency"],
            efficiency2=costs.at["coal", "CO2 intensity"]
            * (1 - costs.at["biomass CHP capture", "capture_rate"]),
            efficiency3=costs.at["coal", "CO2 intensity"]
            * costs.at["biomass CHP capture", "capture_rate"],
            lifetime=costs.at["coal", "lifetime"],
        )

    if options["SMR"]:
        n.madd(
            "Link",
            spatial.nodes,
            suffix=" SMR CC",
            bus0=spatial.gas.nodes,
            bus1=nodes + " H2",
            bus2="co2 atmosphere",
            bus3=spatial.co2.nodes,
            p_nom_extendable=True,
            carrier="SMR CC",
            efficiency=costs.at["SMR CC", "efficiency"],
            efficiency2=costs.at["gas", "CO2 intensity"] * (1 - options["cc_fraction"]),
            efficiency3=costs.at["gas", "CO2 intensity"] * options["cc_fraction"],
            capital_cost=costs.at["SMR CC", "fixed"],
            lifetime=costs.at["SMR CC", "lifetime"],
        )

        n.madd(
            "Link",
            nodes + " SMR",
            bus0=spatial.gas.nodes,
            bus1=nodes + " H2",
            bus2="co2 atmosphere",
            p_nom_extendable=True,
            carrier="SMR",
            efficiency=costs.at["SMR", "efficiency"],
            efficiency2=costs.at["gas", "CO2 intensity"],
            capital_cost=costs.at["SMR", "fixed"],
            lifetime=costs.at["SMR", "lifetime"],
        )


def add_land_transport(n, costs):
    # TODO options?

    logger.info("Add land transport")
    nhours = n.snapshot_weightings.generators.sum()

    transport = pd.read_csv(
        snakemake.input.transport_demand, index_col=0, parse_dates=True
    )
    number_cars = pd.read_csv(snakemake.input.transport_data, index_col=0)[
        "number cars"
    ]
    avail_profile = pd.read_csv(
        snakemake.input.avail_profile, index_col=0, parse_dates=True
    )
    dsm_profile = pd.read_csv(
        snakemake.input.dsm_profile, index_col=0, parse_dates=True
    )
    # TODO: maybe adjust shares @Daniel
    fuel_cell_share = get(options["land_transport_fuel_cell_share"], investment_year)
    electric_share = get(options["land_transport_electric_share"], investment_year)
    oil_share = get(options["land_transport_oil_share"], investment_year)
    gas_share = get(options["land_transport_gas_share"], investment_year)

    # get CLEVER scenario transport demand data for land transport and given investment_year
    # extracted from dictionary with all clever data
    clever_df_ltr = pd.DataFrame(
        {subsec: values[str(investment_year)]
         for subsec, values in clever_dict["transport"].items()
         if "land" in subsec
         }
    )

    # total land transport demand for modelled countries
    countries = snakemake.config["countries"]
    clever_ltr_total = clever_df_ltr.loc[countries, :].sum(axis=1)

    # Getting different shares according to CLEVER data
    # BEV share
    electric_share = clever_df_ltr.loc[countries, "land_ev"].sum() / clever_ltr_total.sum()
    # FCEV share
    fuel_cell_share = clever_df_ltr.loc[countries, "land_h2"].sum() / clever_ltr_total.sum()
    # ICEV share
    oil_share = clever_df_ltr.loc[countries, "land_liquid_fuels"].sum() / clever_ltr_total.sum()
    # gas share
    gas_share = clever_df_ltr.loc[countries, "land_gas"].sum() / clever_ltr_total.sum()

    total_share = fuel_cell_share + electric_share + oil_share + gas_share
    if total_share != 1:
        logger.warning(
            f"Total land transport shares sum up to {total_share:.2%}, corresponding to increased or decreased demand assumptions."
        )

    logger.info(f"FCEV share: {fuel_cell_share:.2%}")
    logger.info(f"EV share: {electric_share:.2%}")
    logger.info(f"ICEV Oil share: {oil_share:.2%}")
    logger.info(f"ICEV Gas share: {gas_share:.2%}")

    nodes = pop_layout.index

    if electric_share > 0:
        n.add("Carrier", "Li ion")

        n.madd(
            "Bus",
            nodes,
            location=nodes,
            suffix=" EV battery",
            carrier="Li ion",
            unit="MWh_el",
        )

        # get pypsa (pes) land transport data
        # TODO: check why running mean of 3h
        pes_ltr = (transport[nodes] + cycling_shift(transport[nodes], 1)
                              + cycling_shift(transport[nodes], 2)) / 3

        p_set_ltr_ele = scale_ltr_to_sce_demand(clever_df_ltr, pes_ltr, "BEV")

        n.madd(
            "Load",
            nodes,
            suffix=" land transport EV",
            bus=nodes + " EV battery",
            carrier="land transport EV",
            p_set=p_set_ltr_ele,
        )

        # regionally distribute electric share
        pes_ltr_ele_reg = pes_ltr.sum().to_frame() / 1e6
        pes_ltr_ele_reg['ctry'] = pes_ltr_ele_reg.index.str[:2]
        clever_ltr_ele_nat = clever_df_ltr["land_ev"]
        electric_share_reg = pes_ltr_ele_reg.ctry.map(clever_ltr_ele_nat.div(clever_ltr_total))
        p_nom = number_cars.mul(electric_share_reg) * options.get("bev_charge_rate", 0.011)

        n.madd(
            "Link",
            nodes,
            suffix=" BEV charger",
            bus0=nodes,
            bus1=nodes + " EV battery",
            p_nom=p_nom,
            carrier="BEV charger",
            p_max_pu=avail_profile[nodes],
            efficiency=options.get("bev_charge_efficiency", 0.9),
            # These were set non-zero to find LU infeasibility when availability = 0.25
            # p_nom_extendable=True,
            # p_nom_min=p_nom,
            # capital_cost=1e6,  #i.e. so high it only gets built where necessary
        )

    if electric_share > 0 and options["v2g"]:
        n.madd(
            "Link",
            nodes,
            suffix=" V2G",
            bus1=nodes,
            bus0=nodes + " EV battery",
            p_nom=p_nom,
            carrier="V2G",
            p_max_pu=avail_profile[nodes],
            efficiency=options.get("bev_charge_efficiency", 0.9),
        )

    if electric_share > 0 and options["bev_dsm"]:
        e_nom = (
            number_cars.mul(electric_share_reg)
            * options.get("bev_energy", 0.05)
            * options["bev_availability"]
        )

        n.madd(
            "Store",
            nodes,
            suffix=" battery storage",
            bus=nodes + " EV battery",
            carrier="battery storage",
            e_cyclic=True,
            e_nom=e_nom,
            e_max_pu=1,
            e_min_pu=dsm_profile[nodes],
        )

    if fuel_cell_share > 0:

        # get pypsa (pes) land transport data
        pes_ltr = transport[nodes]

        # substitute land FCEV transport data with CLEVER demand data
        p_set_ltr_h2 = scale_ltr_to_sce_demand(clever_df_ltr, pes_ltr, "FCEV")

        n.madd(
            "Load",
            nodes,
            suffix=" land transport fuel cell",
            bus=nodes + " H2",
            carrier="land transport fuel cell",
            p_set=p_set_ltr_h2,
        )

    if oil_share > 0:
        if "oil" not in n.buses.carrier.unique():
            n.madd(
                "Bus",
                spatial.oil.nodes,
                location=spatial.oil.locations,
                carrier="oil",
                unit="MWh_LHV",
            )

        # get pypsa (pes) land transport data
        pes_ltr = transport[nodes]

        # substitute land ICEV transport data with CLEVER demand data
        p_set_ltr_liq = scale_ltr_to_sce_demand(clever_df_ltr, pes_ltr, "ICEV_oil")

        n.madd(
            "Load",
            nodes,
            suffix=" land transport oil",
            bus=spatial.oil.nodes,
            carrier="land transport oil",
            p_set=p_set_ltr_liq,
        )

        co2 = (
            p_set_ltr_liq.sum().sum()
            / nhours
            * costs.at["oil", "CO2 intensity"]
        )

        n.add(
            "Load",
            "land transport oil emissions",
            bus="co2 atmosphere",
            carrier="land transport oil emissions",
            p_set=-co2,
        )

    if gas_share > 0:
        if "gas" not in n.buses.carrier.unique():
            n.madd(
                "Bus",
                spatial.gas.nodes,
                location=spatial.gas.locations,
                carrier="gas",
                unit="MWh_LHV",
            )

        # get pypsa (pes) land transport data
        pes_ltr = transport[nodes]

        # substitute land ICEV transport data with CLEVER demand data
        p_set_ltr_gas = scale_ltr_to_sce_demand(clever_df_ltr, pes_ltr, "ICEV_gas")

        n.madd(
            "Load",
            nodes,
            suffix=" land transport gas",
            bus=spatial.gas.nodes,
            carrier="land transport gas",
            p_set=p_set_ltr_gas,
        )

        co2 = (
                p_set_ltr_gas.sum().sum()
                / nhours
                * costs.at["gas", "CO2 intensity"]
        )

        n.add(
            "Load",
            "land transport gas emissions",
            bus="co2 atmosphere",
            carrier="land transport gas emissions",
            p_set=-co2,
        )

def scale_ltr_to_sce_demand(clever_df_ltr, pes_ltr, carrier):
        """
        Scale the demand of land transport for a given carrier to be the equivalent demand of the scenario
        @type carrier: str
        Carrier can either be 'BEV', 'FCEV', 'ICEV_oil' or 'ICEV_gas'.
        """
        # get regional distributed pypsa (pes) demand
        pes_ltr_reg = pes_ltr.sum().to_frame() / 1e6

        # get clever nationally distributed CLEVER demand depending on what carrier we want
        if carrier == 'BEV':
            clever_ltr_nat = clever_df_ltr["land_ev"]
        elif carrier == 'FCEV':
            clever_ltr_nat = clever_df_ltr["land_h2"]
        elif carrier == 'ICEV_oil':
            clever_ltr_nat = clever_df_ltr["land_liquid_fuels"]
        elif carrier == 'ICEV_gas':
            clever_ltr_nat = clever_df_ltr["land_gas"]
        else:
            raise Exception("Invalid carrier information. Please enter either 'BEV', \
            'FCEV', 'ICEV_oil' or 'ICEV_gas'.")

        # distribute CLEVER demand regionally analogue to pypsa (pes) regional distribution
        clever_ltr_reg = distribute_sce_demand_by_pes_layout(clever_ltr_nat, pes_ltr_reg, pop_layout)
        # get regional scale factor from resulting regionally distributed demand that can be applied to time series data
        scale_factor = clever_ltr_reg.div(pes_ltr_reg[0])

        # update BEV/FCEV/ICEV demand time series to CLEVER data using regional scale factor
        p_set_ltr = pes_ltr.mul(scale_factor, axis=1)

        return p_set_ltr


def build_heat_demand(n):
    # copy forward the daily average heat demand into each hour, so it can be multiplied by the intraday profile
    daily_space_heat_demand = (
        xr.open_dataarray(snakemake.input.heat_demand_total)
        .to_pandas()
        .reindex(index=n.snapshots, method="ffill")
    )

    intraday_profiles = pd.read_csv(snakemake.input.heat_profile, index_col=0)

    sectors = ["residential", "services"]
    uses = ["water", "space"]

    heat_demand = {}
    electric_heat_supply = {}
    for sector, use in product(sectors, uses):
        weekday = list(intraday_profiles[f"{sector} {use} weekday"])
        weekend = list(intraday_profiles[f"{sector} {use} weekend"])
        weekly_profile = weekday * 5 + weekend * 2
        intraday_year_profile = generate_periodic_profiles(
            daily_space_heat_demand.index.tz_localize("UTC"),
            nodes=daily_space_heat_demand.columns,
            weekly_profile=weekly_profile,
        )

        if use == "space":
            heat_demand_shape = daily_space_heat_demand * intraday_year_profile
        else:
            heat_demand_shape = intraday_year_profile

        heat_demand[f"{sector} {use}"] = (
            heat_demand_shape / heat_demand_shape.sum()
        ).multiply(pop_weighted_energy_totals[f"total {sector} {use}"]) * 1e6
        electric_heat_supply[f"{sector} {use}"] = (
            heat_demand_shape / heat_demand_shape.sum()
        ).multiply(pop_weighted_energy_totals[f"electricity {sector} {use}"]) * 1e6

    heat_demand = pd.concat(heat_demand, axis=1)
    electric_heat_supply = pd.concat(electric_heat_supply, axis=1)

    # subtract from electricity load since heat demand already in heat_demand
    electric_nodes = n.loads.index[n.loads.carrier == "electricity"]
    n.loads_t.p_set[electric_nodes] = (
        n.loads_t.p_set[electric_nodes]
        - electric_heat_supply.groupby(level=1, axis=1).sum()[electric_nodes]
    )

    return heat_demand


def add_heat(n, costs):
    logger.info("Add heat sector")

    sectors = ["residential", "services"]
    uses = ["water", "space"]

    heat_demand = build_heat_demand(n)

    nodes, dist_fraction, urban_fraction = create_nodes_for_heat_sector()

    # NB: must add costs of central heating afterwards (EUR 400 / kWpeak, 50a, 1% FOM from Fraunhofer ISE)

    # scale heat demand to CLEVER scenario demand
    for sector in sectors:

        # CLEVER services demand is not available as separate water and space heating
        # difference between total energy demand and electricity demand in services buildings is used
        if sector == "services":
            # get regional distributed pes demand and nationally distributed CLEVER demand for both uses summed up
            pes_heat_serv_reg = heat_demand[[f"{sector} space",
                                             f"{sector} water"]].sum().unstack().sum().to_frame() / 1e6
            clever_heat_serv_nat = clever_dict[sector]["total"][str(investment_year)] - \
                                   clever_dict[sector]["electricity"][str(investment_year)]

            # scale and distribute according to total services heat demands
            clever_heat_serv_reg = distribute_sce_demand_by_pes_layout(clever_heat_serv_nat,
                                                                       pes_heat_serv_reg, pop_layout)
            scale_factor = clever_heat_serv_reg / pes_heat_serv_reg[0]

            # update both water and space heat accordingly to CLEVER demand using regional scale factor
            # this also keeps distribution of demands between uses as in pes
            heat_demand[sector + " water"] *= scale_factor
            heat_demand[sector + " space"] *= scale_factor
            # print(f"heat scale factors for sector {sector} water : \n {scale_factor}")
            # print(f"heat scale factors for sector {sector} space : \n {scale_factor}")

        elif sector == "residential":
            for use in uses:

                # get regional distributed pes demand and nationally distributed CLEVER demand per sector
                pes_heat_resid_reg = heat_demand[[f"{sector} {use}"]].sum().unstack().T / 1e6
                sce_heat_resid_nat = clever_dict[sector][f"{use}_heating"][str(investment_year)]

                # scale and distribute
                sce_heat_resid_reg = distribute_sce_demand_by_pes_layout(sce_heat_resid_nat,
                                                                         pes_heat_resid_reg, pop_layout)
                scale_factor = sce_heat_resid_reg.div(pes_heat_resid_reg[f"{sector} {use}"])

                # update heat demand to CLEVER data using regional scale factor
                heat_demand[f"{sector} {use}"] *= scale_factor
                print(f"heat scale factors for sector {sector} {use} : \n {scale_factor}")

    # exogenously reduce space heat demand
    if options["reduce_space_heat_exogenously"]:
        dE = get(options["reduce_space_heat_exogenously_factor"], investment_year)
        logger.info(f"Assumed space heat reduction of {dE:.2%}")
        for sector in sectors:
            heat_demand[sector + " space"] = (1 - dE) * heat_demand[sector + " space"]

    heat_systems = [
        "residential rural",
        "services rural",
        "residential urban decentral",
        "services urban decentral",
        "urban central",
    ]

    cop = {
        "air": xr.open_dataarray(snakemake.input.cop_air_total)
        .to_pandas()
        .reindex(index=n.snapshots),
        "ground": xr.open_dataarray(snakemake.input.cop_soil_total)
        .to_pandas()
        .reindex(index=n.snapshots),
    }

    if options["solar_thermal"]:
        solar_thermal = (
            xr.open_dataarray(snakemake.input.solar_thermal_total)
            .to_pandas()
            .reindex(index=n.snapshots)
        )
        # 1e3 converts from W/m^2 to MW/(1000m^2) = kW/m^2
        solar_thermal = options["solar_cf_correction"] * solar_thermal / 1e3

    for name in heat_systems:
        name_type = "central" if name == "urban central" else "decentral"

        n.add("Carrier", name + " heat")

        n.madd(
            "Bus",
            nodes[name] + f" {name} heat",
            location=nodes[name],
            carrier=name + " heat",
            unit="MWh_th",
        )

        ## Add heat load

        for sector in sectors:
            # heat demand weighting
            if "rural" in name:
                factor = 1 - urban_fraction[nodes[name]]
            elif "urban central" in name:
                factor = dist_fraction[nodes[name]]
            elif "urban decentral" in name:
                factor = urban_fraction[nodes[name]] - dist_fraction[nodes[name]]
            else:
                raise NotImplementedError(
                    f" {name} not in " f"heat systems: {heat_systems}"
                )

            if sector in name:
                heat_load = (
                    heat_demand[[sector + " water", sector + " space"]]
                    .groupby(level=1, axis=1)
                    .sum()[nodes[name]]
                    .multiply(factor)
                )

        if name == "urban central":
            heat_load = (
                heat_demand.groupby(level=1, axis=1)
                .sum()[nodes[name]]
                .multiply(
                    factor * (1 + options["district_heating"]["district_heating_loss"])
                )
            )

        n.madd(
            "Load",
            nodes[name],
            suffix=f" {name} heat",
            bus=nodes[name] + f" {name} heat",
            carrier=name + " heat",
            p_set=heat_load,
        )

        ## Add heat pumps

        heat_pump_type = "air" if "urban" in name else "ground"

        costs_name = f"{name_type} {heat_pump_type}-sourced heat pump"
        efficiency = (
            cop[heat_pump_type][nodes[name]]
            if options["time_dep_hp_cop"]
            else costs.at[costs_name, "efficiency"]
        )

        n.madd(
            "Link",
            nodes[name],
            suffix=f" {name} {heat_pump_type} heat pump",
            bus0=nodes[name],
            bus1=nodes[name] + f" {name} heat",
            carrier=f"{name} {heat_pump_type} heat pump",
            efficiency=efficiency,
            capital_cost=costs.at[costs_name, "efficiency"]
            * costs.at[costs_name, "fixed"],
            p_nom_extendable=True,
            lifetime=costs.at[costs_name, "lifetime"],
        )

        if options["tes"]:
            n.add("Carrier", name + " water tanks")

            n.madd(
                "Bus",
                nodes[name] + f" {name} water tanks",
                location=nodes[name],
                carrier=name + " water tanks",
                unit="MWh_th",
            )

            n.madd(
                "Link",
                nodes[name] + f" {name} water tanks charger",
                bus0=nodes[name] + f" {name} heat",
                bus1=nodes[name] + f" {name} water tanks",
                efficiency=costs.at["water tank charger", "efficiency"],
                carrier=name + " water tanks charger",
                p_nom_extendable=True,
            )

            n.madd(
                "Link",
                nodes[name] + f" {name} water tanks discharger",
                bus0=nodes[name] + f" {name} water tanks",
                bus1=nodes[name] + f" {name} heat",
                carrier=name + " water tanks discharger",
                efficiency=costs.at["water tank discharger", "efficiency"],
                p_nom_extendable=True,
            )

            if isinstance(options["tes_tau"], dict):
                tes_time_constant_days = options["tes_tau"][name_type]
            else:
                logger.warning(
                    "Deprecated: a future version will require you to specify 'tes_tau' ",
                    "for 'decentral' and 'central' separately.",
                )
                tes_time_constant_days = (
                    options["tes_tau"] if name_type == "decentral" else 180.0
                )

            n.madd(
                "Store",
                nodes[name] + f" {name} water tanks",
                bus=nodes[name] + f" {name} water tanks",
                e_cyclic=True,
                e_nom_extendable=True,
                carrier=name + " water tanks",
                standing_loss=1 - np.exp(-1 / 24 / tes_time_constant_days),
                capital_cost=costs.at[name_type + " water tank storage", "fixed"],
                lifetime=costs.at[name_type + " water tank storage", "lifetime"],
            )

        if options["boilers"]:
            key = f"{name_type} resistive heater"

            n.madd(
                "Link",
                nodes[name] + f" {name} resistive heater",
                bus0=nodes[name],
                bus1=nodes[name] + f" {name} heat",
                carrier=name + " resistive heater",
                efficiency=costs.at[key, "efficiency"],
                capital_cost=costs.at[key, "efficiency"] * costs.at[key, "fixed"],
                p_nom_extendable=True,
                lifetime=costs.at[key, "lifetime"],
            )

            key = f"{name_type} gas boiler"

            n.madd(
                "Link",
                nodes[name] + f" {name} gas boiler",
                p_nom_extendable=True,
                bus0=spatial.gas.df.loc[nodes[name], "nodes"].values,
                bus1=nodes[name] + f" {name} heat",
                bus2="co2 atmosphere",
                carrier=name + " gas boiler",
                efficiency=costs.at[key, "efficiency"],
                efficiency2=costs.at["gas", "CO2 intensity"],
                capital_cost=costs.at[key, "efficiency"] * costs.at[key, "fixed"],
                lifetime=costs.at[key, "lifetime"],
            )

        if options["solar_thermal"]:
            n.add("Carrier", name + " solar thermal")

            n.madd(
                "Generator",
                nodes[name],
                suffix=f" {name} solar thermal collector",
                bus=nodes[name] + f" {name} heat",
                carrier=name + " solar thermal",
                p_nom_extendable=True,
                capital_cost=costs.at[name_type + " solar thermal", "fixed"],
                p_max_pu=solar_thermal[nodes[name]],
                lifetime=costs.at[name_type + " solar thermal", "lifetime"],
            )

        if options["chp"] and name == "urban central":
            # add gas CHP; biomass CHP is added in biomass section
            n.madd(
                "Link",
                nodes[name] + " urban central gas CHP",
                bus0=spatial.gas.df.loc[nodes[name], "nodes"].values,
                bus1=nodes[name],
                bus2=nodes[name] + " urban central heat",
                bus3="co2 atmosphere",
                carrier="urban central gas CHP",
                p_nom_extendable=True,
                capital_cost=costs.at["central gas CHP", "fixed"]
                * costs.at["central gas CHP", "efficiency"],
                marginal_cost=costs.at["central gas CHP", "VOM"],
                efficiency=costs.at["central gas CHP", "efficiency"],
                efficiency2=costs.at["central gas CHP", "efficiency"]
                / costs.at["central gas CHP", "c_b"],
                efficiency3=costs.at["gas", "CO2 intensity"],
                lifetime=costs.at["central gas CHP", "lifetime"],
            )

            n.madd(
                "Link",
                nodes[name] + " urban central gas CHP CC",
                bus0=spatial.gas.df.loc[nodes[name], "nodes"].values,
                bus1=nodes[name],
                bus2=nodes[name] + " urban central heat",
                bus3="co2 atmosphere",
                bus4=spatial.co2.df.loc[nodes[name], "nodes"].values,
                carrier="urban central gas CHP CC",
                p_nom_extendable=True,
                capital_cost=costs.at["central gas CHP", "fixed"]
                * costs.at["central gas CHP", "efficiency"]
                + costs.at["biomass CHP capture", "fixed"]
                * costs.at["gas", "CO2 intensity"],
                marginal_cost=costs.at["central gas CHP", "VOM"],
                efficiency=costs.at["central gas CHP", "efficiency"]
                - costs.at["gas", "CO2 intensity"]
                * (
                    costs.at["biomass CHP capture", "electricity-input"]
                    + costs.at["biomass CHP capture", "compression-electricity-input"]
                ),
                efficiency2=costs.at["central gas CHP", "efficiency"]
                / costs.at["central gas CHP", "c_b"]
                + costs.at["gas", "CO2 intensity"]
                * (
                    costs.at["biomass CHP capture", "heat-output"]
                    + costs.at["biomass CHP capture", "compression-heat-output"]
                    - costs.at["biomass CHP capture", "heat-input"]
                ),
                efficiency3=costs.at["gas", "CO2 intensity"]
                * (1 - costs.at["biomass CHP capture", "capture_rate"]),
                efficiency4=costs.at["gas", "CO2 intensity"]
                * costs.at["biomass CHP capture", "capture_rate"],
                lifetime=costs.at["central gas CHP", "lifetime"],
            )

        if options["chp"] and options["micro_chp"] and name != "urban central":
            n.madd(
                "Link",
                nodes[name] + f" {name} micro gas CHP",
                p_nom_extendable=True,
                bus0=spatial.gas.df.loc[nodes[name], "nodes"].values,
                bus1=nodes[name],
                bus2=nodes[name] + f" {name} heat",
                bus3="co2 atmosphere",
                carrier=name + " micro gas CHP",
                efficiency=costs.at["micro CHP", "efficiency"],
                efficiency2=costs.at["micro CHP", "efficiency-heat"],
                efficiency3=costs.at["gas", "CO2 intensity"],
                capital_cost=costs.at["micro CHP", "fixed"],
                lifetime=costs.at["micro CHP", "lifetime"],
            )

    if options["retrofitting"]["retro_endogen"]:
        logger.info("Add retrofitting endogenously")

        # resample heat demand temporal 'heat_demand_r' depending on in config
        # specified temporal resolution, to not overestimate retrofitting
        hours = list(filter(re.compile(r"^\d+h$", re.IGNORECASE).search, opts))
        if len(hours) == 0:
            hours = [n.snapshots[1] - n.snapshots[0]]
        heat_demand_r = heat_demand.resample(hours[0]).mean()

        # retrofitting data 'retro_data' with 'costs' [EUR/m^2] and heat
        # demand 'dE' [per unit of original heat demand] for each country and
        # different retrofitting strengths [additional insulation thickness in m]
        retro_data = pd.read_csv(
            snakemake.input.retro_cost_energy,
            index_col=[0, 1],
            skipinitialspace=True,
            header=[0, 1],
        )
        # heated floor area [10^6 * m^2] per country
        floor_area = pd.read_csv(snakemake.input.floor_area, index_col=[0, 1])

        n.add("Carrier", "retrofitting")

        # share of space heat demand 'w_space' of total heat demand
        w_space = {}
        for sector in sectors:
            w_space[sector] = heat_demand_r[sector + " space"] / (
                heat_demand_r[sector + " space"] + heat_demand_r[sector + " water"]
            )
        w_space["tot"] = (
            heat_demand_r["services space"] + heat_demand_r["residential space"]
        ) / heat_demand_r.groupby(level=[1], axis=1).sum()

        for name in n.loads[
            n.loads.carrier.isin([x + " heat" for x in heat_systems])
        ].index:
            node = n.buses.loc[name, "location"]
            ct = pop_layout.loc[node, "ct"]

            # weighting 'f' depending on the size of the population at the node
            f = urban_fraction[node] if "urban" in name else (1 - urban_fraction[node])
            if f == 0:
                continue
            # get sector name ("residential"/"services"/or both "tot" for urban central)
            sec = [x if x in name else "tot" for x in sectors][0]

            # get floor aread at node and region (urban/rural) in m^2
            floor_area_node = (
                pop_layout.loc[node].fraction * floor_area.loc[ct, "value"] * 10**6
            ).loc[sec] * f
            # total heat demand at node [MWh]
            demand = n.loads_t.p_set[name].resample(hours[0]).mean()

            # space heat demand at node [MWh]
            space_heat_demand = demand * w_space[sec][node]
            # normed time profile of space heat demand 'space_pu' (values between 0-1),
            # p_max_pu/p_min_pu of retrofitting generators
            space_pu = (space_heat_demand / space_heat_demand.max()).to_frame(name=node)

            # minimum heat demand 'dE' after retrofitting in units of original heat demand (values between 0-1)
            dE = retro_data.loc[(ct, sec), ("dE")]
            # get additional energy savings 'dE_diff' between the different retrofitting strengths/generators at one node
            dE_diff = abs(dE.diff()).fillna(1 - dE.iloc[0])
            # convert costs Euro/m^2 -> Euro/MWh
            capital_cost = (
                retro_data.loc[(ct, sec), ("cost")]
                * floor_area_node
                / ((1 - dE) * space_heat_demand.max())
            )
            # number of possible retrofitting measures 'strengths' (set in list at config.yaml 'l_strength')
            # given in additional insulation thickness [m]
            # for each measure, a retrofitting generator is added at the node
            strengths = retro_data.columns.levels[1]

            # check that ambitious retrofitting has higher costs per MWh than moderate retrofitting
            if (capital_cost.diff() < 0).sum():
                logger.warning(f"Costs are not linear for {ct} {sec}")
                s = capital_cost[(capital_cost.diff() < 0)].index
                strengths = strengths.drop(s)

            # reindex normed time profile of space heat demand back to hourly resolution
            space_pu = space_pu.reindex(index=heat_demand.index).fillna(method="ffill")

            # add for each retrofitting strength a generator with heat generation profile following the profile of the heat demand
            for strength in strengths:
                n.madd(
                    "Generator",
                    [node],
                    suffix=" retrofitting " + strength + " " + name[6::],
                    bus=name,
                    carrier="retrofitting",
                    p_nom_extendable=True,
                    p_nom_max=dE_diff[strength]
                    * space_heat_demand.max(),  # maximum energy savings for this renovation strength
                    p_max_pu=space_pu,
                    p_min_pu=space_pu,
                    country=ct,
                    capital_cost=capital_cost[strength]
                    * options["retrofitting"]["cost_factor"],
                )


def create_nodes_for_heat_sector():
    # TODO pop_layout

    # rural are areas with low heating density and individual heating
    # urban are areas with high heating density
    # urban can be split into district heating (central) and individual heating (decentral)

    ct_urban = pop_layout.urban.groupby(pop_layout.ct).sum()
    # distribution of urban population within a country
    pop_layout["urban_ct_fraction"] = pop_layout.urban / pop_layout.ct.map(ct_urban.get)

    sectors = ["residential", "services"]

    nodes = {}
    urban_fraction = pop_layout.urban / pop_layout[["rural", "urban"]].sum(axis=1)

    for sector in sectors:
        nodes[sector + " rural"] = pop_layout.index
        nodes[sector + " urban decentral"] = pop_layout.index

    district_heat_share = pop_weighted_energy_totals["district heat share"]

    # maximum potential of urban demand covered by district heating
    central_fraction = options["district_heating"]["potential"]
    # district heating share at each node
    dist_fraction_node = (
        district_heat_share * pop_layout["urban_ct_fraction"] / pop_layout["fraction"]
    )
    nodes["urban central"] = dist_fraction_node.index
    # if district heating share larger than urban fraction -> set urban
    # fraction to district heating share
    urban_fraction = pd.concat([urban_fraction, dist_fraction_node], axis=1).max(axis=1)
    # difference of max potential and today's share of district heating
    diff = (urban_fraction * central_fraction) - dist_fraction_node
    progress = get(options["district_heating"]["progress"], investment_year)
    dist_fraction_node += diff * progress
    logger.info(
        f"Increase district heating share by a progress factor of {progress:.2%} "
        f"resulting in new average share of {dist_fraction_node.mean():.2%}"
    )

    return nodes, dist_fraction_node, urban_fraction


def add_biomass(n, costs):
    logger.info("Add biomass")

    biomass_potentials = pd.read_csv(snakemake.input.biomass_potentials, index_col=0)

    # need to aggregate potentials if gas not nodally resolved
    if options["gas_network"]:
        biogas_potentials_spatial = biomass_potentials["biogas"].rename(
            index=lambda x: x + " biogas"
        )
    else:
        biogas_potentials_spatial = biomass_potentials["biogas"].sum()

    if options.get("biomass_spatial", options["biomass_transport"]):
        solid_biomass_potentials_spatial = biomass_potentials["solid biomass"].rename(
            index=lambda x: x + " solid biomass"
        )
    else:
        solid_biomass_potentials_spatial = biomass_potentials["solid biomass"].sum()

    n.add("Carrier", "biogas")
    n.add("Carrier", "solid biomass")

    n.madd(
        "Bus",
        spatial.gas.biogas,
        location=spatial.gas.locations,
        carrier="biogas",
        unit="MWh_LHV",
    )

    n.madd(
        "Bus",
        spatial.biomass.nodes,
        location=spatial.biomass.locations,
        carrier="solid biomass",
        unit="MWh_LHV",
    )

    n.madd(
        "Store",
        spatial.gas.biogas,
        bus=spatial.gas.biogas,
        carrier="biogas",
        e_nom=biogas_potentials_spatial,
        marginal_cost=costs.at["biogas", "fuel"],
        e_initial=biogas_potentials_spatial,
    )

    n.madd(
        "Store",
        spatial.biomass.nodes,
        bus=spatial.biomass.nodes,
        carrier="solid biomass",
        e_nom=solid_biomass_potentials_spatial,
        marginal_cost=costs.at["solid biomass", "fuel"],
        e_initial=solid_biomass_potentials_spatial,
    )

    n.madd(
        "Link",
        spatial.gas.biogas_to_gas,
        bus0=spatial.gas.biogas,
        bus1=spatial.gas.nodes,
        bus2="co2 atmosphere",
        carrier="biogas to gas",
        capital_cost=costs.loc["biogas upgrading", "fixed"],
        marginal_cost=costs.loc["biogas upgrading", "VOM"],
        efficiency2=-costs.at["gas", "CO2 intensity"],
        p_nom_extendable=True,
    )

    if options["biomass_transport"]:
        transport_costs = pd.read_csv(
            snakemake.input.biomass_transport_costs,
            index_col=0,
        ).squeeze()

        # add biomass transport
        biomass_transport = create_network_topology(
            n, "biomass transport ", bidirectional=False
        )

        # costs
        bus0_costs = biomass_transport.bus0.apply(lambda x: transport_costs[x[:2]])
        bus1_costs = biomass_transport.bus1.apply(lambda x: transport_costs[x[:2]])
        biomass_transport["costs"] = pd.concat([bus0_costs, bus1_costs], axis=1).mean(
            axis=1
        )

        n.madd(
            "Link",
            biomass_transport.index,
            bus0=biomass_transport.bus0 + " solid biomass",
            bus1=biomass_transport.bus1 + " solid biomass",
            p_nom_extendable=False,
            p_nom=5e4,
            length=biomass_transport.length.values,
            marginal_cost=biomass_transport.costs * biomass_transport.length.values,
            carrier="solid biomass transport",
        )

    # AC buses with district heating
    urban_central = n.buses.index[n.buses.carrier == "urban central heat"]
    if not urban_central.empty and options["chp"]:
        urban_central = urban_central.str[: -len(" urban central heat")]

        key = "central solid biomass CHP"

        n.madd(
            "Link",
            urban_central + " urban central solid biomass CHP",
            bus0=spatial.biomass.df.loc[urban_central, "nodes"].values,
            bus1=urban_central,
            bus2=urban_central + " urban central heat",
            carrier="urban central solid biomass CHP",
            p_nom_extendable=True,
            capital_cost=costs.at[key, "fixed"] * costs.at[key, "efficiency"],
            marginal_cost=costs.at[key, "VOM"],
            efficiency=costs.at[key, "efficiency"],
            efficiency2=costs.at[key, "efficiency-heat"],
            lifetime=costs.at[key, "lifetime"],
        )

        n.madd(
            "Link",
            urban_central + " urban central solid biomass CHP CC",
            bus0=spatial.biomass.df.loc[urban_central, "nodes"].values,
            bus1=urban_central,
            bus2=urban_central + " urban central heat",
            bus3="co2 atmosphere",
            bus4=spatial.co2.df.loc[urban_central, "nodes"].values,
            carrier="urban central solid biomass CHP CC",
            p_nom_extendable=True,
            capital_cost=costs.at[key, "fixed"] * costs.at[key, "efficiency"]
            + costs.at["biomass CHP capture", "fixed"]
            * costs.at["solid biomass", "CO2 intensity"],
            marginal_cost=costs.at[key, "VOM"],
            efficiency=costs.at[key, "efficiency"]
            - costs.at["solid biomass", "CO2 intensity"]
            * (
                costs.at["biomass CHP capture", "electricity-input"]
                + costs.at["biomass CHP capture", "compression-electricity-input"]
            ),
            efficiency2=costs.at[key, "efficiency-heat"]
            + costs.at["solid biomass", "CO2 intensity"]
            * (
                costs.at["biomass CHP capture", "heat-output"]
                + costs.at["biomass CHP capture", "compression-heat-output"]
                - costs.at["biomass CHP capture", "heat-input"]
            ),
            efficiency3=-costs.at["solid biomass", "CO2 intensity"]
            * costs.at["biomass CHP capture", "capture_rate"],
            efficiency4=costs.at["solid biomass", "CO2 intensity"]
            * costs.at["biomass CHP capture", "capture_rate"],
            lifetime=costs.at[key, "lifetime"],
        )

    if options["biomass_boiler"]:
        # TODO: Add surcharge for pellets
        nodes_heat = create_nodes_for_heat_sector()[0]
        for name in [
            "residential rural",
            "services rural",
            "residential urban decentral",
            "services urban decentral",
        ]:
            n.madd(
                "Link",
                nodes_heat[name] + f" {name} biomass boiler",
                p_nom_extendable=True,
                bus0=spatial.biomass.df.loc[nodes_heat[name], "nodes"].values,
                bus1=nodes_heat[name] + f" {name} heat",
                carrier=name + " biomass boiler",
                efficiency=costs.at["biomass boiler", "efficiency"],
                capital_cost=costs.at["biomass boiler", "efficiency"]
                * costs.at["biomass boiler", "fixed"],
                lifetime=costs.at["biomass boiler", "lifetime"],
            )

    # Solid biomass to liquid fuel
    if options["biomass_to_liquid"]:
        n.madd(
            "Link",
            spatial.biomass.nodes,
            suffix=" biomass to liquid",
            bus0=spatial.biomass.nodes,
            bus1=spatial.oil.nodes,
            bus2="co2 atmosphere",
            carrier="biomass to liquid",
            lifetime=costs.at["BtL", "lifetime"],
            efficiency=costs.at["BtL", "efficiency"],
            efficiency2=-costs.at["solid biomass", "CO2 intensity"]
            + costs.at["BtL", "CO2 stored"],
            p_nom_extendable=True,
            capital_cost=costs.at["BtL", "fixed"],
            marginal_cost=costs.at["BtL", "efficiency"] * costs.loc["BtL", "VOM"],
        )

        # TODO: Update with energy penalty
        n.madd(
            "Link",
            spatial.biomass.nodes,
            suffix=" biomass to liquid CC",
            bus0=spatial.biomass.nodes,
            bus1=spatial.oil.nodes,
            bus2="co2 atmosphere",
            bus3=spatial.co2.nodes,
            carrier="biomass to liquid",
            lifetime=costs.at["BtL", "lifetime"],
            efficiency=costs.at["BtL", "efficiency"],
            efficiency2=-costs.at["solid biomass", "CO2 intensity"]
            + costs.at["BtL", "CO2 stored"] * (1 - costs.at["BtL", "capture rate"]),
            efficiency3=costs.at["BtL", "CO2 stored"] * costs.at["BtL", "capture rate"],
            p_nom_extendable=True,
            capital_cost=costs.at["BtL", "fixed"]
            + costs.at["biomass CHP capture", "fixed"] * costs.at["BtL", "CO2 stored"],
            marginal_cost=costs.at["BtL", "efficiency"] * costs.loc["BtL", "VOM"],
        )

    # BioSNG from solid biomass
    if options["biosng"]:
        n.madd(
            "Link",
            spatial.biomass.nodes,
            suffix=" solid biomass to gas",
            bus0=spatial.biomass.nodes,
            bus1=spatial.gas.nodes,
            bus3="co2 atmosphere",
            carrier="BioSNG",
            lifetime=costs.at["BioSNG", "lifetime"],
            efficiency=costs.at["BioSNG", "efficiency"],
            efficiency3=-costs.at["solid biomass", "CO2 intensity"]
            + costs.at["BioSNG", "CO2 stored"],
            p_nom_extendable=True,
            capital_cost=costs.at["BioSNG", "fixed"],
            marginal_cost=costs.at["BioSNG", "efficiency"] * costs.loc["BioSNG", "VOM"],
        )

        # TODO: Update with energy penalty for CC
        n.madd(
            "Link",
            spatial.biomass.nodes,
            suffix=" solid biomass to gas CC",
            bus0=spatial.biomass.nodes,
            bus1=spatial.gas.nodes,
            bus2=spatial.co2.nodes,
            bus3="co2 atmosphere",
            carrier="BioSNG",
            lifetime=costs.at["BioSNG", "lifetime"],
            efficiency=costs.at["BioSNG", "efficiency"],
            efficiency2=costs.at["BioSNG", "CO2 stored"]
            * costs.at["BioSNG", "capture rate"],
            efficiency3=-costs.at["solid biomass", "CO2 intensity"]
            + costs.at["BioSNG", "CO2 stored"]
            * (1 - costs.at["BioSNG", "capture rate"]),
            p_nom_extendable=True,
            capital_cost=costs.at["BioSNG", "fixed"]
            + costs.at["biomass CHP capture", "fixed"]
            * costs.at["BioSNG", "CO2 stored"],
            marginal_cost=costs.at["BioSNG", "efficiency"] * costs.loc["BioSNG", "VOM"],
        )


def add_industry(n, costs):
    logger.info("Add industrial demand")

    nodes = pop_layout.index
    nhours = n.snapshot_weightings.generators.sum()
    nyears = nhours / 8760

    # 1e6 to convert TWh to MWh
    industrial_demand = (
        pd.read_csv(snakemake.input.industrial_demand, index_col=0) * 1e6
    ) * nyears

    n.madd(
        "Bus",
        spatial.biomass.industry,
        location=spatial.biomass.locations,
        carrier="solid biomass for industry",
        unit="MWh_LHV",
    )

    # get CLEVER industrial biomass demand
    clever_bio_nat = clever_dict["industry"]["solid_biomass"][str(investment_year)]

    if options.get("biomass_spatial", options["biomass_transport"]):
        # distribute CLEVER scenario biomass demand
        pes_bio_reg = industrial_demand.loc[spatial.biomass.locations, "solid biomass"].to_frame() / 1e6
        clever_bio_reg = distribute_sce_demand_by_pes_layout(clever_bio_nat, pes_bio_reg, pop_layout)
        p_set_bio = clever_bio_reg.rename(index=lambda x: x + " solid biomass for industry") * 1e6 / nhours
    else:
        p_set_bio = clever_bio_nat.sum() * 1e6 / nhours

    n.madd(
        "Load",
        spatial.biomass.industry,
        bus=spatial.biomass.industry,
        carrier="solid biomass for industry",
        p_set=p_set_bio,
    )

    n.madd(
        "Link",
        spatial.biomass.industry,
        bus0=spatial.biomass.nodes,
        bus1=spatial.biomass.industry,
        carrier="solid biomass for industry",
        p_nom_extendable=True,
        efficiency=1.0,
    )

    n.madd(
        "Link",
        spatial.biomass.industry_cc,
        bus0=spatial.biomass.nodes,
        bus1=spatial.biomass.industry,
        bus2="co2 atmosphere",
        bus3=spatial.co2.nodes,
        carrier="solid biomass for industry CC",
        p_nom_extendable=True,
        capital_cost=costs.at["cement capture", "fixed"]
        * costs.at["solid biomass", "CO2 intensity"],
        efficiency=0.9,  # TODO: make config option
        efficiency2=-costs.at["solid biomass", "CO2 intensity"]
        * costs.at["cement capture", "capture_rate"],
        efficiency3=costs.at["solid biomass", "CO2 intensity"]
        * costs.at["cement capture", "capture_rate"],
        lifetime=costs.at["cement capture", "lifetime"],
    )

    n.madd(
        "Bus",
        spatial.gas.industry,
        location=spatial.gas.locations,
        carrier="gas for industry",
        unit="MWh_LHV",
    )

    # get scenario industrial gas demand
    clever_gas_nat = clever_dict["industry"]["gas"][str(investment_year)]

    if options["gas_network"]:
        # distribute CLEVER scenario gas demand
        pes_gas_nat = industrial_demand.loc[nodes, "methane"].to_frame() / 1e6
        clever_gas_reg = distribute_sce_demand_by_pes_layout(clever_gas_nat, pes_gas_nat, pop_layout)
        # CLEVER industry demand is in TWh, pypsa demand in MWh
        p_set_gas = clever_gas_reg.rename(index=lambda x: x + " gas for industry") * 1e6 / nhours
    else:
        p_set_gas = clever_gas_nat.sum() * 1e6 / nhours

    n.madd(
        "Load",
        spatial.gas.industry,
        bus=spatial.gas.industry,
        carrier="gas for industry",
        p_set=p_set_gas,
    )

    n.madd(
        "Link",
        spatial.gas.industry,
        bus0=spatial.gas.nodes,
        bus1=spatial.gas.industry,
        bus2="co2 atmosphere",
        carrier="gas for industry",
        p_nom_extendable=True,
        efficiency=1.0,
        efficiency2=costs.at["gas", "CO2 intensity"],
    )

    n.madd(
        "Link",
        spatial.gas.industry_cc,
        bus0=spatial.gas.nodes,
        bus1=spatial.gas.industry,
        bus2="co2 atmosphere",
        bus3=spatial.co2.nodes,
        carrier="gas for industry CC",
        p_nom_extendable=True,
        capital_cost=costs.at["cement capture", "fixed"]
        * costs.at["gas", "CO2 intensity"],
        efficiency=0.9,
        efficiency2=costs.at["gas", "CO2 intensity"]
        * (1 - costs.at["cement capture", "capture_rate"]),
        efficiency3=costs.at["gas", "CO2 intensity"]
        * costs.at["cement capture", "capture_rate"],
        lifetime=costs.at["cement capture", "lifetime"],
    )

    # get CLEVER scenario industrial hydrogen demand
    clever_h2_nat = clever_dict["industry"]["h2"][str(investment_year)]
    pes_h2_reg = industrial_demand.loc[nodes, "hydrogen"].to_frame() / 1e6

    # distribute CLEVER scenario industrial hydrogen demand
    clever_h2_reg = distribute_sce_demand_by_pes_layout(clever_h2_nat, pes_h2_reg, pop_layout)
    p_set_h2 = clever_h2_reg * 1e6 / nhours

    n.madd(
        "Load",
        nodes,
        suffix=" H2 for industry",
        bus=nodes + " H2",
        carrier="H2 for industry",
        p_set=p_set_h2,
    )

    shipping_hydrogen_share = get(options["shipping_hydrogen_share"], investment_year)
    shipping_methanol_share = get(options["shipping_methanol_share"], investment_year)
    shipping_oil_share = get(options["shipping_oil_share"], investment_year)
    shipping_gas_share = get(options["shipping_gas_share"], investment_year)

    # calculate shares for oil and gas

    total_share = shipping_hydrogen_share + shipping_methanol_share + shipping_oil_share + shipping_gas_share
    if total_share != 1:
        logger.warning(
            f"Total shipping shares sum up to {total_share:.2%}, corresponding to increased or decreased demand assumptions."
        )

    domestic_navigation = pop_weighted_energy_totals.loc[
        nodes, "total domestic navigation"
    ].squeeze()
    international_navigation = (
        pd.read_csv(snakemake.input.shipping_demand, index_col=0).squeeze() * nyears
    )
    all_navigation = domestic_navigation + international_navigation
    # p_set = all_navigation * 1e6 / nhours
    countries = snakemake.config["countries"]

    if shipping_hydrogen_share:
        oil_efficiency = options.get(
            "shipping_oil_efficiency", options.get("shipping_average_efficiency", 0.4)
        )
        efficiency = oil_efficiency / costs.at["fuel cell", "efficiency"]
        shipping_hydrogen_share = get(
            options["shipping_hydrogen_share"], investment_year
        )

        if options["shipping_hydrogen_liquefaction"]:
            n.madd(
                "Bus",
                nodes,
                suffix=" H2 liquid",
                carrier="H2 liquid",
                location=nodes,
                unit="MWh_LHV",
            )

            n.madd(
                "Link",
                nodes + " H2 liquefaction",
                bus0=nodes + " H2",
                bus1=nodes + " H2 liquid",
                carrier="H2 liquefaction",
                efficiency=costs.at["H2 liquefaction", "efficiency"],
                capital_cost=costs.at["H2 liquefaction", "fixed"],
                p_nom_extendable=True,
                lifetime=costs.at["H2 liquefaction", "lifetime"],
            )

            shipping_bus = nodes + " H2 liquid"
        else:
            shipping_bus = nodes + " H2"

        efficiency = (
            options["shipping_oil_efficiency"] / costs.at["fuel cell", "efficiency"]
        )
        p_set_hydrogen = shipping_hydrogen_share * p_set * efficiency

        n.madd(
            "Load",
            nodes,
            suffix=" H2 for shipping",
            bus=shipping_bus,
            carrier="H2 for shipping",
            p_set=p_set_hydrogen,
        )

    if shipping_methanol_share:
        n.madd(
            "Bus",
            spatial.methanol.nodes,
            carrier="methanol",
            location=spatial.methanol.locations,
            unit="MWh_LHV",
        )

        n.madd(
            "Store",
            spatial.methanol.nodes,
            suffix=" Store",
            bus=spatial.methanol.nodes,
            e_nom_extendable=True,
            e_cyclic=True,
            carrier="methanol",
        )

        n.madd(
            "Link",
            spatial.h2.locations + " methanolisation",
            bus0=spatial.h2.nodes,
            bus1=spatial.methanol.nodes,
            bus2=nodes,
            bus3=spatial.co2.nodes,
            carrier="methanolisation",
            p_nom_extendable=True,
            p_min_pu=options.get("min_part_load_methanolisation", 0),
            capital_cost=costs.at["methanolisation", "fixed"]
            * options["MWh_MeOH_per_MWh_H2"],  # EUR/MW_H2/a
            lifetime=costs.at["methanolisation", "lifetime"],
            efficiency=options["MWh_MeOH_per_MWh_H2"],
            efficiency2=-options["MWh_MeOH_per_MWh_H2"] / options["MWh_MeOH_per_MWh_e"],
            efficiency3=-options["MWh_MeOH_per_MWh_H2"] / options["MWh_MeOH_per_tCO2"],
        )

        efficiency = (
            options["shipping_oil_efficiency"] / options["shipping_methanol_efficiency"]
        )
        p_set_methanol = shipping_methanol_share * p_set.sum() * efficiency

        n.madd(
            "Load",
            spatial.methanol.nodes,
            suffix=" shipping methanol",
            bus=spatial.methanol.nodes,
            carrier="shipping methanol",
            p_set=p_set_methanol,
        )

        # CO2 intensity methanol based on stoichiometric calculation with 22.7 GJ/t methanol (32 g/mol), CO2 (44 g/mol), 277.78 MWh/TJ = 0.218 t/MWh
        co2 = p_set_methanol / options["MWh_MeOH_per_tCO2"]

        n.add(
            "Load",
            "shipping methanol emissions",
            bus="co2 atmosphere",
            carrier="shipping methanol emissions",
            p_set=-co2,
        )

    if shipping_oil_share:
        # p_set_oil = shipping_oil_share * p_set.sum()

        # get CLEVER oil shipping demand and sum for copperplated oil demand
        clever_shi_oil = clever_dict["transport"]["shipping_liquid_fuels"][str(investment_year)].loc[countries]
        p_set_shi_oil = clever_shi_oil.sum() * 1e6 / nhours


        n.madd(
            "Load",
            spatial.oil.nodes,
            suffix=" shipping oil",
            bus=spatial.oil.nodes,
            carrier="shipping oil",
            p_set=p_set_shi_oil,
        )

        co2 = p_set_shi_oil * costs.at["oil", "CO2 intensity"]

        n.add(
            "Load",
            "shipping oil emissions",
            bus="co2 atmosphere",
            carrier="shipping oil emissions",
            p_set=-co2,
        )

    if shipping_gas_share:
        # p_set_gas = shipping_gas_share * p_set.sum()

        # get CLEVER oil shipping demand
        clever_shi_gas_nat = clever_dict["transport"]["shipping_gas"][str(investment_year)].loc[countries]

        if options["gas_network"]:
            # use total pypsa shipping demand to distribute CLEVER gas shipping demand
            pes_shi_reg = all_navigation.to_frame()
            clever_shi_gas_reg = distribute_sce_demand_by_pes_layout(clever_shi_gas_nat, pes_shi_reg, pop_layout)
            # convert to MW
            p_set_shi_gas = clever_shi_gas_reg * 1e6 / nhours

            n.madd(
                "Load",
                nodes,
                suffix=" shipping gas",
                bus=spatial.gas.nodes,
                carrier="shipping gas",
                p_set=p_set_shi_gas,
            )

        else:
            p_set_shi_gas = clever_shi_gas_nat.sum() * 1e6 / nhours

            n.madd(
                "Load",
                spatial.gas.nodes,
                suffix=" shipping gas",
                bus=spatial.gas.nodes,
                carrier="shipping gas",
                p_set=p_set_shi_gas,
            )

        # co2 only with EU node
        co2 = p_set_shi_gas.sum() * costs.at["gas", "CO2 intensity"]

        n.add(
            "Load",
            "shipping gas emissions",
            bus="co2 atmosphere",
            carrier="shipping gas emissions",
            p_set=-co2,
        )

    if "oil" not in n.buses.carrier.unique():
        n.madd(
            "Bus",
            spatial.oil.nodes,
            location=spatial.oil.locations,
            carrier="oil",
            unit="MWh_LHV",
        )

    if "oil" not in n.stores.carrier.unique():
        # could correct to e.g. 0.001 EUR/kWh * annuity and O&M
        n.madd(
            "Store",
            [oil_bus + " Store" for oil_bus in spatial.oil.nodes],
            bus=spatial.oil.nodes,
            e_nom_extendable=True,
            e_cyclic=True,
            carrier="oil",
        )

    if "oil" not in n.generators.carrier.unique():
        n.madd(
            "Generator",
            spatial.oil.nodes,
            bus=spatial.oil.nodes,
            p_nom_extendable=True,
            carrier="oil",
            marginal_cost=costs.at["oil", "fuel"],
        )

    if options["oil_boilers"]:
        nodes_heat = create_nodes_for_heat_sector()[0]

        for name in [
            "residential rural",
            "services rural",
            "residential urban decentral",
            "services urban decentral",
        ]:
            n.madd(
                "Link",
                nodes_heat[name] + f" {name} oil boiler",
                p_nom_extendable=True,
                bus0=spatial.oil.nodes,
                bus1=nodes_heat[name] + f" {name}  heat",
                bus2="co2 atmosphere",
                carrier=f"{name} oil boiler",
                efficiency=costs.at["decentral oil boiler", "efficiency"],
                efficiency2=costs.at["oil", "CO2 intensity"],
                capital_cost=costs.at["decentral oil boiler", "efficiency"]
                * costs.at["decentral oil boiler", "fixed"],
                lifetime=costs.at["decentral oil boiler", "lifetime"],
            )

    n.madd(
        "Link",
        nodes + " Fischer-Tropsch",
        bus0=nodes + " H2",
        bus1=spatial.oil.nodes,
        bus2=spatial.co2.nodes,
        carrier="Fischer-Tropsch",
        efficiency=costs.at["Fischer-Tropsch", "efficiency"],
        capital_cost=costs.at["Fischer-Tropsch", "fixed"]
        * costs.at["Fischer-Tropsch", "efficiency"],  # EUR/MW_H2/a
        efficiency2=-costs.at["oil", "CO2 intensity"]
        * costs.at["Fischer-Tropsch", "efficiency"],
        p_nom_extendable=True,
        p_min_pu=options.get("min_part_load_fischer_tropsch", 0),
        lifetime=costs.at["Fischer-Tropsch", "lifetime"],
    )

    demand_factor = options.get("HVC_demand_factor", 1)

    p_set_nap = demand_factor * clever_dict["industry"]["naphtha"].loc[
        countries, str(investment_year)].sum() * 1e6 / nhours

    if demand_factor != 1:
        logger.warning(f"Changing HVC demand by {demand_factor*100-100:+.2f}%.")

    n.madd(
        "Load",
        ["naphtha for industry"],
        bus=spatial.oil.nodes,
        carrier="naphtha for industry",
        p_set=p_set_nap,
    )

    demand_factor = options.get("aviation_demand_factor", 1)
    # all_aviation = ["total international aviation", "total domestic aviation"]
    # p_set = (
    #     demand_factor
    #     * pop_weighted_energy_totals.loc[nodes, all_aviation].sum(axis=1).sum()
    #     * 1e6
    #     / nhours
    # )
    if demand_factor != 1:
        logger.warning(f"Changing aviation demand by {demand_factor*100-100:+.2f}%.")

    # get CLEVER aviation demand on liquid fuels (kerosene) and sum to one node
    clever_avi_liq = clever_dict["transport"]["aviation_liquid_fuels"][str(investment_year)].loc[countries]
    p_set_avi_ker = demand_factor * clever_avi_liq.sum() * 1e6 / nhours

    n.madd(
        "Load",
        ["kerosene for aviation"],
        bus=spatial.oil.nodes,
        carrier="kerosene for aviation",
        p_set=p_set_avi_ker,
    )

    # NB: CO2 gets released again to atmosphere when plastics decay or kerosene is burned
    # except for the process emissions when naphtha is used for petrochemicals, which can be captured with other industry process emissions
    # tco2 per hour
    co2_release = ["naphtha for industry", "kerosene for aviation"]
    co2 = (
        n.loads.loc[co2_release, "p_set"].sum() * costs.at["oil", "CO2 intensity"]
        - industrial_demand.loc[nodes, "process emission from feedstock"].sum() / nhours
    )

    n.add(
        "Load",
        "oil emissions",
        bus="co2 atmosphere",
        carrier="oil emissions",
        p_set=-co2,
    )

    # TODO simplify bus expression
    n.madd(
        "Load",
        nodes,
        suffix=" low-temperature heat for industry",
        bus=[
            node + " urban central heat"
            if node + " urban central heat" in n.buses.index
            else node + " services urban decentral heat"
            for node in nodes
        ],
        carrier="low-temperature heat for industry",
        p_set=industrial_demand.loc[nodes, "low-temperature heat"] / nhours,
    )

    # get CLEVER current (base year) industrial electricity demand
    base_year = clever_dict["industry"]["electricity"].columns.min()
    clever_cel_nat = clever_dict["industry"]["electricity"][base_year] * 1e6
    pes_cel_reg = industrial_demand.loc[nodes, "current electricity"].to_frame()

    # distribute CLEVER demand
    clever_cel_reg = distribute_sce_demand_by_pes_layout(clever_cel_nat, pes_cel_reg, pop_layout)

    # replace pes with sce demand
    industrial_demand["current electricity"] = clever_cel_reg

    # remove today's industrial electricity demand by scaling down total electricity demand
    for ct in n.buses.country.dropna().unique():
        # TODO map onto n.bus.country

        loads_i = n.loads.index[
            (n.loads.index.str[:2] == ct) & (n.loads.carrier == "electricity")
        ]
        if n.loads_t.p_set[loads_i].empty:
            continue
        factor = (
            1
            - industrial_demand.loc[loads_i, "current electricity"].sum()
            / n.loads_t.p_set[loads_i].sum().sum()
        )
        n.loads_t.p_set[loads_i] *= factor

    # get CLEVER scenario industrial electricity demand
    clever_ele_nat = clever_dict["industry"]["electricity"][str(investment_year)]
    pes_ele_reg = industrial_demand.loc[nodes, "electricity"].to_frame() / 1e6

    # distribute CLEVER electricity demand
    clever_ele_reg = distribute_sce_demand_by_pes_layout(clever_ele_nat, pes_ele_reg, pop_layout)

    # replace pes with CLEVER electricity demand
    p_set_ele = clever_ele_reg * 1e6 / nhours

    n.madd(
        "Load",
        nodes,
        suffix=" industry electricity",
        bus=nodes,
        carrier="industry electricity",
        p_set=p_set_ele,
    )

    n.madd(
        "Bus",
        spatial.co2.process_emissions,
        location=spatial.co2.locations,
        carrier="process emissions",
        unit="t_co2",
    )

    sel = ["process emission", "process emission from feedstock"]
    if options["co2_spatial"] or options["co2network"]:
        p_set = (
            -industrial_demand.loc[nodes, sel]
            .sum(axis=1)
            .rename(index=lambda x: x + " process emissions")
            / nhours
        )
    else:
        p_set = -industrial_demand.loc[nodes, sel].sum(axis=1).sum() / nhours

    # this should be process emissions fossil+feedstock
    # then need load on atmosphere for feedstock emissions that are currently going to atmosphere via Link Fischer-Tropsch demand
    n.madd(
        "Load",
        spatial.co2.process_emissions,
        bus=spatial.co2.process_emissions,
        carrier="process emissions",
        p_set=p_set,
    )

    n.madd(
        "Link",
        spatial.co2.process_emissions,
        bus0=spatial.co2.process_emissions,
        bus1="co2 atmosphere",
        carrier="process emissions",
        p_nom_extendable=True,
        efficiency=1.0,
    )

    # assume enough local waste heat for CC
    n.madd(
        "Link",
        spatial.co2.locations,
        suffix=" process emissions CC",
        bus0=spatial.co2.process_emissions,
        bus1="co2 atmosphere",
        bus2=spatial.co2.nodes,
        carrier="process emissions CC",
        p_nom_extendable=True,
        capital_cost=costs.at["cement capture", "fixed"],
        efficiency=1 - costs.at["cement capture", "capture_rate"],
        efficiency2=costs.at["cement capture", "capture_rate"],
        lifetime=costs.at["cement capture", "lifetime"],
    )

    if options.get("ammonia"):
        if options["ammonia"] == "regional":
            p_set = (
                industrial_demand.loc[spatial.ammonia.locations, "ammonia"].rename(
                    index=lambda x: x + " NH3"
                )
                / nhours
            )
        else:
            p_set = industrial_demand["ammonia"].sum() / nhours

        n.madd(
            "Load",
            spatial.ammonia.nodes,
            bus=spatial.ammonia.nodes,
            carrier="NH3",
            p_set=p_set,
        )


def add_waste_heat(n):
    # TODO options?

    logger.info("Add possibility to use industrial waste heat in district heating")

    # AC buses with district heating
    urban_central = n.buses.index[n.buses.carrier == "urban central heat"]
    if not urban_central.empty:
        urban_central = urban_central.str[: -len(" urban central heat")]

        # TODO what is the 0.95 and should it be a config option?
        if options["use_fischer_tropsch_waste_heat"]:
            n.links.loc[urban_central + " Fischer-Tropsch", "bus3"] = (
                urban_central + " urban central heat"
            )
            n.links.loc[urban_central + " Fischer-Tropsch", "efficiency3"] = (
                0.95 - n.links.loc[urban_central + " Fischer-Tropsch", "efficiency"]
            )

        # TODO integrate usable waste heat efficiency into technology-data from DEA
        if options.get("use_electrolysis_waste_heat", False):
            n.links.loc[urban_central + " H2 Electrolysis", "bus2"] = (
                urban_central + " urban central heat"
            )
            n.links.loc[urban_central + " H2 Electrolysis", "efficiency2"] = (
                0.84 - n.links.loc[urban_central + " H2 Electrolysis", "efficiency"]
            )

        if options["use_fuel_cell_waste_heat"]:
            n.links.loc[urban_central + " H2 Fuel Cell", "bus2"] = (
                urban_central + " urban central heat"
            )
            n.links.loc[urban_central + " H2 Fuel Cell", "efficiency2"] = (
                0.95 - n.links.loc[urban_central + " H2 Fuel Cell", "efficiency"]
            )


def add_agriculture(n, costs):
    logger.info("Add agriculture, forestry and fishing sector.")

    nodes = pop_layout.index
    nhours = n.snapshot_weightings.generators.sum()

    countries = snakemake.config["countries"]

    # pes total agricultural demand and fraction of subsectors
    pes_agr_total_reg = pop_weighted_energy_totals.loc[nodes, "total agriculture"]
    frac_agr_heat = pop_weighted_energy_totals.loc[nodes, "total agriculture heat"].div(pes_agr_total_reg)
    frac_agr_elec = pop_weighted_energy_totals.loc[nodes, "total agriculture electricity"].div(pes_agr_total_reg)
    frac_agr_mach = pop_weighted_energy_totals.loc[nodes, "total agriculture machinery"].div(pes_agr_total_reg)

    # get total CLEVER total agriculture demand to distribute to electricity, heating, machinery oil
    clever_agr_total_nat = clever_dict["agriculture"]["total"][str(investment_year)].loc[countries]

    # scale and distribute CLEVER total agriculture demand with PyPSA total agriculture demand
    clever_agr_total_reg = distribute_sce_demand_by_pes_layout(clever_agr_total_nat, pes_agr_total_reg.to_frame(),
                                                               pop_layout)

    # electricity
    p_set_agr_elec = clever_agr_total_reg * frac_agr_elec * 1e6 / nhours

    n.madd(
        "Load",
        nodes,
        suffix=" agriculture electricity",
        bus=nodes,
        carrier="agriculture electricity",
        p_set=p_set_agr_elec,
    )

    # heat
    p_set_agr_heat = clever_agr_total_reg * frac_agr_heat * 1e6 / nhours

    n.madd(
        "Load",
        nodes,
        suffix=" agriculture heat",
        bus=nodes + " services rural heat",
        carrier="agriculture heat",
        p_set=p_set_agr_heat,
    )

    # machinery

    # get CLEVER scenario agriculture demand data for carriers
    # extracted from dictionary with all clever data for investment year
    clever_df_agr_carrier = pd.DataFrame(
        {subsec: values.loc[countries,str(investment_year)]
         for subsec, values in clever_dict["agriculture"].items()
         }
    )
    # data for base year
    base_year = clever_dict["agriculture"]["total"].columns.min()
    clever_df_agr_carrier_base = pd.DataFrame(
        {subsec: values.loc[countries,base_year]
         for subsec, values in clever_dict["agriculture"].items()
         }
    )
    # determine oil share, when liquid fuels greater than in base year oil share is still 1.
    # Otherwise, oil share is fraction relative to liquid fuels base year
    oil_share = pd.Series(
        np.where(clever_df_agr_carrier["liquid_fuels"] > clever_df_agr_carrier_base["liquid_fuels"], 1,
                 clever_df_agr_carrier["liquid_fuels"] / clever_df_agr_carrier_base["liquid_fuels"]), index=countries)
    # electric share and gas share are distributed from remaining share by
    # their relative fraction to each other in agriculture carrier demands
    electric_share = (1 - oil_share) * clever_df_agr_carrier["electricity"] / (
                clever_df_agr_carrier["electricity"] + clever_df_agr_carrier["gas"])
    gas_share = (1 - oil_share) * clever_df_agr_carrier["gas"] / (
                clever_df_agr_carrier["electricity"] + clever_df_agr_carrier["gas"])
    total_share = (electric_share + oil_share + gas_share).mean()

    # regionalise oil, electric and gas shares
    pes_agr_total_reg_df = pes_agr_total_reg.to_frame()
    pes_agr_total_reg_df["ctry"] = pes_agr_total_reg.index.str[:2]
    oil_share_reg = pes_agr_total_reg_df.ctry.map(oil_share)
    electric_share_reg = pes_agr_total_reg_df.ctry.map(electric_share)
    gas_share_reg = pes_agr_total_reg_df.ctry.map(gas_share)

    if total_share != 1:
        logger.warning(
            f"Total agriculture machinery shares sum up to {total_share:.2%}, corresponding to increased or decreased demand assumptions."
        )

    p_set_agr_mach = clever_agr_total_reg * frac_agr_mach * 1e6 / nhours

    if electric_share.any() > 0:
        efficiency_gain = (
            options["agriculture_machinery_fuel_efficiency"]
            / options["agriculture_machinery_electric_efficiency"]
        )

        n.madd(
            "Load",
            nodes,
            suffix=" agriculture machinery electric",
            bus=nodes,
            carrier="agriculture machinery electric",
            p_set=electric_share_reg
            / efficiency_gain
            * p_set_agr_mach,
        )

    if oil_share.any() > 0:
        n.madd(
            "Load",
            ["agriculture machinery oil"],
            bus=spatial.oil.nodes,
            carrier="agriculture machinery oil",
            p_set=(oil_share_reg * p_set_agr_mach).sum(),
        )

        co2 = (
            (oil_share_reg
            * p_set_agr_mach).sum()
            * costs.at["oil", "CO2 intensity"]
        )

        n.add(
            "Load",
            "agriculture machinery oil emissions",
            bus="co2 atmosphere",
            carrier="agriculture machinery oil emissions",
            p_set=-co2,
        )

    if gas_share.any() > 0:

        n.madd(
            "Load",
            nodes,
            suffix=" agriculture machinery gas",
            bus=spatial.gas.nodes,
            carrier="agriculture machinery gas",
            p_set=gas_share_reg * p_set_agr_mach,
        )

        co2 = (
            (gas_share_reg
            * p_set_agr_mach).sum()
            * costs.at["gas", "CO2 intensity"]
        )

        n.add(
            "Load",
            "agriculture machinery gas emissions",
            bus="co2 atmosphere",
            carrier="agriculture machinery gas emissions",
            p_set=-co2,
        )


def decentral(n):
    """
    Removes the electricity transmission system.
    """
    n.lines.drop(n.lines.index, inplace=True)
    n.links.drop(n.links.index[n.links.carrier.isin(["DC", "B2B"])], inplace=True)


def remove_h2_network(n):
    n.links.drop(
        n.links.index[n.links.carrier.str.contains("H2 pipeline")], inplace=True
    )

    if "EU H2 Store" in n.stores.index:
        n.stores.drop("EU H2 Store", inplace=True)


def maybe_adjust_costs_and_potentials(n, opts):
    for o in opts:
        if "+" not in o:
            continue
        oo = o.split("+")
        carrier_list = np.hstack(
            (
                n.generators.carrier.unique(),
                n.links.carrier.unique(),
                n.stores.carrier.unique(),
                n.storage_units.carrier.unique(),
            )
        )
        suptechs = map(lambda c: c.split("-", 2)[0], carrier_list)
        if oo[0].startswith(tuple(suptechs)):
            carrier = oo[0]
            attr_lookup = {"p": "p_nom_max", "e": "e_nom_max", "c": "capital_cost"}
            attr = attr_lookup[oo[1][0]]
            factor = float(oo[1][1:])
            # beware if factor is 0 and p_nom_max is np.inf, 0*np.inf is nan
            if carrier == "AC":  # lines do not have carrier
                n.lines[attr] *= factor
            else:
                if attr == "p_nom_max":
                    comps = {"Generator", "Link", "StorageUnit"}
                elif attr == "e_nom_max":
                    comps = {"Store"}
                else:
                    comps = {"Generator", "Link", "StorageUnit", "Store"}
                for c in n.iterate_components(comps):
                    if carrier == "solar":
                        sel = c.df.carrier.str.contains(
                            carrier
                        ) & ~c.df.carrier.str.contains("solar rooftop")
                    else:
                        sel = c.df.carrier.str.contains(carrier)
                    c.df.loc[sel, attr] *= factor
            logger.info(f"changing {attr} for {carrier} by factor {factor}")


# TODO this should rather be a config no wildcard
def limit_individual_line_extension(n, maxext):
    logger.info(f"Limiting new HVAC and HVDC extensions to {maxext} MW")
    n.lines["s_nom_max"] = n.lines["s_nom"] + maxext
    hvdc = n.links.index[n.links.carrier == "DC"]
    n.links.loc[hvdc, "p_nom_max"] = n.links.loc[hvdc, "p_nom"] + maxext


aggregate_dict = {
    "p_nom": "sum",
    "s_nom": "sum",
    "v_nom": "max",
    "v_mag_pu_max": "min",
    "v_mag_pu_min": "max",
    "p_nom_max": "sum",
    "s_nom_max": "sum",
    "p_nom_min": "sum",
    "s_nom_min": "sum",
    "v_ang_min": "max",
    "v_ang_max": "min",
    "terrain_factor": "mean",
    "num_parallel": "sum",
    "p_set": "sum",
    "e_initial": "sum",
    "e_nom": "sum",
    "e_nom_max": "sum",
    "e_nom_min": "sum",
    "state_of_charge_initial": "sum",
    "state_of_charge_set": "sum",
    "inflow": "sum",
    "p_max_pu": "first",
    "x": "mean",
    "y": "mean",
}


def cluster_heat_buses(n):
    """
    Cluster residential and service heat buses to one representative bus.

    This can be done to save memory and speed up optimisation
    """

    def define_clustering(attributes, aggregate_dict):
        """Define how attributes should be clustered.
        Input:
            attributes    : pd.Index()
            aggregate_dict: dictionary (key: name of attribute, value
                                        clustering method)

        Returns:
            agg           : clustering dictionary
        """
        keys = attributes.intersection(aggregate_dict.keys())
        agg = dict(
            zip(
                attributes.difference(keys),
                ["first"] * len(df.columns.difference(keys)),
            )
        )
        for key in keys:
            agg[key] = aggregate_dict[key]
        return agg

    logger.info("Cluster residential and service heat buses.")
    components = ["Bus", "Carrier", "Generator", "Link", "Load", "Store"]

    for c in n.iterate_components(components):
        df = c.df
        cols = df.columns[df.columns.str.contains("bus") | (df.columns == "carrier")]

        # rename columns and index
        df[cols] = df[cols].apply(
            lambda x: x.str.replace("residential ", "").str.replace("services ", ""),
            axis=1,
        )
        df = df.rename(
            index=lambda x: x.replace("residential ", "").replace("services ", "")
        )

        # cluster heat nodes
        # static dataframe
        agg = define_clustering(df.columns, aggregate_dict)
        df = df.groupby(level=0).agg(agg, **agg_group_kwargs)
        # time-varying data
        pnl = c.pnl
        agg = define_clustering(pd.Index(pnl.keys()), aggregate_dict)
        for k in pnl.keys():
            pnl[k].rename(
                columns=lambda x: x.replace("residential ", "").replace(
                    "services ", ""
                ),
                inplace=True,
            )
            pnl[k] = pnl[k].groupby(level=0, axis=1).agg(agg[k], **agg_group_kwargs)

        # remove unclustered assets of service/residential
        to_drop = c.df.index.difference(df.index)
        n.mremove(c.name, to_drop)
        # add clustered assets
        to_add = df.index.difference(c.df.index)
        import_components_from_dataframe(n, df.loc[to_add], c.name)


def apply_time_segmentation(
    n, segments, solver_name="cbc", overwrite_time_dependent=True
):
    """
    Aggregating time series to segments with different lengths.

    Input:
        n: pypsa Network
        segments: (int) number of segments in which the typical period should be
                  subdivided
        solver_name: (str) name of solver
        overwrite_time_dependent: (bool) overwrite time dependent data of pypsa network
        with typical time series created by tsam
    """
    try:
        import tsam.timeseriesaggregation as tsam
    except:
        raise ModuleNotFoundError(
            "Optional dependency 'tsam' not found." "Install via 'pip install tsam'"
        )

    # get all time-dependent data
    columns = pd.MultiIndex.from_tuples([], names=["component", "key", "asset"])
    raw = pd.DataFrame(index=n.snapshots, columns=columns)
    for c in n.iterate_components():
        for attr, pnl in c.pnl.items():
            # exclude e_min_pu which is used for SOC of EVs in the morning
            if not pnl.empty and attr != "e_min_pu":
                df = pnl.copy()
                df.columns = pd.MultiIndex.from_product([[c.name], [attr], df.columns])
                raw = pd.concat([raw, df], axis=1)

    # normalise all time-dependent data
    annual_max = raw.max().replace(0, 1)
    raw = raw.div(annual_max, level=0)

    # get representative segments
    agg = tsam.TimeSeriesAggregation(
        raw,
        hoursPerPeriod=len(raw),
        noTypicalPeriods=1,
        noSegments=int(segments),
        segmentation=True,
        solver=solver_name,
    )
    segmented = agg.createTypicalPeriods()

    weightings = segmented.index.get_level_values("Segment Duration")
    offsets = np.insert(np.cumsum(weightings[:-1]), 0, 0)
    timesteps = [raw.index[0] + pd.Timedelta(f"{offset}h") for offset in offsets]
    snapshots = pd.DatetimeIndex(timesteps)
    sn_weightings = pd.Series(
        weightings, index=snapshots, name="weightings", dtype="float64"
    )

    n.set_snapshots(sn_weightings.index)
    n.snapshot_weightings = n.snapshot_weightings.mul(sn_weightings, axis=0)

    # overwrite time-dependent data with timeseries created by tsam
    if overwrite_time_dependent:
        values_t = segmented.mul(annual_max).set_index(snapshots)
        for component, key in values_t.columns.droplevel(2).unique():
            n.pnl(component)[key] = values_t[component, key]

    return n


def set_temporal_aggregation(n, opts, solver_name):
    """
    Aggregate network temporally.
    """
    for o in opts:
        # temporal averaging
        m = re.match(r"^\d+h$", o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m.group(0))
            break
        # representative snapshots
        m = re.match(r"(^\d+)sn$", o, re.IGNORECASE)
        if m is not None:
            sn = int(m[1])
            logger.info(f"Use every {sn} snapshot as representative")
            n.set_snapshots(n.snapshots[::sn])
            n.snapshot_weightings *= sn
            break
        # segments with package tsam
        m = re.match(r"^(\d+)seg$", o, re.IGNORECASE)
        if m is not None:
            segments = int(m[1])
            logger.info(f"Use temporal segmentation with {segments} segments")
            n = apply_time_segmentation(n, segments, solver_name=solver_name)
            break
    return n

# get demand data from clever scenario in one dict of dicts
def get_clever_demand() -> dict:
    """
    read demands for clever scenario for all sectors
    """
    clever_demand = {}
    # dictionary that states for all sectors which subsectors there are to read
    sectors = {"agriculture": ["electricity", "liquid_fuels", "gas", "total"],
               "services": ["electricity", "ambient_heat", "network_heat", "solar_thermal", "total"],
               "residential": ["electricity", "space_heating", "space_heating_electricity", "water_heating",
                               "water_heating_electricity", "network_heat", "total"],
               "industry": ["electricity", "gas", "h2", "naphtha", "solid_biomass"],
               "transport": ["land_ev", "land_h2", "land_gas", "land_liquid_fuels", "shipping_liquid_fuels",
                             "shipping_gas", "aviation_liquid_fuels"]
               }
    # read for all sectors each subsector and store in list in corresponding dictionary entry
    for sector, subsectors in sectors.items():
        # store subsector demands in dictionary for each sector
        clever_demand[sector] = {}
        for subsector in subsectors:
            # read demand with country code as index
            demand_subsector = pd.read_csv(
                snakemake.input.clever_demand_files + f"/clever_demand_{sector}_{subsector}.csv",
                decimal=',', delimiter=';',
                index_col=0).dropna()
            # rename Greece and UK to fit PyPSA country ID: 'EL' --> 'GR' in pypsa, 'UK' --> 'GB' in pypsa
            demand_subsector.rename(index={'EL': 'GR', 'UK': 'GB'}, inplace=True)
            # store subsector demand
            clever_demand[sector][subsector] = demand_subsector

    return clever_demand

def distribute_sce_demand_by_pes_layout(sce_demand_nat, pes_demand_reg, pop_layout):

    # get embedded country codes
    pes_demand_reg['ctry'] = pes_demand_reg.index.str[:2]
    # calculate fractions for cluster within a country
    pes_demand_reg['fraction'] = pes_demand_reg.groupby("ctry").transform(lambda x: x / x.sum())

    # in case pes demand is zero, but sce demand not - weight based on population
    if pes_demand_reg.fraction.isnull().any():
        pes_demand_reg['fraction'] = pes_demand_reg.fraction.fillna(pop_layout.fraction)

    # mapping scenario demand to cluster and multiplying by fraction to distribute scenario demand accordingly
    sce_demand_reg = pes_demand_reg.ctry.map(sce_demand_nat).mul(pes_demand_reg.fraction)

    return sce_demand_reg

def scale_district_heating_dem(n, year):

    # Code from branch https://github.com/PyPSA/pypsa-eur-sec/tree/PAC

    # get CLEVER share for modelled countries and pes demands
    share = calculate_clever_dh_share(clever_dict, year)
    pes_heat_total = n.loads_t.p_set.loc[:, n.loads.carrier.str.contains("heat")].sum().sum()
    pes_heat_decentral = n.loads_t.p_set.loc[:, (n.loads.carrier.str.contains("heat")) &
                                             ~((n.loads.carrier.str.contains("urban central heat")))].sum().sum()
    pes_heat_central = n.loads_t.p_set.loc[:, n.loads.carrier == "urban central heat"].sum().sum()

    # calculate scaling factors in relation to dh and non dh shares from pes
    share_pes = pes_heat_central / pes_heat_total
    scale_factor_not_dh = (1 - share) / (1 - share_pes)
    scale_factor_dh = share / share_pes

    # convert scaling factor to array that can be multiplied with matrix
    scale_factor_not_dh_array = n.loads_t.p_set.loc[:, (n.loads.carrier.str.contains("heat")) & ~(
    (n.loads.carrier.str.contains("urban central heat")))].columns.str[:2].map(scale_factor_not_dh).values
    scale_factor_dh_array = n.loads_t.p_set.loc[:, n.loads.carrier == "urban central heat"].columns.str[:2].map(
        scale_factor_dh).values

    # scale demands

    n.loads_t.p_set.loc[:, (n.loads.carrier.str.contains("heat")) &
                            ~((n.loads.carrier.str.contains("urban central heat")))] *= scale_factor_not_dh_array
    n.loads_t.p_set.loc[:, n.loads.carrier=="urban central heat"] *= scale_factor_dh_array

    # print("district heating share is scaled up by; ", scale_factor_dh)

def calculate_clever_dh_share(clever_dict, year):
    """
    calculates the dh share of clever heating demand data.
    paramters:
    @param clever_dict: input dictionary of CLEVER data set
    @param year: investment year
    return:
    @return: dh share per country

    Note:   while residential total heating demand can be derived from space and water heating demands,
            the same can not be done for services subsector. The approach here is to subtract non-heating
            electricity demand from total energy demand to yield total heating demand. non-heating electricity
            demand is approximated by applying the same fraction of heating/non-heating electricity demand as for
            residential subsector.
    """
    # calculate DH share from network heat relative to total heat demand for all modelled countries
    countries = snakemake.config["countries"]
    # total DH demand from residential and services
    clever_dh_total = (clever_dict["residential"]["network_heat"][str(year)] \
                       + clever_dict["services"]["network_heat"][str(year)])[countries]
    # residential total heating demand
    clever_heat_residential = (clever_dict["residential"]["space_heating"][str(year)] \
                               + clever_dict["residential"]["water_heating"][str(year)])[countries]
    # calculate share of residential electricity heating of residential total electricity to apply to services
    share_electric_heat = (clever_dict["residential"]["space_heating_electricity"][str(year)] +
                           clever_dict["residential"]["water_heating_electricity"][str(year)]) / \
                          clever_dict["residential"]["electricity"][str(year)]
    # approximate services total heating demand by subtracting
    # non-heating electricity demand from total services energy demand
    clever_heat_services = (clever_dict["services"]["total"][str(year)] \
                            - (1 - share_electric_heat) \
                            * clever_dict["services"]["electricity"][str(year)])[countries]
    # calculate DH share as total DH demand divided by total heating demand services and residential
    dh_share = clever_dh_total / (clever_heat_services + clever_heat_residential)

    return dh_share

def build_sce_cap_prod(input_path_cap, output_path, indicator="capacity"):
    """
    builds scenario installed electricity production capacities and saves as csv

    parameter:
    @param input_path_cap: path to capacity/production files
    @param output_path: path and name of output file
    @param indicator: whether capacity or production is build
    return:
    NONE
    """

    # carriers to import
    if indicator == "capacity":
        carriers = ["pv", "onwind", "offwind", "hydro"] #, "marinepower", "gth_csp"]
    elif indicator == "production":
        carriers = ["nuclear", "oil", "gas", "coal_lignite", "electrolyser", "hydro", "chp_gas", "chp_biomass"]

    # countries modelled and chosen years
    countries = snakemake.config["countries"]
    years = np.array(snakemake.config["scenario"]["planning_horizons"]).astype(str)

    # import capacities in dictionary
    df_caps = {carr: pd.read_csv(input_path_cap + f"/clever_supply_{indicator}_{carr}.csv",
                                    decimal=',', delimiter=';', index_col=0) \
                            .fillna(0.0) \
                            .rename(index={'EL': 'GR', 'UK': 'GB'}) \
                            .loc[countries, years] for carr in carriers
               }

    # add country and carrier columns
    for carr in carriers:
        df_caps[carr] *= 1e3 # conversion from GW to MW or TWH to GWh respectively
        df_caps[carr]["country"] = df_caps[carr].index.str[:2]
        df_caps[carr]["carrier"] = carr
        if carr == "coal_lignite":
            df_caps[carr]["carrier"] = "coal & lignite"

    # append dataframes in dict to each other
    df = pd.concat(df_caps, axis=0)
    # set multiindex as country and carrier and sort by countries
    df.set_index(["country","carrier"], inplace=True)
    df = df.sort_index(level=[0])
    df.index.names = ["country","carrier"]

    # add NaN rows for conventionals
    if indicator == "capacity":
        carr = ["nuclear", "oil", "gas", "coal & lignite", "electrolyser", "chp_gas", "chp_biomass"] + carriers
        ctys = list(df.index.get_level_values("country").unique())
        multi_idx = pd.MultiIndex.from_product([ctys, carr], names=['country', 'carrier'])
        df = df.reindex(multi_idx)

    # save to csv in stated output path and file name
    df = df.rename({"pv": "solar"}) # rename pv to solar
    df.to_csv(output_path, index=True)

def add_gens(n, costs, year):
    carriers = ["coal", "lignite", "nuclear", "oil"]
    buses_i = [bus for bus in n.buses.location.unique() if bus != "EU"]
    for carrier in carriers:
        print(carrier)
        print(buses_i)
        eu_carrier = {"coal": "coal",
                      "lignite": "lignite",
                      "nuclear": "uranium",
                      "oil": "oil",
                      "gas": "gas",}
        if carrier=="nuclear":
            c = 0
        else:
            c=costs.at[eu_carrier[carrier], "CO2 intensity"]

        n.madd(
            "Link",
            buses_i,
            suffix= " " + carrier +"-"+year,
            bus0="EU " + eu_carrier[carrier],
            bus1=buses_i,
            bus2="co2 atmosphere",
            carrier=carrier,
            build_year=year,
            lifetime=100,
            p_nom_extendable=True,
            p_nom=0,
            p_nom_min=0,
            efficiency=costs.at[carrier, "efficiency"],
            efficiency2=c,
            marginal_cost=costs.at[carrier, "efficiency"]
            * costs.at[carrier, "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at[carrier, "efficiency"]
            * costs.at[carrier, "fixed"],  # NB: fixed cost is per MWel
        )

    carriers = ["ror", "hydro", "PHS"]
    for carrier in carriers:
        if carrier == "ror":
            n.madd(
                "Generator",
                buses_i,
                " " + carrier +"-"+year,
                bus=buses_i,
                carrier=carrier,
                p_nom_extendable=True,
                p_nom_min=0,
                efficiency=costs.at["ror", "efficiency"],
                capital_cost=299140.224929,
                build_year=year,
                lifetime=100
            )
        elif carrier in ["PHS", "hydro"]:
            n.madd(
                "StorageUnit",
                buses_i,
                " " + carrier+"-"+year,
                bus=buses_i,
                carrier=carrier,
                p_nom_extendable=True,
                p_nom_min=0,
                capital_cost=177345.216619,#costs.at[carrier, "capital_cost"],
                #marginal_cost=costs.at[carrier, "marginal_cost"],
                efficiency_store=costs.at[carrier, "efficiency"],
                efficiency_dispatch=costs.at[carrier, "efficiency"],
                cyclic_state_of_charge=True,
                max_hours=0,
                build_year=year,
                lifetime=100
            )

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_sector_network",
            configfiles="config/config.yaml",
            simpl="",
            opts="",
            clusters="30",
            ll="v1.0",
            sector_opts="25H-T-H-B-I-A-dist1",
            planning_horizons="2050",
        )

    logging.basicConfig(level=snakemake.config["logging"]["level"])

    update_config_with_sector_opts(snakemake.config, snakemake.wildcards.sector_opts)

    options = snakemake.config["sector"]

    opts = snakemake.wildcards.sector_opts.split("-")

    investment_year = int(snakemake.wildcards.planning_horizons[-4:])

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    nhours = n.snapshot_weightings.generators.sum()
    nyears = nhours / 8760

    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.config["costs"],
        nyears,
    )

    pop_weighted_energy_totals = (
        pd.read_csv(snakemake.input.pop_weighted_energy_totals, index_col=0) * nyears
    )

    # import all clever scenario demand as dictionary of dictionaries
    clever_dict = get_clever_demand()

    patch_electricity_network(n)

    # scale electricity time series data to CLEVER demand
    # get scenarios electricity demand for residential and services buildings
    clever_ele_res = clever_dict["residential"]["electricity"][str(investment_year)]
    clever_ele_ser = clever_dict["services"]["electricity"][str(investment_year)]

    # get scenarios electricity demand for industry
    clever_ele_ind = clever_dict["industry"]["electricity"][str(investment_year)]

    # get regional distributed pes demand and nationally distributed clever demand per sector
    clever_ele_nat = pd.concat([clever_ele_res, clever_ele_ser, clever_ele_ind], axis=1).sum(axis=1)
    pes_ele_reg = n.loads_t.p_set.loc[:, n.loads.carrier == "electricity"].sum().to_frame() / 1e6

    # scale and distribute
    clever_ele_reg = distribute_sce_demand_by_pes_layout(clever_ele_nat, pes_ele_reg, pop_layout)
    scale_factor = clever_ele_reg.div(pes_ele_reg[0])

    # update electricity demand to sce data using regional scale factor
    n.loads_t.p_set.loc[:, n.loads.carrier == "electricity"] = \
        n.loads_t.p_set.loc[:, n.loads.carrier == "electricity"].mul(scale_factor, axis=1)

    spatial = define_spatial(pop_layout.index, options)

    if snakemake.config["foresight"] == "myopic":
        add_lifetime_wind_solar(n, costs)

        conventional = snakemake.config["existing_capacities"]["conventional_carriers"]
        for carrier in conventional:
            add_carrier_buses(n, carrier)

    add_co2_tracking(n, options)

    add_generation(n, costs)

    add_gens(n, costs, str(investment_year))

    add_storage_and_grids(n, costs)

    # TODO merge with opts cost adjustment below
    for o in opts:
        if o[:4] == "wave":
            wave_cost_factor = float(o[4:].replace("p", ".").replace("m", "-"))
            logger.info(
                f"Including wave generators with cost factor of {wave_cost_factor}"
            )
            add_wave(n, wave_cost_factor)
        if o[:4] == "dist":
            options["electricity_distribution_grid"] = True
            options["electricity_distribution_grid_cost_factor"] = float(
                o[4:].replace("p", ".").replace("m", "-")
            )
        if o == "biomasstransport":
            options["biomass_transport"] = True

    if "nodistrict" in opts:
        options["district_heating"]["progress"] = 0.0

    if "T" in opts:
        add_land_transport(n, costs)

    if "H" in opts:
        add_heat(n, costs)

        scale_district_heating_dem(n, investment_year)

    if "B" in opts:
        add_biomass(n, costs)

    if options["ammonia"]:
        add_ammonia(n, costs)

    if "I" in opts:
        add_industry(n, costs)

    if "I" in opts and "H" in opts:
        add_waste_heat(n)

    if "A" in opts:  # requires H and I
        add_agriculture(n, costs)

    if options["dac"]:
        add_dac(n, costs)

    if "decentral" in opts:
        decentral(n)

    if "noH2network" in opts:
        remove_h2_network(n)

    if options["co2network"]:
        add_co2_network(n, costs)

    if options["allam_cycle"]:
        add_allam(n, costs)

    input_path_clever = snakemake.input.clever_supply_files
    output_path_cap = snakemake.config["electricity"]["agg_p_nom_limits"]
    output_path_prod = snakemake.config["electricity"]["agg_e_gen_limits"]

    # creates csv with installed capacities in MW per target year aggregated to country level
    build_sce_cap_prod(input_path_clever, output_path_cap, "capacity")
    # creates csv with production amount in GWh per target year aggregated to country level
    build_sce_cap_prod(input_path_clever, output_path_prod, "production")

    solver_name = snakemake.config["solving"]["solver"]["name"]
    n = set_temporal_aggregation(n, opts, solver_name)

    limit_type = "config"
    limit = get(snakemake.config["co2_budget"], investment_year)
    for o in opts:
        if "cb" not in o:
            continue
        limit_type = "carbon budget"
        fn = "results/" + snakemake.params.RDIR + "/csvs/carbon_budget_distribution.csv"
        if not os.path.exists(fn):
            emissions_scope = snakemake.config["energy"]["emissions"]
            report_year = snakemake.config["energy"]["eurostat_report_year"]
            build_carbon_budget(
                o, snakemake.input.eurostat, fn, emissions_scope, report_year
            )
        co2_cap = pd.read_csv(fn, index_col=0).squeeze()
        limit = co2_cap.loc[investment_year]
        break
    for o in opts:
        if "Co2L" not in o:
            continue
        limit_type = "wildcard"
        limit = o[o.find("Co2L") + 4 :]
        limit = float(limit.replace("p", ".").replace("m", "-"))
        break
    logger.info(f"Add CO2 limit from {limit_type}")
    add_co2limit(n, nyears, limit)

    for o in opts:
        if not o[:10] == "linemaxext":
            continue
        maxext = float(o[10:]) * 1e3
        limit_individual_line_extension(n, maxext)
        break

    if options["electricity_distribution_grid"]:
        insert_electricity_distribution_grid(n, costs)

    maybe_adjust_costs_and_potentials(n, opts)

    if options["gas_distribution_grid"]:
        insert_gas_distribution_costs(n, costs)

    if options["electricity_grid_connection"]:
        add_electricity_grid_connection(n, costs)

    first_year_myopic = (snakemake.config["foresight"] == "myopic") and (
        snakemake.config["scenario"]["planning_horizons"][0] == investment_year
    )

    if options.get("cluster_heat_buses", False) and not first_year_myopic:
        cluster_heat_buses(n)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))

    n.export_to_netcdf(snakemake.output[0])
