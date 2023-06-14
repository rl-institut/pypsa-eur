import pandas as pd
import numpy as np

def import_sce_data(file_path, sheet_name):

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    df = df.query(
        '''
        `Scenario` == 'Distributed Energy' & \
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
    heat_dh_total = share.loc[float(year), 'Distributed Energy'] * heat_total

    # calculate scaling factors
    scale_factor_not_dh = (heat_total - heat_dh_total) / heat_decentral
    scale_factor_dh = heat_dh_total / n.loads_t.p_set.loc[:, n.loads.carrier=="urban central heat"].sum().sum()

    # scale demands
    n.loads_t.p_set.loc[:, (n.loads.carrier.str.contains("heat")) &
                            ~((n.loads.carrier.str.contains("urban central heat")))] *= scale_factor_not_dh
    n.loads_t.p_set.loc[:, n.loads.carrier=="urban central heat"] *= scale_factor_dh

    print("district heating share is scaled up by; ", scale_factor_dh)


def build_sce_capacities(input_path, sheet_name, output_path):

    carrier_mapping = {'wind onshore': 'onwind',
                       'onshore wind_stand alone': 'onwind',
                       'wind offshore': 'offwind',
                       'offshore wind_stand alone': 'offwind',
                       'solar_stand alone': 'solar'}
    # {'|'.join(['Beer', 'Alcohol', 'Beverage', 'Drink']): 'Drink'}
    eu27_str = 'AT|BE|BG|CZ|DE|DK|EE|ES|FI|FR|GB|GR|HR|HU|IE|IT|LT|LU|LV|NL|PL|PT|RO|SE|SI|SK'  #'CY','MT'

    df = pd.read_excel(input_path, sheet_name)

    df_sup = df.query(
        '''
        `Scenario` == 'Distributed Energy' & \
        `Parameter` == 'Capacity (MW)' & \
        `Fuel` != ['Other RES', 'Other Non RES'] & \
        `Climate Year` == 'CY 2009'
        ''').apply(lambda x: x.str.lower() if x.name in ['Fuel'] else x) \
        .rename(columns={"Fuel": "carrier"}) \
        .replace({'carrier': carrier_mapping})

    # add country column
    df_sup = df_sup[df_sup.Node.str.contains(eu27_str)]
    df_sup["country"] = df_sup.Node.str[:2]

    df_agg_caps = pd.pivot_table(df_sup, values='Value', index=['country', 'carrier'],
                                 columns=['Year'], aggfunc=np.sum)

    df_agg_caps.to_csv(output_path, index=True)