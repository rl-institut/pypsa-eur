import pandas as pd


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
                `Energy_carrier` == ['electricity','hydrogen','methane','liquids','solids','biomass','others']
            ''').drop(['Sector_viz_platform'], axis=1) \
            .replace({'Sector': {'tertiary': 'services'}})

        return df_buil

    elif sector == 'industry':

        df_ind = df.query(
            '''
                `Sector` == 'industry'
                ''').drop(['Sector_viz_platform'], axis=1)

        return df_ind

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

def get_elec_demand_by_use(df_buil, subsector, year):

    df = df_buil.query(
        f'''
        `Sector` == '{subsector}' & \
        `Energy_carrier` == ['electricity'] & \
        `Year` == {year}
        ''').drop(['Sector','Subsector','Energy_carrier','Year'], axis=1) \
        .rename(columns={"Value": f"{subsector}"}) \
        .groupby('Country').sum()

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

def distribute_sce_demand_by_pes_layout(sce_demand_nat, pes_demand_reg, pop_layout):

    pes_demand_reg['cty'] = pes_demand_reg.index.str[:2]
    pes_demand_reg['fraction'] = pes_demand_reg.groupby("cty").transform(lambda x: x / x.sum())

    # in case pes demand is zero, but sce demand not - weight based on population
    if pes_demand_reg.fraction.isnull().any():
        pes_demand_reg['fraction'] = pes_demand_reg.fraction.fillna(pop_layout.fraction)

    sce_demand_reg = pes_demand_reg.cty.map(sce_demand_nat).mul(pes_demand_reg.fraction)

    return sce_demand_reg