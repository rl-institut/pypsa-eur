import pandas as pd

def import_sce_data(file_path, sheet_name):

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    df = df.query(
        '''
        `Scenario` == 'Distributed Energy' & \
        `Output_type` == 'Energy_demand' & \
        `Sector` == ['Residential','Tertiary','Agriculture','Transport','Industry']
        ''').drop(['Scenario', 'Output_type'], axis=1).replace({'Country': {'UK': 'GB'}}) \
        .apply(lambda x: x.str.lower() if x.name in ['Sector', 'Subsector', 'Energy_carrier'] else x) \
        .set_index('Country').sort_index()

    return df


def get_sce_data_by_sector(df, sector):

    if sector == 'heat':

        df_heat = df.query(
                '''
                `Sector` == ['residential','tertiary'] & \
                `Subsector` == ['space_heating', 'water_heating'] & \
                `Energy_carrier` == 'all'
                ''').drop(['Energy_carrier', 'Sector_viz_platform'], axis=1) \
                .replace({'Sector': {'tertiary': 'services'}}) \
                .apply(lambda x: x.str[:-8] if x.name in ['Subsector'] else x)

        return df_heat

    elif sector == 'all':

        return df


def get_heat_demand_by_use(df_heat, sector, use, year):

    df = df_heat.query(
        f'''
        `Sector` == '{sector}' & \
        `Subsector` == '{use}' & \
        `Year` == {year}
        ''').drop(['Sector', 'Subsector', 'Year'], axis=1) \
        .rename(columns={"Value": f"{sector} {use}"})

    return df