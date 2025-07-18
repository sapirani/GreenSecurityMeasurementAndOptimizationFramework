from utils.general_consts import BatteryColumns


def get_powershell_result_list_format(result: bytes):
    """
    This function parse bytes result returned from powershell
    :param result: bytes result returned from powershell. The result should be in list format
    :return: list of dictionaries. We use list because some powershell commands return multimple answers
    """
    lines_list = str(result).split("\\r\\n")[2:-4]
    specific_item_dict = {}
    items_list = []
    for line in lines_list:
        if line == "":
            items_list.append(specific_item_dict)
            specific_item_dict = {}
            continue

        split_line = line.split(":")
        specific_item_dict[split_line[0].strip()] = split_line[1].strip()

    items_list.append(specific_item_dict)
    return items_list


def convert_mwh_to_other_metrics(amount_of_mwh):
    """
    convert mwh to woods, coal, etc.
    :param amount_of_mwh: amount to convert
    :return: tuple of equivalents (woods, coal, etc.)
    """
    kwh_to_mwh = 1e6
    # link: https://www.epa.gov/energy/greenhouse-gases-equivalencies-calculator-calculations-and-references
    co2 = (0.709 * amount_of_mwh) / kwh_to_mwh  # 1 kwh = 0.709 kg co2
    coal_burned = (0.453592 * 0.784 * amount_of_mwh) / kwh_to_mwh  # 1 kwh = 0.784 pound coal
    number_of_smartphones_charged = (86.2 * amount_of_mwh) / kwh_to_mwh  # 1 kwh = 86.2 smartphones

    # the following are pretty much the same. Maybe should consider utilization when converting from heat to electricity
    # link: https://www.cs.mcgill.ca/~rwest/wikispeedia/wpcd/wp/w/Wood_fuel.htm
    # link: https://www3.uwsp.edu/cnr-ap/KEEP/Documents/Activities/Energy%20Fact%20Sheets/FactsAboutWood.pdf
    # link: https://stwww1.weizmann.ac.il/energy/%D7%AA%D7%9B%D7%95%D7%9C%D7%AA-%D7%94%D7%90%D7%A0%D7%A8%D7%92%D7%99%D7%94-%D7%A9%D7%9C-%D7%93%D7%9C%D7%A7%D7%99%D7%9D/
    kg_of_woods_burned = amount_of_mwh / (3.5 * kwh_to_mwh)  # 3.5 kwh = 1 kg of wood

    return co2, coal_burned, number_of_smartphones_charged, kg_of_woods_burned


def calc_delta_capacity(battery_df):
    """
    :return: capacity and percentage drain of the battery during the measurements
    """
    if battery_df.empty:
        return 0, 0
    before_scanning_capacity = battery_df.iloc[0].at[BatteryColumns.CAPACITY]
    current_capacity = battery_df.iloc[len(battery_df) - 1].at[BatteryColumns.CAPACITY]

    before_scanning_percent = battery_df.iloc[0].at[BatteryColumns.PERCENTS]
    current_capacity_percent = battery_df.iloc[len(battery_df) - 1].at[BatteryColumns.PERCENTS]

    return before_scanning_capacity - current_capacity, before_scanning_percent - current_capacity_percent
