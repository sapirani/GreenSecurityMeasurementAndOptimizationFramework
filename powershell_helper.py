

def get_powershell_result_list_format(result: bytes):
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
