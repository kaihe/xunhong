# chatllm parse api call inputs
from langchain.agents import initialize_agent, Tool
from metra_api import MetraData
metra = MetraData(reload=False)


def query_train_time(string):
    origin, destination = string.split(",")
    if origin == "None":
        return "missing information origin station"
    elif destination == "None":
        return "missing information destination station"
    else:
        return metra.query_route_time(origin, destination)

def query_route(string):
    origin, destination = string.split(",")
    if origin == "None":
        return "missing information origin station"
    elif destination == "None":
        return "missing information destination station"
    else:
        return metra.query_route(origin, destination)

def query_ticket(string):
    origin, destination = string.split(",")
    if origin == "None":
        return "missing information origin station"
    elif destination == "None":
        return "missing information destination station"
    else:
        return metra.query_ticket(origin, destination)

def query_stations(line_name):
    return metra.list_line_stations(line_name)

def query_nearby_station(location):
    if location == 'None':
        return 'need a location to query nearby station'
    else:
        return metra.query_station_nearby(location)

def query_device(string):
    station, device_type = string.split(",")
    if station == "None":
        return "请问您在哪个车站？"
    elif device_type == "None":
        return "请问要查询那种设备？"
    elif station == "None" and device_type == "None":
        return "请问您要查询哪个站的什么设备？"
    return metra.query_device(station_name=station, device_name=device_type)

def ask_for_information(missing_info):
    if missing_info == 'origin_station':
        return '请问您从哪个车站出发?'
    elif missing_info == 'destination_station':
        return '请问去哪个车站？'
    elif missing_info == 'which_station':
        return '请问要查询哪个车站?'
    elif missing_info == 'which_device':
        return '请问要查询什么设施?'
    elif missing_info == 'location':
        return '请问您现在哪个位置？'
    else:
        return '抱歉，请再明确一下您的问题'

mtools = [
    Tool(
        name = "Train Time Query",
        func=query_train_time,
        description="useful for when you need to answer the latest or earliest train. The input to this tool should be a comma separated list of strings of length two, representing the origin station and destination. For example `南山站,None` means from 南山站 and the destination is unknown.",
        return_direct=False
    ),
    Tool(
        name = "Transfer Query",
        func=query_route,
        description="useful for when you need to answer train route transfer information. The input to this tool should be a comma separated list of strings of length two, representing the origin station and destination station. For example `None,南山站` means the origin station is unknown and destination is 南山站. `None,None` means both the origin and destination are unknown",
        return_direct=False
    ),
    Tool(
        name = "Ticket Query",
        func=query_ticket,
        description="useful for when you need to answer ticket price information. The input to this tool should be a comma separated list of strings of length two, representing the origin station and destination station. For example `南山站,碧海湾` means ticket price query from 南山站 to 碧海湾",
        return_direct=False
    ),
    Tool(
        name = "List Stations",
        func=query_stations,
        description="useful for when you need to list stations of a specific line. Input is the line name. For example, `三号线`,`APM线`,`十三号线`. `None` if user haven't specify which line.",
        return_direct=False
    ),
    Tool(
        name = "Device Query",
        func=query_device,
        description="useful for when you need to find a specific device in a station. The input to this tool should be a comma separated list of numbers of length two, representing the station name and device name. For example `南山站,洗手间` means find device 洗手间 at station 南山站",
        return_direct=False
    ),
    # Tool(
    #     name = "Station Query",
    #     func=query_nearby_station,
    #     description="useful for when you need to find the nearby station of a location. The input to this tool is a location or address",
    #     return_direct=False
    # ),
    Tool(
        name = "Information specifier",
        func = ask_for_information,
        description="use this tool if you need to ask user to specify his/her query. The input to this tool is one of the following strings: `origin_station,destination_station,which_station,which_device,location`",
        return_direct=True
    )
]
