import json
import random, pickle
import os,re
import pandas as pd
import requests
from collections import defaultdict

API_ERROR = 'data not found in API!!'
CALL_API = '需要调用外部api来回答这个问题,'

# 干掉了一些奇怪的设施
device_config = {
    '羊城通充值点':['地铁卡充值的地方','地铁卡充值点'],
    '楼梯升降机':['升降机','电梯'],
    '专用电梯':[],
    '卫生间':['洗手间','厕所','公厕','男厕所','女厕所'],
    '自动柜员机':['贩卖机'],
    '自动照相机':['自动照相机','照相机','照相点'],
    '自动售卖机':['自动售卖机'],
    '优惠券打印机':[],
    '手机充电机':['手机充电的地方'],
    '便利店':[],
    '面包西饼':['面包西饼店','面包店','西饼店'],
    '糖果小吃':['糖果店','糖果小吃店','小吃店'],
    '书报文具':['报刊亭'],
    '餐饮':['吃饭的地方','饭店'],
    '盲道':[],
    '自助售卡充值机':['售卡机'],
    '母婴室':[],
    '轮椅坡道':['无障碍通道'],
    '出站扶梯':[],
    '进站扶梯':[],
    '无障碍卫生间':[],
    '第三卫生间':[]
}

station_alias = {
    '机场南（1号航站楼）':['机场南','机场','白云机场','白云机场1号航站楼','机场1号航站楼'],
    '机场北（2号航站楼）':['机场北','白云机场2号航站楼', '机场2号航站楼']
}

class MetraData:

    def __init__(self, reload=True) -> None:

        self.url_base = 'https://gzmtrapi-t.jiaxincloud.com/mtrassit'
        # name to id mapping
        self.station_dict = {}
        self.station_id_dict = {}

        # name to id mapping
        self.line_dict = {}

        # device name to id
        self.device_dict = {}

        # line name to station name list
        self.line_station_names = {}
        
        # station name to direction list
        self.station_directions = defaultdict(dict)

        if reload:
            self._init_lines()
            self._init_stations()
            self._init_device()
            with open('config-sample/metra_obj.pickle', 'wb') as fout:
                pickle.dump([self.station_dict, self.line_dict, self.line_station_names, self.station_directions, self.device_dict], fout)
        else:
            self.station_dict, self.line_dict, self.line_station_names, self.station_directions, self.device_dict = pickle.load(open('config-sample/metra_obj.pickle', 'rb'))
            
            # remap device alias, remove unusual devices
            device_dict = {}
            for k, v in device_config.items():
                device_dict[k] = self.device_dict[k]
                for _k in v:
                    device_dict[_k] = self.device_dict[k]
            self.device_dict = device_dict

            # map station alias
            for k, v in station_alias.items():
                _id = self.station_dict[k]
                for alias in v:
                    self.station_dict[alias] = _id

            to_add = []
            for k, _id in self.station_dict.items():
                if k.endswith('站'):
                    to_add.append((k[:-1], _id))
                else:
                    to_add.append((k+'站', _id))

            for k, v in to_add:
                self.station_dict[k] = v

        self.station_id_dict = {v:k for k, v in self.station_dict.items()}

    def _init_lines(self):
        url = f'{self.url_base}/rest/mtr/lines'
        res = requests.get(url).json()

        for rec in res['lines']:
            self.line_dict[rec['nameCn']] = rec['lineId']

        return self.line_dict

    def _init_stations(self):
        for line_name, line_id in self.line_dict.items():
            url = f'{self.url_base}/rest/mtr/lines/{line_id}/stations'
            station_info = requests.get(url).json()['stations']
            dir1 = station_info[0]['nameCn']
            dir2 = station_info[-1]['nameCn']
            self.line_station_names[line_name] = [s['nameCn'] for s in station_info]

            for station in station_info:
                self.station_dict[station['nameCn']] = station['stationId']
                self.station_directions[station['nameCn']][line_name] = {
                    'dir1':dir1,
                    'dir2':dir2
                }

    def _init_device(self):
        url = f'{self.url_base}/rest/mtr/categories'
        device_info = requests.get(url).json()['categories']
        for d in device_info:
            self.device_dict[d['nameCn']] = d['categoryId']

    def _check_station(self, station_name):
        if not station_name in self.station_dict:
            return f'抱歉广州地铁没有{station_name}站' 
    
    def _check_device(self, device_name):
        if device_name not in self.device_dict:
            return f"抱歉广州地铁没有{device_name}"

    def _check_line(self, line_name):
        if not line_name in self.line_station_names:
            return f"抱歉广州地铁没有{line_name}"

    def _query_time(self, from_station, to_station):
        from_id = self.station_dict[from_station]
        to_id = self.station_dict[to_station]
        _url = f'{self.url_base}/rest/mtr/serviceTimes?stationId={from_id}&toStationId={to_id}'

        service_time = requests.get(_url).json()['serviceTimes']
        service_time = [s for s in service_time if s['stationId'] == from_id]
        try:
            return service_time[0]['startTime'], service_time[0]['endTime']
            
        except (KeyError,IndexError):
            return API_ERROR

    def list_line_stations(self, line_name):
        if self._check_line(line_name) is not None:
            return self._check_line(line_name)
        
        return ','.join(self.line_station_names[line_name])

    def query_station_time(self, type, station_name):
        if self._check_station(station_name) is not None:
            return self._check_station(station_name)

        directions = self.station_directions[station_name]
        txt_list = []
        for line_name, dirs in directions.items():
            dir1 = dirs['dir1']
            dir2 = dirs['dir2']

            dir1_time = self._query_time(type, from_station=station_name, to_station=dir1)
            dir2_time = self._query_time(type, from_station=station_name, to_station=dir2)

            _facelet = '首班车' if type == 'startTime' else '末班车'

            txt = f'{line_name}{station_name}前往{dir1}的{_facelet}时间是{dir1_time}，往{dir2}方向的{_facelet}时间是{dir2_time}'
            txt_list.append(txt)

        return '。'.join(txt_list)+'。'

    def query_route_time(self, from_station, to_station):
        if self._check_station(from_station):
            return self._check_station(from_station)
        
        if self._check_station(to_station):
            return self._check_station(to_station)

        start_time, end_time = self._query_time(from_station, to_station)
        txt = f"{from_station}前往{to_station}的首班车时间是{start_time},末班车时间是{end_time}"
        return txt

    def query_device(self, station_name, device_name):
        if self._check_station(station_name):
            return self._check_station(station_name)
        
        if self._check_device(device_name):
            return self._check_device(device_name)

        station_id = self.station_dict[station_name]
        device_id = self.device_dict[device_name]

        _url = f'{self.url_base}/rest/mtr/devices?stationId={station_id}&categoryId={device_id}'
        data = requests.get(_url).json()
        if len(data.get('stations',[]))>0:
            devices = data['stations']
            txt = f'{device_name}位于'
        elif len(data.get('recommendStations',[]))>0:
            txt = f'最近的{device_name}位于'
            devices = data['recommendStations']
        else:
            devices = []

        if len(devices)>0:
            for d in devices:
                if 'stationId' in d:
                    _name = self.station_id_dict[d['stationId']]
                    txt += f"{_name}的{d['locationCn']}"
                else:
                    txt += f"{d['locationCn']}"
        else:
            txt = f'{station_name}附近没有{device_name}'
            
        return txt

    def query_route(self, from_station, to_station):
        if self._check_station(from_station):
            return self._check_station(from_station)
        
        if self._check_station(to_station):
            return self._check_station(to_station)

        from_id = self.station_dict[from_station]
        to_id = self.station_dict[to_station]
        _url = f'{self.url_base}/rest/mtr/transfer?stationId={from_id}&toStationId={to_id}'
        try:
            route = requests.get(_url).json()['transfer']['routeCn']
            route_desc = json.loads(route)['desc']
        except KeyError:
            route_desc = API_ERROR
        return route_desc

    def query_ticket(self, from_station, to_station):
        if self._check_station(from_station):
            return self._check_station(from_station)
        
        if self._check_station(to_station):
            return self._check_station(to_station)

        from_id = self.station_dict[from_station]
        to_id = self.station_dict[to_station]
        _url = f'{self.url_base}/rest/mtr/ticketPrice?stationId={from_id}&toStationId={to_id}'
        try:
            price = requests.get(_url).json()['ticketPrice']
            return f'从{from_station}到{to_station}的地铁票价是{price}元'
        except KeyError:
            return API_ERROR
    

    def query_station_nearby(self, location):
        _url = f'{self.url_base}/rest/mtr/near?location={location}'
        desc = requests.get(_url).json()
        return desc
    
    
    def proceed_api_call(self, api_str):
        # find func call pattern
        p = f'({CALL_API}[\w|_]*\([^()]*\))'
        matched = re.findall(p, api_str)

        ans = api_str
        for func_str in matched:
            api_result = self._proceed_api_call(func_str.replace(CALL_API,''))
            ans = ans.replace(func_str, api_result)

        '''
        因为答案的format没考虑周全，训练数据中混入了格式混乱的数据。需要在这里硬处理一下。治本的方法把API返回的文本和template文案约定好标点符号的使用
        '''
        ans = ans.replace('。，','。')

        return ans

        # if '，' in api_str:
        #     api_strs = api_str.split('，')
        #     results = [self._proceed_api_call(s) for s in api_strs]
        #     text = results[0]
        #     for r in results[1:]:
        #         if r[-1] in ['，','。']:
        #             text += r
        #         else:
        #             text += '，'+r
        #     return text
        # else:
        #     return self._proceed_api_call(api_str)

    def _proceed_api_call(self, api_str):
        return eval(f'self.{api_str}')

if __name__ == '__main__':
    data = MetraData(reload=False)
    
    
    result = data.query_station_nearby('高价收废品')
    print(result)
    
    