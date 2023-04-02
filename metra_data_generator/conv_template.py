from metra_api import MetraData
import pandas as pd
import random
import os,re

metra = MetraData(reload=False)

def gen_template_data(count=3000):
    templates = []
    template_dir = 'raw_data/templates'
    for f in os.listdir(template_dir):
        templates.extend(open(os.path.join(template_dir, f)).read().split('----'))

    convs = []
    var_mapping = {
        '<<station_name2>>':list(metra.station_dict.keys()),
        '<<to_station2>>':list(metra.station_dict.keys()),
        '<<line_name>>':list(metra.line_station_names.keys()),
        '<<station_name3>>':list(metra.station_dict.keys()),
        '<<line_name1>>':list(metra.line_station_names.keys()),
        '<<device_name>>':list(metra.device_dict.keys()),
        '<<device_name1>>':list(metra.device_dict.keys()),
        '<<device_name2>>':list(metra.device_dict.keys()),
        '<<to_station>>':list(metra.station_dict.keys()),
        '<<station_name>>':list(metra.station_dict.keys()),
        '<<station_name1>>':list(metra.station_dict.keys()),
        '<<from_station>>':list(metra.station_dict.keys()),
        '<<from_station2>>':list(metra.station_dict.keys()),
    }
    
    for _ in range(count):
        template = random.choice(templates)
        convs.append(format_template(template, var_mapping))

    assert '<<' not in [c.replace('<<CALL_API>>','') for c in convs]

    return convs


def template_factory(template_path, var_mapping,diag_cnt = 200):
    templates = open(template_path).read().split('----')
    diags = []
    for t in templates:
        diags.extend(
            [format_template(t, var_mapping) for _ in range(diag_cnt)]
        )
    return diags

def format_template(tem, var_mapping):
    tem = tem.strip()
    diag = []
    
    # decide and fix var value for the whole diag
    var_values = {}
    for k, v in var_mapping.items():
        var_values[k] = random.choice(v)

    for line in tem.split('\n'):
        assert any(t in line for t in ['客户：', '悠悠：']), f'error line: {line}'
        role, body = line.split('：')

        line_candidates = body.strip().split('|')
        _line = random.choice(line_candidates)
        _line = f'{role}：{_line}'
        for k, v in var_values.items():
            _line = _line.replace(k, v)
        diag.append(_line)
    return '\n'.join(diag)

if __name__ == '__main__':
    data = gen_template_data()
    print(data[100])