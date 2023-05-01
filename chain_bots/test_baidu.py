
# -*- coding: utf-8 -*
import wenxin_api # 可以通过"pip install wenxin-api"命令安装
from wenxin_api.tasks.free_qa import FreeQA
wenxin_api.ak = "qQIgeaZAcQzmgKO5iMkiSoV0KQxfTGGU" #输入您的API Key
wenxin_api.sk = "9XIV8uwomhoBM9HblN1ckt6IpAp4yuTq" #输入您的Secret Key
input_dict = {
    "text": "问题：写一首关于春天的七言绝句？回答：",
    "seq_len": 512,
    "topp": 0.5,
    "penalty_score": 1.2,
    "min_dec_len": 2,
    "min_dec_penalty_text": "。?：！[<S>]",
    "is_unidirectional": 0,
    "task_prompt": "qa",
    "mask_type": "paragraph"
}
rst = FreeQA.create(**input_dict)
print(rst)