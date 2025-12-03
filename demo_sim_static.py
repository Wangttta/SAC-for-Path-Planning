# -*- coding: utf-8 -*-
"""
路径搜索示例 (单观测)
 Created on Wed Mar 13 2024 18:18:07
 Modified on 2024-3-13 18:18:07
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
#

# 1.环境实例化
from path_plan_env import NormalizedActionsWrapper # regist to gym first
import gymnasium as gym

env = gym.make("PathSearch-v0", max_search_steps=50, render_mode="human")
env = NormalizedActionsWrapper(env)


# 2.策略加载
import onnxruntime as ort
policy = ort.InferenceSession("./path_plan_env/policy_static.onnx")


# 3.仿真LOOP
from copy import deepcopy

MAX_EPISODE = 20
for episode in range(MAX_EPISODE):
    ## 获取初始观测
    time_steps = 0
    obs, _ = env.reset()
    ## 进行一回合仿真
    while True:
        # 决策
        obs = obs.reshape(1, *obs.shape)                      # (*shape, ) -> (1, *shape, )
        act = policy.run(['action'], {'observation': obs})[0] # return [action, ...]
        act = act.flatten()                                   # (1, dim, ) -> (dim, )
        # 仿真
        time_steps += 1
        next_obs, _, done, timeout, info = env.step(act)
        # 回合结束
        if done or timeout:
            print('回合: ', episode,'| 状态: ', info,'| 步数: ', time_steps) 
            break
        else:
            obs = deepcopy(next_obs)
    #end for
#end for




#             ⠰⢷⢿⠄
#         ⠀⠀⠀⠀⠀⣼⣷⣄
#         ⠀⠀⣤⣿⣇⣿⣿⣧⣿⡄
#         ⢴⠾⠋⠀⠀⠻⣿⣷⣿⣿⡀
#         🏀   ⢀⣿⣿⡿⢿⠈⣿
#          ⠀⠀⢠⣿⡿⠁⢠⣿⡊⠀⠙
#          ⠀⠀⢿⣿⠀⠀⠹⣿
#           ⠀⠀⠹⣷⡀⠀⣿⡄
#            ⠀⣀⣼⣿⠀⢈⣧ 
#
#       你。。。干。。。嘛。。。
#       哈哈。。唉哟。。。