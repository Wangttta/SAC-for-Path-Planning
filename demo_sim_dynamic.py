# -*- coding: utf-8 -*-
"""
动力学路径规划示例 (混合观测)
 Created on Wed Mar 13 2024 18:18:07
 Modified on 2024-3-13 18:18:07
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
# 泛化性测试：改变地图
from shapely import geometry as geo
from path_plan_env.env import MAP
MAP.obstacles = [
    geo.Point(4.5, 3.5).buffer(2.5),
    geo.Point(0, 2.5).buffer(2),
    geo.Point(-6, -5).buffer(3.5),
    geo.Point(6, -5).buffer(2),
    geo.Polygon([(-10, 0), (-10, 5), (-5, 5), (-7.5, 0)])
]


# 1.环境实例化
import path_plan_env # regist to gym first
import gymnasium as gym

env = gym.make("PathPlan-v0", max_time_steps=800, render_mode="human") # 动作空间本身就是 -1,1


# 2.策略加载
import onnxruntime as ort
policy = ort.InferenceSession("./path_plan_env/policy_dynamic.onnx")


# 3.仿真LOOP
from copy import deepcopy

MAX_EPISODE = 5
for episode in range(MAX_EPISODE):
    ## 获取初始观测
    time_steps = 0
    obs, _ = env.reset()
    ## 进行一回合仿真
    while True:
        # 决策
        seq_points = obs['seq_points'].reshape(1, *obs['seq_points'].shape) # (1, seq_len, *points_shape, )
        seq_vector = obs['seq_vector'].reshape(1, *obs['seq_vector'].shape) # (1, seq_len, vector_dim, )
        act = policy.run(['action'], {'seq_points': seq_points, 'seq_vector': seq_vector})[0] # return [action, ...]
        act = act.flatten()                                                                   # (1, dim, ) -> (dim, )
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
