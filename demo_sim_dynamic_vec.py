# -*- coding: utf-8 -*-
"""
动力学路径规划示例 (混合观测) + 向量化仿真
 Created on Wed Mar 13 2024 18:18:07
 Modified on 2024-3-13 18:18:07
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
#

if __name__ == "__main__": # gym.make_vec只能在main中运行
    # 1.环境实例化
    import path_plan_env # regist to gym first
    import gymnasium as gym
    from gymnasium.wrappers.vector import DictInfoToList

    env = gym.make_vec("PathPlan-v0", num_envs=3, max_time_steps=800, render_mode="human") # 动作空间本身就是 -1,1
    env = DictInfoToList(env)

    # 2.策略加载
    import onnxruntime as ort
    policy = ort.InferenceSession("./path_plan_env/policy_dynamic.onnx")

    # 3.仿真LOOP
    from copy import deepcopy

    MAX_STEPS = 4000
    obs, _ = env.reset(seed=list(range(env.num_envs)))
    for steps in range(MAX_STEPS):
        # 决策
        act = policy.run(['action'], obs)[0] # return [action, ...]
        # 仿真
        next_obs, _, _, _, info = env.step(act)
        print(f"step: {steps} | info: {info}")
        obs = deepcopy(next_obs)
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
