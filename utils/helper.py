import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from minigrid.wrappers import ImgObsWrapper


def convert_minigrid_to_text_S3R1(env, obs):

    grid = env.unwrapped.grid  # Access raw grid data
    grid_text = ""
    
    agent_x, agent_y = env.unwrapped.agent_pos  # Get agent's position
    agent_dir = env.unwrapped.agent_dir  # Get agent's direction

    # print(f"Agent is at ({agent_x}, {agent_y}) facing {agent_dir}") 

    mat = [[None for _ in range(7)] for _ in range(3)]
    
    for j in range(0,3):
        for i in range(0,7):

            cell = grid.get(i, j)
            if cell is None:
                grid_text += ". "
                mat[j][i] = None
            else:
                grid_text += cell.type.upper() + " "
                mat[j][i]=cell.type.upper()
                #color of the cell?
                # print(cell.color)
                # print(f"{cell}")
        grid_text += "\n"

    # print(mat)

    grid_text = ""
    
    if(agent_dir == 3): ## if its left

        x = agent_y
        y = agent_x 
        for i in range(0,3):
            for j in range(0,7):
                if (j+1 == y or j==y):
                    if mat[i][j] == None:
                        grid_text += ". "
                    else:
                        grid_text += mat[i][j] + " "
                else : 
                    grid_text += "? "
            grid_text += "\n"

    elif (agent_dir == 1): ## if its right

        x = agent_y
        y = agent_x

        for i in range(0,3):
            for j in range(0,7):
                if (j-1 == y or j==y):
                    if mat[i][j] == None:
                        grid_text += ". "
                    else:
                        grid_text += mat[i][j] + " "
                else : 
                    grid_text += "? "
            grid_text += "\n"

    elif (agent_dir == 2): ## if its up
        x = agent_y
        y = agent_x

        for i in range(0,3):
            for j in range(0,7):
                if ((i==0 or i==1) and (abs(j-y) <= 1)):
                    if mat[i][j] == None:
                        grid_text += ". "
                    else:
                        grid_text += mat[i][j] + " "
                else : 
                    grid_text += "? "
            grid_text += "\n"

    elif (agent_dir == 0): ## if its down
        x = agent_y
        y = agent_x

        for i in range(0,3):
            for j in range(0,7):
                if ((i==2 or i==1) and (abs(j-y) <= 1)):
                    if mat[i][j] == None:
                        grid_text += ". "
                    else:
                        grid_text += mat[i][j] + " "
                else : 
                    grid_text += "? "
            grid_text += "\n"

    print(grid_text)

    return grid_text
