import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py import functions
import mujoco_py
from collections import deque

class RaceEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    #state = np.zeros(shape=15, dtype = float32)
    def __init__(self):
        self.state = np.zeros(shape=19, dtype=np.float32)
        self.vel = np.zeros(shape=19, dtype=np.float32)
        self.num_agents =2
        self.mem =10
        self.GOAL_REWARD=10000
        self.ant_done = [False]*self.num_agents              
        #self.old_ant_pos=[[[0,0]]*self.num_agents]*10
        self.mem_ant_pos =deque([[[0,0]]*self.num_agents]*self.mem,maxlen=self.mem)

        self.mapping = {"torso_geom": 0, "aux_1_geom":1, "front_left_leg_geom":2,
           "front_left_ankle_geom":3, "aux_2_geom":4, "front_right_leg_geom":5,
           "front_right_ankle_geom":6, "aux_3_geom":7, "back_left_leg_geom":8,
           "back_left_ankle_geom":9, "aux_4_geom":10, "back_right_leg_geom":11,
           "back_right_ankle_geom":12,
           "torso_geom2":13, "aux_1_geom2":14, "front_left_leg_geom2": 15,
         "front_left_ankle_geom2": 16, "aux_2_geom2": 17, "front_right_leg_geom2": 18,
         "front_right_ankle_geom2": 19, "aux_3_geom2": 20, "back_left_leg_geom2": 21,
         "back_left_ankle_geom2": 22, "aux_4_geom2": 23, "back_right_leg_geom2": 24,
         "back_right_ankle_geom2": 25,"platform": 39, "line1": 40,
         "line2":41, "line3": 42, "line4": 43, "line5": 44, "line6": 45,

         }

        mujoco_env.MujocoEnv.__init__(self, "./race.xml", 5)
        utils.EzPickle.__init__(self)
        

    def step(self, actions,n=1):
        self.do_simulation(actions, 5)
        
        self.num_agents= n
        num_agents=n
        name=["torso","torso2"]
        xbound=[-2,2]
        ybound=[-1.5,10]
        y_check =[2.5,5,10,15,20]
        self.ant_pos=[None]*num_agents
        self.ant_vel =[None]*num_agents

        self.ant_ang = [None]*num_agents
        index_ant=[None]*num_agents
        self.reward=[0]*num_agents


        for i in range(num_agents):

            index_ant[i]=self.model.body_name2id(name[i])
            self.ant_pos[i]=self.sim.data.body_xpos[index_ant[i]]
            self.ant_vel[i] = self.sim.data.body_xvelp[index_ant[i]] 

            R = self.sim.data.body_xmat[index_ant[i]].reshape((3, 3))

            # Get the normal vector of the plane the object is resting on
            plane_normal = np.array([0, 0, 1])

            # Compute the pitch and roll angles from the rotation matrix
            pitch = np.arctan2(R[0, 2], np.sqrt(R[0, 0]**2 + R[0, 1]**2)) * 180 / np.pi
            roll = np.arctan2(-R[1, 2], R[2, 2]) * 180 / np.pi

            # Compute the orientation of the object relative to the plane it is resting on
            obj_normal = np.dot(R, np.array([0, 0, 1]))
            cos_theta = np.dot(obj_normal, plane_normal) / (np.linalg.norm(obj_normal) * np.linalg.norm(plane_normal))
            self.ant_ang[i] = np.arccos(cos_theta) * 180 / np.pi

                
            if not self.ant_done[i]:
                #print(self.ant_pos[0],self.ant_pos[i][0]<xbound[0] or self.ant_pos[i][0]>xbound[1] or self.ant_pos[i][1]<ybound[0])
                if self.ant_pos[i][0]<xbound[0] or self.ant_pos[i][0]>xbound[1] or self.ant_pos[i][1]<ybound[0]:
                    #print(self.ant_pos[i],self.ant_pos[i-1],self.ant_done)
                    self.ant_done[i]=True
                    self.reward[i] += -200


                if self.ant_ang[i] >120:
                    self.ant_done[i]=True
                    self.reward[i] += -500


        #print(self.ant_pos[0],self.ant_vel[0])
        agentids=np.arange(num_agents)
        for i in range(num_agents):
            if not self.ant_done[i]:
                self.reward[i] += 100*self.ant_vel[i][1] -5+ 10*(self.ant_pos[i][1] )#- self.mem_ant_pos[0][i][1])
                ydiff= self.ant_pos[i][1] - self.ant_pos[i-1][1]

                for y in y_check:
                    if self.ant_pos[i][1] >=y:
                        self.reward[i] += 10#self.GOAL_REWARD//2
                        #self.reward[i-1] += -self.GOAL_REWARD

                # if ydiff >0:
                #     #print(ydiff)
                #     self.reward[i] +=ydiff*11

                # else:
                #     self.reward[i] -=ydiff*9

                if self.ant_pos[i][1] >=ybound[1]:
                    print(i,"reached")
                    self.reward[i] += self.GOAL_REWARD
                    for j in np.delete(agentids,i):

                        self.reward[j] += -self.GOAL_REWARD

                    self.ant_done =[True]*num_agents

        self.mem_ant_pos.append(self.ant_pos)
        ob = self._get_obs()  
        return ob, self.reward, self.ant_done, {}

    

    def _get_obs(self):

        retn_obv = np.concatenate([ant_pos.flat for ant_pos in self.ant_pos ]+[ant_vel.flat for ant_vel in self.ant_vel ]+
        
            # contact_forces
            #[self.link_force().flat, ## Remove it out for this topic

            # 3-dim position and velocity
            [self.sim.data.qpos.flat[0:],
            self.sim.data.qvel.flat[0:],

        ])
        return retn_obv

        #获取6-aixs接触力
    def link_force(self):
        force = np.zeros(shape=(52,6), dtype = np.float64)
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            c_array = np.zeros(6, dtype=np.float64)
            functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
            force[self.mapping[self.sim.model.geom_id2name(contact.geom1)]] = c_array[:6]
            force[self.mapping[self.sim.model.geom_id2name(contact.geom2)]] = c_array[:6]
        return force[:52].flat.copy()

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        self.ant_done = [False]*self.num_agents
        return self._get_obs()

        #摄像机视角
    def viewer_setup(self):
        if self.viewer is not None:
            self.viewer._run_speed = 0.5
            self.viewer.cam.trackbodyid = 0
            # self.viewer.cam.lookat[2] += .8
            self.viewer.cam.elevation = -25
            self.viewer.cam.type = 1
            self.sim.forward()
            self.viewer.cam.distance = self.model.stat.extent * 1.0
        # Make sure that the offscreen context has the same camera setup
        if self.sim._render_context_offscreen is not None:
            self.sim._render_context_offscreen.cam.trackbodyid = 0
            # self.sim._render_context_offscreen.cam.lookat[2] += .8
            self.sim._render_context_offscreen.cam.elevation = -25
            self.sim._render_context_offscreen.cam.type = 1
            self.sim._render_context_offscreen.cam.distance = \
                self.model.stat.extent * 1.0
        self.buffer_size = (1280, 800)
