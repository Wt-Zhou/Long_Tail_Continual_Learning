import math
import os
import os.path as osp

import numpy as np
import torch
from rtree import index as rindex


class Results():
    def __init__(self, history_frame, create_new_train_file=False):

        if create_new_train_file:
            if osp.exists("DCP_results/trained_state_index.dat"):
                os.remove("DCP_results/trained_state_index.dat")
                os.remove("DCP_results/trained_state_index.idx")
            if osp.exists("DCP_results/trained_state.txt"):
                os.remove("DCP_results/trained_state.txt")

            self.all_state_list = []
            self.trained_state_counter = 0
            self.test_state_counter = 0
            print("Creat New Rtree")

        else:
            self.all_state_list = np.loadtxt("DCP_results/trained_state.txt").tolist()
            self.trained_state_counter = len(self.all_state_list)
            self.test_state_counter = 0
            self.reload_train_state_to_rtree(history_frame)
            print("Loaded Saved Rtree, len:",self.trained_state_counter)
            


        self.trained_state_format = " ".join(("%f",)*(history_frame * 20))+"\n"
        self.trained_state_outfile = open("DCP_results/trained_state.txt", "a")
        self.test_state_outfile = open("DCP_results/test_state.txt", "a")
        trained_state_tree_prop = rindex.Property()
        test_state_tree_prop = rindex.Property()
        trained_state_tree_prop.dimension = history_frame * 20 # 4 vehicles
        test_state_tree_prop.dimension = history_frame * 20 # 4 vehicles
        
        self.trained_state_dist = np.full(shape=history_frame * 20, fill_value=0.5)
        # self.trained_state_tree = rindex.Index('DCP_results/trained_state_index',properties=trained_state_tree_prop)
        self.test_state_tree = rindex.Index('DCP_results/test_state_index',properties=test_state_tree_prop)
        
        self.all_test_state_list = []
        self.visited_state_effiency_d = []
        self.visited_state_effiency_v = []
        self.visited_state_safety = []
        self.visited_state_conservative_level = []
        self.visited_state_performance = []
        self.visited_state_q_bound = []

    def calculate_visited_times(self, state):
        
        return sum(1 for _ in self.trained_state_tree.intersection(state.tolist()))
    
    def reload_train_state_to_rtree(self, history_frame):
        if osp.exists("DCP_results/trained_state_index.dat"):
            os.remove("DCP_results/trained_state_index.dat")
            os.remove("DCP_results/trained_state_index.idx")
        trained_state_tree_prop = rindex.Property()
        trained_state_tree_prop.dimension = history_frame * 20 
        self.trained_state_dist = np.full(shape=history_frame * 20, fill_value=0.5)
        self.trained_state_tree = rindex.Index('DCP_results/trained_state_index',properties=trained_state_tree_prop)
        
        reload_counter = 0
        for obs in self.all_state_list:
            obs = np.array(obs).flatten().tolist()
            self.trained_state_tree.insert(reload_counter,
            tuple((obs-self.trained_state_dist).tolist()+(obs+self.trained_state_dist).tolist()))
            reload_counter += 1
            print("reload_counter",reload_counter)


# 1. Counted Trained Dataset                    
    def add_training_data(self, his_obs_frames):
        obs = his_obs_frames
        obs = np.array(obs).flatten().tolist()
        self.all_state_list.append(obs)
        print("obs",obs)
        print("obs1",tuple((obs-self.trained_state_dist).tolist()+(obs+self.trained_state_dist).tolist()))

        self.trained_state_tree.insert(self.trained_state_counter,
            tuple((obs-self.trained_state_dist).tolist()+(obs+self.trained_state_dist).tolist()))
        self.trained_state_outfile.write(self.trained_state_format % tuple(obs))
        self.trained_state_counter += 1

    def calculate_training_distribution(self):
        self.mark_list = np.zeros(self.trained_state_counter)
        for i in range(self.trained_state_counter):
            if self.mark_list[i] == 0:
                state = self.all_state_list[i]
                str_state = str(state) # the state will change after rtree query
                # mark similar state
                trained_times = 0#sum(1 for _ in self.trained_state_tree.intersection(state)) #using sum from rtree would lead to repeat problem

                for n in self.trained_state_tree.intersection(state):
                    if self.mark_list[n] == 0:
                        self.mark_list[n] = 1
                        trained_times += 1
                    
                # print("train_data", trained_times)

                # write to txt
                with open("DCP_results/train_data.txt", 'a') as fw:
                    fw.write(str_state) 
                    fw.write(", ")
                    fw.write(str(trained_times)) 
                    fw.write("\n")
                    fw.close()               

# 2. Estimated Q-lower bound
    def sampled_trained_state(self, i):
        if i < self.trained_state_counter:
            trained_state = self.all_state_list[i]
            return trained_state
        else:
            return None
                
    def estimated_q_lower_bound(self, state, candidate_action_index, estimated_q_lower_bound, true_q_value):
        trained_times = sum(1 for _ in self.trained_state_tree.intersection(state)) 
        with open("DCP_results/confidence.txt", 'a') as fw:
                fw.write(str(state)) 
                fw.write(", ")
                fw.write(str(trained_times)) 
                fw.write(", ")
                fw.write(str(candidate_action_index)) 
                fw.write(", ")
                fw.write(str(estimated_q_lower_bound)) 
                fw.write(", ")
                fw.write(str(true_q_value)) 
                fw.write("\n")
                fw.close()   
                
                
# 3. Estimate DCP Performance
    def clear_old_test_data(self):
        if osp.exists("DCP_results/test_state_index.dat"):
            os.remove("DCP_results/test_state_index.dat")
            os.remove("DCP_results/test_state_index.idx")
        if osp.exists("DCP_results/test_state.txt"):
            os.remove("DCP_results/test_state.txt")
    
    def add_test_data(self, his_obs_frames, candidate_trajectories_tuple, trajectory, collision, estimated_q_lower_bound):    
        
        obs = his_obs_frames
        obs = np.array(obs).flatten().tolist()
        self.all_test_state_list.append(obs)
      
        self.test_state_tree.insert(self.test_state_counter,
            tuple((obs-self.trained_state_dist).tolist()+(obs+self.trained_state_dist).tolist()))
        self.test_state_outfile.write(self.trained_state_format % tuple(obs))
        self.test_state_counter += 1
        
        # safety metrics
        if collision:
            self.visited_state_safety.append(0)
        else:
            self.visited_state_safety.append(1)
            
        # effiency metrics
        trajectory_d = []
        trajectory_v = []
        for i in range(len(trajectory.x)):
            trajectory_d.append(math.fabs(trajectory.d[i]))
            trajectory_v.append(trajectory.s_d[i])
            
        self.visited_state_effiency_d.append(np.mean(trajectory_d))
        self.visited_state_effiency_v.append(np.mean(trajectory_v))
        
        # overall performance
        # not reward
        # performance = trajectory.cf - 500 * collision 
        # this is one_step_reward: the cost of werling
        Jp = trajectory.d_ddd[0]**2 
        Js = trajectory.s_ddd[0]**2

        # square of diff from target speed
        ds = (30.0 / 3.6 - trajectory.s_d[0])**2 # target speed

        cd = 0.1 * Jp + 0.1 * 0.1 + 0.05 * trajectory.d[0]**2
        cv = 0.1 * Js + 0.1 * 0.1 + 0.05 * ds
        performance = -1.0 * cd - 1.0 * cv - collision * 500
        
        self.visited_state_performance.append(performance)
        self.visited_state_q_bound.append(estimated_q_lower_bound)
        
        # conservative level
        sorted_fplist = sorted(candidate_trajectories_tuple, key=lambda candidate_trajectories_tuple: candidate_trajectories_tuple[1])
        min_cost = sorted_fplist[0][1]
        conservative_level = trajectory.cf/min_cost
        self.visited_state_conservative_level.append(conservative_level)               
        return None
    
    def calculate_performance_metrics(self):
        self.mark_list = np.zeros(self.test_state_counter)
        print("len",len(self.all_test_state_list))
        print("len",len(self.mark_list))
        print("self.test_state_counter",self.test_state_counter)
        for i in range(self.test_state_counter-1):
            if self.mark_list[i] == 0:
                state = self.all_test_state_list[i]
                # mark similar state
                state_effiency_v = 0
                state_effiency_d = 0
                state_safety = 0
                state_conservative_level = 0
                state_performance = 0
                state_q_bound = 0
                trained_times = sum(1 for _ in self.trained_state_tree.intersection(state)) #using sum from rtree would lead to repeat problem
                visited_times = 0
                for n in self.test_state_tree.intersection(state):
                    if self.mark_list[n] == 0:
                        state_effiency_d += self.visited_state_effiency_d[n]
                        state_effiency_v += self.visited_state_effiency_v[n]
                        state_safety += self.visited_state_safety[n]
                        state_conservative_level += self.visited_state_conservative_level[n]
                        state_performance += self.visited_state_performance[n]
                        state_q_bound += self.visited_state_q_bound[n]
                        self.mark_list[n] = 1
                        visited_times += 1
    
                state_effiency_d /= visited_times
                state_effiency_v /= visited_times
                state_safety /= visited_times
                state_conservative_level /= visited_times
                state_performance /= visited_times
                state_q_bound /= visited_times
                
                print("results", trained_times, state_performance, state_conservative_level, state_safety, state_effiency_d, state_effiency_v, visited_times)
                # write to txt
                with open("DCP_results/DCP_performance.txt", 'a') as fw:
                    fw.write(str(state)) 
                    fw.write(", ")
                    fw.write(str(trained_times)) 
                    fw.write(", ")
                    fw.write(str(state_q_bound)) 
                    fw.write(", ")
                    fw.write(str(state_performance)) 
                    fw.write(", ")
                    fw.write(str(state_conservative_level)) 
                    fw.write(", ")
                    fw.write(str(state_safety)) 
                    fw.write(", ")
                    fw.write(str(state_effiency_d)) 
                    fw.write(", ")
                    fw.write(str(state_effiency_v)) 
                    fw.write(", ")
                    fw.write(str(visited_times)) 
                    fw.write("\n")
                    fw.close()        
                               

# 4. Fixed State DCP Performance
    def record_dcp_performance(self, state, candidate_action_index, estimated_q_lower_bound, true_q_value, 
                               safety_rate, efficiency):
        trained_times = sum(1 for _ in self.trained_state_tree.intersection(state)) 
        print("estimated_q_lower_bound", estimated_q_lower_bound)
        print("true_q_value", true_q_value)
        print("safety_rate", safety_rate)
        print("efficiency", efficiency)
        print("trained_times", trained_times)
        with open("DCP_results/dcp_performance_fixed_state.txt", 'a') as fw:
                fw.write(str(state)) 
                fw.write(", ")
                fw.write(str(trained_times)) 
                fw.write(", ")
                fw.write(str(candidate_action_index)) 
                fw.write(", ")
                fw.write(str(estimated_q_lower_bound)) 
                fw.write(", ")
                fw.write(str(true_q_value)) 
                fw.write(", ")
                fw.write(str(safety_rate)) 
                fw.write(", ")
                fw.write(str(efficiency)) 
                fw.write("\n")
                fw.close()   
      

# Old functions       
    def add_data_for_real_time_metrics(self, his_obs_frames, trajectory, collision):    
        
        obs = his_obs_frames
        obs = np.array(obs).flatten().tolist()

        self.all_state_list.append(obs)
      
        self.visited_state_tree.insert(self.visited_state_counter,
            tuple((obs-self.visited_state_dist).tolist()+(obs+self.visited_state_dist).tolist()))
        # self.visited_state_tree.insert(self.visited_state_counter,
        #     tuple((obs-self.visited_state_dist).tolist()[0]+(obs+self.visited_state_dist).tolist()[0]))
        self.visited_state_outfile.write(self.visited_state_format % tuple(obs))
        self.visited_state_counter += 1
        
        # safety metrics
        if collision:
            self.visited_state_safety.append(0)
        else:
            self.visited_state_safety.append(1)
            
        # effiency metrics
        trajectory_d = []
        trajectory_v = []
        for i in range(len(trajectory.x)):
            trajectory_d.append(math.fabs(trajectory.d[i]))
            trajectory_v.append(trajectory.s_d[i])
            
        self.visited_state_effiency_d.append(np.mean(trajectory_d))
        self.visited_state_effiency_v.append(np.mean(trajectory_v))
        
        return None
    
    def add_data_for_dataset_prediciton_metrics(self, his_obs_frames, fde, ade):    
        
        obs = his_obs_frames
        obs = np.array(obs).flatten().tolist()
        
        self.all_state_list.append(obs)
        self.visited_state_tree.insert(self.visited_state_counter,
            tuple((obs-self.visited_state_dist).tolist()+(obs+self.visited_state_dist).tolist()))
        self.visited_state_outfile.write(self.visited_state_format % tuple(obs))
        self.visited_state_counter += 1
        
        self.prediction_ade.append(ade) 
        self.prediction_fde.append(fde) 
        
        return None
    
    def calculate_predition_results(self, dataset, predict_future_paths, history_frame):
        # run over the dataset
        for one_trajectory in dataset:
            vehicle_num = 0
            for i in range(len(one_trajectory[0])):
                if one_trajectory[0][i][0] != -999: # use -999 as signal, very unstable
                    vehicle_num += 1
            # get model prediction
            history_obs = one_trajectory[0:history_frame]
            for i, obs in enumerate(history_obs):
                if i < len(history_obs)-1:
                    predict_future_paths(obs, done=False)
                else:
                    paths_of_all_models = predict_future_paths(obs, done=False)
            
            ade = 0
            fde = 0
            for c in range(1,vehicle_num):
                min_ade_head = 99999
                min_fde_head = 99999
                for predict_path in paths_of_all_models:
                    if predict_path.c == c:
                        # ade - over a whole trajectory 
                        de_list = []
                        for k in range(len(predict_path.x)):
                            dx = predict_path.x[k] - one_trajectory[k+history_frame][predict_path.c][0]
                            dy = predict_path.y[k] - one_trajectory[k+history_frame][predict_path.c][1]
                            de_list.append(math.sqrt(dx**2 + dy**2))
                            # print("dx",dx,predict_path.x[k],one_trajectory[k+history_frame][predict_path.c][0])
                            # print("dy",dy,predict_path.y[k],one_trajectory[k+history_frame][predict_path.c][1])
                        if np.mean(de_list) < min_ade_head:
                            min_ade_head = np.mean(de_list)
                            
                        # print("len predict path",len(predict_path.x))
                        # print("len predict path",len(predict_path.y))
                        # print("len one_trajectory path",len(one_trajectory))
                        # print("min_ade_head",np.mean(de_list))
                        # print("min_ade_head",min_ade_head)
                        # fde - final point of trajectory 
                        d_x = predict_path.x[-1] - one_trajectory[-1][predict_path.c][0]
                        d_y = predict_path.y[-1] - one_trajectory[-1][predict_path.c][1]
                        # print("d_x",d_x,predict_path.x[-1],one_trajectory[-1][predict_path.c][0])
                        # print("d_y",d_y,predict_path.y[-1],one_trajectory[-1][predict_path.c][1])

                        if math.sqrt(d_x**2 + d_y**2) < min_fde_head:
                            min_fde_head = math.sqrt(d_x**2 + d_y**2)
                #         print("min_fde_head",math.sqrt(d_x**2 + d_y**2))
                #         print("min_fde_head",min_fde_head)
                # print("min_ade_head",min_ade_head)
                # print("min_fde_head",min_fde_head)

                ade += min_ade_head # divide by heads num
                fde += min_fde_head
                        
            ade /= vehicle_num
            fde /= vehicle_num
                        
            self.add_data_for_dataset_prediciton_metrics(history_obs, fde, ade)
                        
        # count the results from rtree
        self.mark_list = np.zeros(self.visited_state_counter)
        for i in range(self.visited_state_counter):
            if self.mark_list[i] == 0:
                state = self.all_state_list[i]

                visited_times = sum(1 for _ in self.visited_state_tree.intersection(state))
                # mark similar state
                state_ade = 0
                state_fde = 0
                for n in self.visited_state_tree.intersection(state):
                    state_ade += self.prediction_ade[n]
                    state_fde += self.prediction_fde[n]

                    self.mark_list[n] = 1
    
                state_ade /= visited_times
                state_fde /= visited_times
                
                print("results", visited_times, state_ade, state_fde)
                # write to txt
                with open("prediction_results.txt", 'a') as fw:
                    fw.write(str(state)) 
                    fw.write(", ")
                    fw.write(str(visited_times)) 
                    fw.write(", ")
                    fw.write(str(state_ade)) 
                    fw.write(", ")
                    fw.write(str(state_fde)) 
                    fw.write("\n")
                    fw.close()           
              
    def calculate_all_state_visited_time(self):
        self.mark_list = np.zeros(self.visited_state_counter)
        for i in range(self.visited_state_counter):
            if self.mark_list[i] == 0:
                state = self.all_state_list[i]
                # mark similar state
                state_effiency_v = 0
                state_effiency_d = 0
                state_safety = 0
                visited_times = 0#sum(1 for _ in self.visited_state_tree.intersection(state)) #using sum from rtree would lead to repeat problem

                for n in self.visited_state_tree.intersection(state):
                    if self.mark_list[n] == 0:
                        state_effiency_d += self.visited_state_effiency_d[n]
                        state_effiency_v += self.visited_state_effiency_v[n]
                        state_safety += self.visited_state_safety[n]
                        self.mark_list[n] = 1
                        visited_times += 1
    
                state_effiency_d /= visited_times
                state_effiency_v /= visited_times
                state_safety /= visited_times
                
                print("results", visited_times, state_effiency_d, state_effiency_v, state_safety)
                # print("state", state)
  
                # write to txt
                with open("results.txt", 'a') as fw:
                    fw.write(str(state)) 
                    fw.write(", ")
                    fw.write(str(visited_times)) 
                    fw.write(", ")
                    fw.write(str(state_safety)) 
                    fw.write(", ")
                    fw.write(str(state_effiency_d)) 
                    fw.write(", ")
                    fw.write(str(state_effiency_v)) 
                    fw.write("\n")
                    fw.close()               
      