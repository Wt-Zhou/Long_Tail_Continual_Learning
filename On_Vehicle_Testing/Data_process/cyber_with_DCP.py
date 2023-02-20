import math

from Agent.zzz.dynamic_map import DynamicMap, Lane, Lanepoint, Vehicle
from Agent.zzz.JunctionTrajectoryPlanner_simple_predict import \
    JunctionTrajectoryPlanner_SP

from cyber_record_process import Record_Process


class cyber_to_dcp():
    def __init__(self):
        None
        self.trajectory_planner = JunctionTrajectoryPlanner_SP()
        self.dynamic_map = DynamicMap()
        self.record_process = Record_Process()
        
    def generate_ref_path_from_record_traj(self, trajectory):
        ego_obs_list = trajectory[0]
        self.ref_path = Lane()
        for i in range(0,len(ego_obs_list)//100): # The Apollo record data is too dense!
            lanepoint = Lanepoint()
            lanepoint.position.x = ego_obs_list[i*90][0].position.x
            lanepoint.position.y = ego_obs_list[i*90][0].position.y
            self.ref_path.central_path.append(lanepoint)
            t_array.append(lanepoint)
        self.ref_path.central_path_array = np.array(t_array)
        self.ref_path.speed_limit = 60/3.6 # m/s
        self.dynamic_map.update_ref_path_from_routing(self.ref_path) 
        return None
    
    def get_obs(self, traj):
        obs_list = []
        for i in range(len(traj)):
            obs = []
            ego_pose = traj[0][i][0]
            ego_obs = [ego_pose.position.x, ego_pose.position.y, ego_pose.linear_velocity.x, ego_pose.linear_velocity.y, ego_pose.heading]
            obs.append(ego_obs)
            
            perception_obstacle = traj[1][i][0]
            for one_obs in message.perception_obstacle:
                obs.append([one_obs.position.x, one_obs.position.y, one_obs.velocity.x, one_obs.velocity.y, one_obs.theta])
                
            obs_list.append(obs)
        return obs_list
    
    
    def circle(self, files):
        for file in files:
            recorded_trajectory = self.record_process.get_trajectory_from_file(file)
            self.generate_ref_path_from_record_traj(recorded_trajectory)
            for obs in self.get_obs(recorded_trajectory):
                candidate_trajectories_tuple, rule_index = self.generate_candidate_trajectories(obs)
                predicted_results = self.generate_ensemble_prediction_results(obs)
                self.update_prediction_ensembles(obs)
                
                long_tail_rate = self.calculated_long_tail_rate(candidate_trajectories_tuple, predicted_results)
                
                
    
    def generate_candidate_trajectories(self, obs):
        
        self.dynamic_map.update_map_from_list_obs(obs)
            
        trajectory_action, rule_index = self.trajectory_planner.trajectory_update(self.dynamic_map)
        candidate_trajectories_tuple = self.trajectory_planner.generate_candidate_trajectories(self.dynamic_map)

        chosen_action_id = rule_index
        rule_trajectory = self.trajectory_planner.trajectory_update_CP(chosen_action_id)

        return candidate_trajectories_tuple, rule_index
    
    def generate_ensemble_prediction_results(self, obs):
        predicted_results = []
        return predicted_results
    
    def calculated_long_tail_rate(self, candidate_traj, predicted_results):
        long_tail_rate = []
        for action in candidate_traj:
            q_ego = action[1]
            q_value = 0
            long_tail_rate.append(q_ego+q_value)
        return long_tail_rate
    
    def update_prediction_ensembles(self, obs):
        return None
    
    


if __name__ == '__main__':
    folder_path = "/home/zwt/apollo/data/bag"  
    offline_dcp = cyber_to_dcp()
    
    files = offline_dcp.record_process.get_files(folder_path)
    files = sorted(files)
    offline_dcp.circle(files)
    