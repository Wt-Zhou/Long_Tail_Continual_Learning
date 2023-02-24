import csv
import math
import os
import time

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from cyber_record.record import Record


class Record_Process():
    def __init__(self):
        None


    def parse_pose(self, file_name):
        '''
        save pose to csv file
        '''
        f = open("pose.csv", 'a')
        writer = csv.writer(f)
        
        header = ["timestamp_sec", "x", "y","vx","vy","ax","ay","heading","angular_vx","angular_vy"]  
        writer.writerow(header)
        
        record = Record(file_name)
        print("file_name",file_name)
        
        for topic, message, t in record.read_messages('/apollo/localization/pose'):
            line = [message.header.timestamp_sec, message.pose.position.x, message.pose.position.y, message.pose.linear_velocity.x, message.pose.linear_velocity.y, 
                           message.pose.linear_acceleration.x, message.pose.linear_acceleration.y, message.pose.heading, message.pose.angular_velocity.x, 
                           message.pose.angular_velocity.y]
            writer.writerow(line)

        f.close()
        
    def plot_pose(self, files):
        
        x = []  
        y = []  
        ds = 0
        dt = 0
        
        for file_name in files:
            last_x = 0
            last_y = 0
            last_t = 0
            try:
                record = Record(file_name)
                print("file_name",file_name)
                
                for topic, message, t in record.read_messages('/apollo/localization/pose'):
                    x.append(message.pose.position.x)
                    y.append(message.pose.position.y)
                    
                    if last_x != 0:
                        ds += math.sqrt((message.pose.position.x-last_x)**2+(message.pose.position.y-last_y)**2)
                    if last_t != 0:
                        dt += message.header.timestamp_sec - last_t
                    last_x = message.pose.position.x
                    last_y = message.pose.position.y
                    last_t = message.header.timestamp_sec
            except:
                pass

        # plt.plot(x, y)
        # plt.xlabel("X Label")
        # plt.ylabel("Y Label")
        # plt.autoscale()
        # plt.gca().set_aspect(1.0) 
        # plt.gca().set_facecolor('none')
        # plt.show()
        print("total s",ds) # all bags:117318.01273395626, 107114.84702897705
        print("total t",dt) # all bags:430256.8142709732, 21905.34637284279

        fig = plt.figure(facecolor='none', edgecolor='none')
        ax = fig.add_subplot(111,facecolor='none')
        ax.scatter(x, y, marker='o', facecolors='none', edgecolors='black', s=0.1)

        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.axis('off')
        
        plt.gca().set_aspect(1.0) 
        plt.show()

    def plot_info_with_time(self, files, steps):
        v = []
        a = []
        for file_name in files:
            try:
                record = Record(file_name)
                print("file_name",file_name)
                
                for topic, message, t in record.read_messages('/apollo/localization/pose'):
                    if len(v) <= steps:
                        v.append(math.sqrt(message.pose.linear_velocity.x**2+message.pose.linear_velocity.y**2))
                        a.append(math.sqrt(message.pose.linear_acceleration.x**2+message.pose.linear_acceleration.y**2))
                    else:
                        break
            except:
                print("bug")
                pass
        plt.plot(range(len(v)), v, 'o', color='black', markersize=0.1)
        plt.plot(range(len(a)), a, 'o', color='blue', markersize=0.1)
        plt.autoscale()
        plt.gca().set_facecolor('none')
        # plt.gca().set_aspect(1.0) 
        plt.show()
        
    def plot_long_tail(self, files, steps):
        step = 0
        num_items = 500  # 最多统计的障碍物数量
        obs_num_count = [0] * num_items
        obs_v_car_count = [0] * num_items
        obs_v_bike_count = [0] * num_items
        obs_v_people_count = [0] * num_items
        obs_v_other_count = [0] * num_items
        obs_a_count = [0] * num_items
        for file_name in files:
            if step < steps:
                try:
                    record = Record(file_name)
                    print("file_name",file_name)
                    
                    for topic, message, t in record.read_messages('/apollo/perception/obstacles'):
                        obs_num_count[len(message.perception_obstacle)] += 1
                        
                        v_car_list = []
                        v_bike_list = []
                        v_people_list = []
                        v_other_list = []
                        a_list = []
                        
                        for one_obs in message.perception_obstacle:
                            if one_obs.type == 5:
                                v_car_list.append(math.sqrt(one_obs.velocity.x**2+one_obs.velocity.y**2))
                            elif one_obs.type == 3:
                                v_people_list.append(math.sqrt(one_obs.velocity.x**2+one_obs.velocity.y**2))
                            elif one_obs.type == 4:
                                v_bike_list.append(math.sqrt(one_obs.velocity.x**2+one_obs.velocity.y**2))
                            else:
                                v_other_list.append(math.sqrt(one_obs.velocity.x**2+one_obs.velocity.y**2))
                            a_list.append(math.sqrt(one_obs.acceleration.x**2+one_obs.acceleration.y**2))
                            
                        if len(v_car_list) > 0:
                            v_car = int(sum(v_car_list)/len(v_car_list)*10)
                            obs_v_car_count[v_car] += 1
                        if len(v_people_list) > 0:
                            v_people = int(sum(v_people_list)/len(v_people_list)*10)
                            obs_v_people_count[v_people] += 1
                        if len(v_bike_list) > 0:
                            v_bike = int(sum(v_bike_list)/len(v_bike_list)*10)
                            obs_v_bike_count[v_bike] += 1
                        if len(v_other_list) > 0:
                            v_other = int(sum(v_other_list)/len(v_other_list)*10)
                            obs_v_other_count[v_other] += 1
                        if len(a_list) > 0:
                            a = int(sum(a_list)/len(a_list)*50)
                            obs_a_count[a] += 1
                        
                        step += 1
                except:
                    print('bugs')
                    pass

        print("total_step",step)  
        print("obstacle_count",obs_num_count)   
        print("obs_v_car_count",obs_v_car_count)   
        print("obs_v_bike_count",obs_v_bike_count)   
        print("obs_v_people_count",obs_v_people_count)   
        print("obs_v_other_count",obs_v_other_count)   
        print("obs_a_count",obs_a_count)  
        
        with open('output_50000.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['obs_num_count', 'obs_v_car_count','obs_v_bike_count','obs_v_people_count','obs_v_other_count', 'obs_a_count'])
            for i in range(num_items):
                writer.writerow([obs_num_count[i], obs_v_car_count[i],obs_v_bike_count[i],obs_v_people_count[i],obs_v_other_count[i], obs_a_count[i]]) 
        
    def parse_scenario_info(self, file_name):
        '''
        save scenario information to csv file
        '''
        f = open("0216_pose.csv", 'a')
        writer = csv.writer(f)
        
        header = ["timestamp_sec", "x", "y","vx","vy","ax","ay","heading","angular_vx","angular_vy"]  
        writer.writerow(header)
        
        record = Record(file_name)
        for topic, message, t in record.read_messages('/apollo/localization/pose'):
            line = [message.header.timestamp_sec, message.pose.position.x, message.pose.position.y, message.pose.linear_velocity.x, message.pose.linear_velocity.y, 
                           message.pose.linear_acceleration.x, message.pose.linear_acceleration.y, message.pose.heading, message.pose.angular_velocity.x, 
                           message.pose.angular_velocity.y]
            writer.writerow(line)

        f.close()
 
    def get_files(self, path):
        files = []
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                files.append(os.path.abspath(file_path))
            elif os.path.isdir(file_path):
                files += self.get_files(file_path)
        return files

    def plot_a_case(self):
        file_name = "/home/zwt/apollo/data/bag/0215/0215_afternoon_1/20230215143531.record.00004" 
        start_time = 1676443213.579 * 10**9
        end_time=1676443223.579 * 10**9

        
        record = Record(file_name)
        
        v = []
        steer = []
        brake = []
        throttle = []
        a = []
        ego_x = []
        ego_y = []
        obs_x = []
        obs_y = []
        obs_x2 = []
        obs_y2 = []
        obs_x3 = []
        obs_y3 = []
        
        i = 0
        for topic, message, t in record.read_messages('/apollo/localization/pose', \
                                                        start_time=start_time, end_time=end_time):   #100Hz
        
            if i % 10 == 0:
                ego_x.append(message.pose.position.x)
                ego_y.append(message.pose.position.y)
                v.append(math.sqrt(message.pose.linear_velocity.x**2 + message.pose.linear_velocity.y**2))
                a.append(math.sqrt(message.pose.linear_acceleration.x**2 + message.pose.linear_acceleration.y**2))
            i += 1
            
        for topic, message, t in record.read_messages('/apollo/perception/obstacles', \
                                                        start_time=start_time, end_time=end_time):  #100Hz
            for obs in message.perception_obstacle:
                if obs.id == 3903:                    
                    obs_x.append(obs.position.x)
                    obs_y.append(obs.position.y)

            
            
        for topic, message, t in record.read_messages('/apollo/canbus/chassis', \
                                                        start_time=start_time, end_time=end_time):   #10Hz
            steer.append(message.steering_percentage)
            brake.append(message.brake_percentage)
            throttle.append(message.throttle_percentage)

        # font_path = '/usr/share/fonts/SimHei.ttf'
        font_path = '/usr/share/fonts/simsun.ttc'
        prop = fm.FontProperties(fname=font_path)

        obs_x = [x-ego_x[0] for x in obs_x]
        obs_x2 = [x-ego_x[0] for x in obs_x2]
        obs_x3 = [x-ego_x[0] for x in obs_x3]
        ego_x = [x-ego_x[0] for x in ego_x]
        obs_y = [y-ego_y[0] for y in obs_y]
        obs_y2 = [y-ego_y[0] for y in obs_y2]
        obs_y3 = [y-ego_y[0] for y in obs_y3]
        ego_y = [y-ego_y[0] for y in ego_y]

        # plot_variable
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['axes.unicode_minus'] = False 
        
        plt.plot([i/10 for i in range(len(v))], v, 'o-', color='black', markersize=1)
        plt.autoscale()
        plt.gca().set_facecolor('none')
        plt.xlabel('时间(s)',fontsize=13)
        plt.ylim(0, max(v))
        plt.ylabel('速度(m/s)',fontsize=13)
        plt.gca().set_aspect(0.75) 
        plt.show()
    
        plt.plot([i/10 for i in range(len(a))], a, 'o-', color='black', markersize=1)
        plt.autoscale()
        plt.gca().set_facecolor('none')
        plt.xlabel('时间(s)',fontsize=13)
        plt.ylim(0, max(a))
        plt.ylabel('加速度(m/s$^2$)',fontsize=13)
        plt.gca().set_aspect(0.75) 
        plt.show()
        
        plt.plot([i/100 for i in range(len(steer))], steer, 'o-', color='black', markersize=1)
        plt.autoscale()
        plt.gca().set_facecolor('none')
        plt.xlabel('时间(s)',fontsize=13)
        plt.ylabel('方向盘转角($^{\circ}$)',fontsize=13)
        plt.ylim(min(steer)-10, max(steer)+10)
        plt.gca().set_aspect(0.1) 
        plt.show()
        
        # plot trajectory
        fig = plt.figure(facecolor='none', edgecolor='none')
        ax = fig.add_subplot(111,facecolor='none')
        ax.scatter(ego_x, ego_y, marker='o', facecolors='none', edgecolors='blue', s=10,  label='无人驾驶车辆')
        ax.scatter(obs_x, obs_y, marker='o', facecolors='none', edgecolors='red', s=10,  label='主要环境交通参与者')
        # ax.scatter(obs_x2, obs_y2, marker='o', facecolors='none', edgecolors='orange', s=10,  label='主要环境交通参与者2')
        # ax.scatter(obs_x3, obs_y3, marker='o', facecolors='none', edgecolors='yellow', s=10,  label='主要环境交通参与者3')
        
        ax.set_xlabel('X轴坐标(m)',fontsize=13)
        ax.set_ylabel('Y轴坐标(m)',fontsize=13)

        ax.set_xlim(min(ego_x)-0, max(ego_x)+10)
        ax.set_ylim(min(ego_y)-10, max(ego_y)+20)

        # ax.axis('off')
        ax.legend(loc='upper right', prop=prop)

        plt.gca().set_aspect(1.0) 
        plt.show()

    def get_trajectory_from_file(self, file_name):
        # file_name = "/home/zwt/apollo/data/bag/0217/20230217143216.record.00001"   
        record = Record(file_name)
    
        env_obs_list = []
        ego_obs_list = []
        action_list = []
        trajectory = []
            
        for topic, message, t in record.read_messages('/apollo/planning'):  
            action_list.append([message,t])
        print("action_list",len(action_list))
        
        i = 0
        for topic_perception, message_perception, t_perception in record.read_messages('/apollo/perception/obstacles'):
            last_t = action_list[i][1]
            if t_perception > last_t - 0.2* 10**9 and t_perception < last_t:
                perception_obstacle = message_perception.perception_obstacle
                env_obs_list.append([perception_obstacle,t_perception])
                i += 1
            if i >= len(action_list):
                break
        
        i = 0
        for topic_pose, message_pose, t_pose in record.read_messages('/apollo/localization/pose'):
            last_t = action_list[i][1]
            if t_pose > last_t - 0.1* 10**9 and t_pose < last_t:
                pose = message_pose.pose
                ego_obs_list.append([pose,t_pose])
                i += 1
            if i >= len(action_list):
                break
        
        
        trajectory = [ego_obs_list,env_obs_list,action_list]           
        print("ego_obs_list",len(ego_obs_list))
        print("env_obs_list",len(env_obs_list))

        return trajectory

if __name__ == '__main__':
    folder_path = "/home/zwt/apollo/data/bag"  
    record_process = Record_Process()    

    files = record_process.get_files(folder_path)
    files = sorted(files)

    message_name = '/apollo/localization/pose'
    
    # record_process.plot_pose(["/home/zwt/apollo/data/bag/0215/0215_morning/20230215104439.record.00001"])
    # record_process.plot_info_with_time(files,990000)
    record_process.plot_a_case()

    # for file_name in files:
    # record_process.plot_long_tail(files, 50000)
    
