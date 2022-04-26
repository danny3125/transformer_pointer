import cnc_input
#import img_index
import numpy as np
import matplotlib.pyplot as plt
import torch
class input_handler:
    def __init__(self, jsonfilename):
        self.target_metrices = cnc_input.main(['-i', jsonfilename])
        self.X_all = []
        self.X_central = []
        self.waiting_time_range = 10
        self.visit_time_range = 4
        self.cornershape = 4
        self.dim_of_point = 2
        self.num_rec = 34
    def zig_zag_path(self,path_corners_index,mask_list_num): #path corners index = [[start_corner_index, end_corner_index], ....] = array 2d (path_lengh,2)
        mapping_tool = []
        for i in range(1,len(mask_list_num)):
            for j in range(mask_list_num[i]-mask_list_num[i-1]):
                mapping_tool.append(i-1)
        #now I have the mapping tool, it's time to map it again to a corner list [0,1] => [0,0,0,0,1,1,1,1]
        path_gazebo = []
        path_corners = []
        self.X_all = input_handler.every_point(self)
        for index in path_corners_index:
            path_corners.extend([self.X_all[self.cornershape*mapping_tool[int(index[0]/self.cornershape)]+(index[0]%self.cornershape)],self.X_all[self.cornershape*mapping_tool[int(index[1]/self.cornershape)]+(index[1]%self.cornershape)]])

        data = np.array(path_corners)
        plt.plot(data[:, 0], data[:, 1],color = 'black')
        data_1 = np.array(self.X_all)
        data_1 = np.reshape(data_1,(self.num_rec,self.cornershape,self.dim_of_point))
        for rec in data_1:
            rec = np.concatenate((rec,[rec[0]]),axis= 0)
            plt.plot(rec[:, 0], rec[:, 1],color = 'red')
        plt.show()
        '''
        for index in path_corners_index: #find the longer side => zig-zag to end point
            corner_num = index[0] % 4
            if (abs(self.X_all[index[0]][0] - self.X_all[index[1]][0])) > (abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])): #if longer side = horizon side = row side
                if corner_num == 0 :
                    y_way = range(0,int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),int(self.target_metrices[1]))
                    x_way_left = range(0,len(self.target_metrices[0][int(index[0] / 4)][0][0]),int(self.target_metrices[1]))
                    x_way_right = range(len(self.target_metrices[0][int(index[0] / 4)][0][0]),0,-int(self.target_metrices[1]))
                elif corner_num == 3:  # start = left down ,out = left up
                    y_way = range(0,-int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])), -int(self.target_metrices[1]))
                    x_way_left = range(0,len(self.target_metrices[0][int(index[0] / 4)][0][0]),int(self.target_metrices[1]))
                    x_way_right = range(len(self.target_metrices[0][int(index[0] / 4)][0][0]),0,-int(self.target_metrices[1]))
                elif corner_num == 1: #start = right up, out = right down
                    y_way = range(0,int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),int(self.target_metrices[1]))
                    x_way_left = range(0,-len(self.target_metrices[0][int(index[0] / 4)][0][0]),-int(self.target_metrices[1]))
                    x_way_right = range(-len(self.target_metrices[0][int(index[0] / 4)][0][0]),0,int(self.target_metrices[1]))
                else: # corner_num == 2: #start = right down, out = right up
                    y_way = range(0,-int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])), -int(self.target_metrices[1]))
                    x_way_left = range(0,-len(self.target_metrices[0][int(index[0] / 4)][0][0]),-int(self.target_metrices[1]))
                    x_way_right = range(-len(self.target_metrices[0][int(index[0] / 4)][0][0]),0,int(self.target_metrices[1]))
                # x_way_left = when the agent is on the left side , then it should move to the right side
                way_2 = x_way_left
                way_1 = y_way
                turn = 0
                for i in way_1:
                    turn += 1
                    for j in way_2:
                        path_gazebo.append([self.X_all[index[0]][0]+j,self.X_all[index[0]][1]+i])
                    if (turn % 2) == 1:
                        way_2 = x_way_right
                    else:
                        way_2 = x_way_left
            else:                                       # long side = straight side = column
                if corner_num == 0 : # lu -> ?
                    y_way = range(0,len(self.target_metrices[0][int(index[0] / 4)][0]),int(self.target_metrices[1]))
                    x_way_left = range(0,int(abs(self.X_all[index[0]][0] - self.X_all[index[1]][0])),int(self.target_metrices[1]))
                    x_way_right = range(int(abs(self.X_all[index[0]][0] - self.X_all[index[1]][0])),0,-int(self.target_metrices[1]))
                elif corner_num == 3:  # start = left down ,out = ?
                    y_way = range(len(self.target_metrices[0][int(index[0] / 4)][0]),0, -int(self.target_metrices[1]))
                    x_way_left = range(0,-int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),-int(self.target_metrices[1]))
                    x_way_right = range(-int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),0,int(self.target_metrices[1]))
                elif corner_num == 1: # start = right up, out = right down
                    y_way = range(0,-len(self.target_metrices[0][int(index[0] / 4)][0]),-int(self.target_metrices[1]))
                    x_way_left = range(0,int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),int(self.target_metrices[1]))
                    x_way_right = range(int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),0,-int(self.target_metrices[1]))
                else: #corner_num == 2: # start = right down, out = right up
                    y_way = range(0,-len(self.target_metrices[0][int(index[0] / 4)][0]), -int(self.target_metrices[1]))
                    x_way_left = range(0,-int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),-int(self.target_metrices[1]))
                    x_way_right = range(-int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),0,int(self.target_metrices[1]))
                # x_way_left = when the agent is on the left side , then it should move to the right side
                way_2 = x_way_left
                way_1 = y_way
                turn = 0
                for i in way_1:
                    turn +=1
                    for j in way_2:
                        path_gazebo.append([self.X_all[index[0]][0]+j,self.X_all[index[0]][1]+i])
                    if (turn % 2) == 1:
                        way_2 = x_way_right
                    else:
                        way_2 = x_way_left
        '''
        return path_gazebo



    def A_walkonchip(self): # unused
        allsteps = []
        for matrix in self.target_metrices:
            allsteps.extend((input_handler.sides_in_matrix(matrix[0], matrix[1], matrix[2])))
        return allsteps


    def sides_in_matrix(matrix, startpoint_x, startpoint_y): #unused
        sides_of_matrix = ((startpoint_x, startpoint_y), (startpoint_x + len(matrix), startpoint_y) \
                               , (startpoint_x, startpoint_y + len(matrix[0]),
                                  (startpoint_x + len(matrix), startpoint_y + len(matrix[0]))))
        '''
        for i in range(1, len(matrix),6):
            for j in range(1,len(matrix[0]),6):
                walkpoints.append((startpoint_x + i, startpoint_y + j))
        return walkpoints'''
        return sides_of_matrix

    def every_point(self):
        self.X_all = []
        for matrix in self.target_metrices[0]:
            rectangle = matrix[0]
            x_lu = matrix[1]
            y_lu = matrix[2]
            x_ru = x_lu + len(rectangle[0])
            y_ru = y_lu
            x_ld = x_lu
            y_ld = y_lu + len(rectangle)
            x_rd = x_ru 
            y_rd = y_ld 
            
            self.X_all.extend([[x_lu,y_lu],[x_ru,y_ru],[x_rd,y_rd],[x_ld,y_ld]])
        return self.X_all
    def baseline_points(self):#map_center should be a tuple that represents the split point of ROIs
        #the baseline mode of data structure, which is just adding an extra axis to the point, that
        # is, if a point is going to be visited twice, just split it into two different points, with
        #different time stamps
        mask_list_num = [0]
        self.X_all = input_handler.every_point(self)
        self.X_central = input_handler.central_point(self)
        visited_time_count = 0
        output = []
        waiting_time_list = []
        reshape_tool = np.asarray(self.X_all) # X_all.shape = (len(X_all),2)
        reshape_tool = reshape_tool.reshape(((int(len(self.X_all)/self.cornershape)),self.cornershape,\
                                             self.dim_of_point))

        for i in range(len(reshape_tool)):    # 4*2
        # decide which region a rectangle should be
            #decide how much time a point should be waited until next time it can be visited
            np.random.seed(i)
            visited_time = np.random.choice(range(1,self.visit_time_range),1).tolist()[0]
            visited_time_count += visited_time
            mask_list_num.append(visited_time_count)
            #decide how many times a point should be visited
            waiting_time = self.waiting_time_range/2#np.random.choice(range(1,self.waiting_time_range),1).tolist()[0]
            for j in range(visited_time):
                reshape_temp = np.insert(reshape_tool[i],self.dim_of_point,waiting_time*j,axis=1)
                output.extend(reshape_temp.tolist())
                if j > 0:
                    waiting_time_list.extend([waiting_time]*self.cornershape)
                else:
                    waiting_time_list.extend([0]*self.cornershape)
            
        return output,mask_list_num,waiting_time_list

    def central_point(self):
        self.X_central = []
        for matrix in self.target_metrices[0]:
            rectangle = matrix[0]
            x_lu = matrix[1]
            y_lu = matrix[2]
            self.X_central.extend([[x_lu + 0.5*len(rectangle),y_lu + 0.5*len(rectangle[0])]])
        return self.X_central
    # using barrier_avoid to let the agent take a movement between decisions
    # the barrier points should be different from normal tsp points, they should be loaded in another way, and be considered in another way 
    #def barrier_avoid(self, recent_points):


    def outcorner_getout(self,rectangle_inf,B,mask_list_num):# horizontal line = row
        feature = torch.Tensor([])
        mapping_tool = []
        # is odd? is row?
        for i in range(1,len(mask_list_num)):
            for j in range(mask_list_num[i]-mask_list_num[i-1]):
                mapping_tool.append(i-1)
        for inf in rectangle_inf:
            which_rec = mapping_tool[int(inf)]
            rectangle = self.target_metrices[0][which_rec][0]
            index = 4*int(inf)
            corner = inf - int(inf)
            if len(rectangle) > len(rectangle[0]): # is column the long side?
                short_side = len(rectangle[0])
                if (int(short_side / self.target_metrices[1]) % 2 == 0): # take even times to spray
                    if corner == 0: #is a left up corner
                        feature = torch.cat((feature,torch.Tensor([index + 1,index + 2,index + 3])),0)#append [outcorner,getout_corners]
                    elif corner == 0.25: # is a right up
                        feature = torch.cat((feature,torch.Tensor([index + 0,index + 2,index + 3])),0)
                    elif corner == 0.5: #is a right down
                        feature = torch.cat((feature,torch.Tensor([index + 3,index + 0,index + 1])),0)
                    else:               # is a left down
                        feature = torch.cat((feature,torch.Tensor([index + 2,index + 0,index + 1])),0)
                else:    #take odd time to spray
                    if corner == 0: #is a left up corner
                        feature = torch.cat((feature,torch.Tensor([index + 2,index + 1,index + 3])),0) #append [outcorner,getout_corners]
                    elif corner == 0.25: # is a right up 
                        feature = torch.cat((feature,torch.Tensor([index + 3,index + 0,index + 2])),0)
                    elif corner == 0.5: #is a right down
                        feature = torch.cat((feature,torch.Tensor([index + 0,index + 1,index + 3])),0)
                    else:               # is a left down
                        feature = torch.cat((feature,torch.Tensor([index + 1,index + 0,index + 2])),0)
                        
            else:
                short_side = len(rectangle) #row = long side
                if (int(short_side / self.target_metrices[1]) % 2 == 0): # take even times to spray
                    if corner == 0: #is a left up corner, outcorner = left down
                        feature = torch.cat((feature,torch.Tensor([index + 3,index + 1,index + 2])),0) #append [outcorner,getout_corners]
                    elif corner == 0.25: # is a right up , out = right down
                        feature = torch.cat((feature,torch.Tensor([index + 2,index + 0,index + 3])),0)
                    elif corner == 0.5: #is a right down, out = right up
                        feature = torch.cat((feature,torch.Tensor([index + 1,index + 0,index + 3])),0)
                    else:               # is a left down, out = left up
                        feature = torch.cat((feature,torch.Tensor([index + 1,index + 0,index + 2])),0)
                else:
                    if corner == 0: #is a left up corner
                        feature = torch.cat((feature,torch.Tensor([index + 2,index + 1,index + 3])),0) #append [outcorner,getout_corners]
                    elif corner == 0.25: # is a right up 
                        feature = torch.cat((feature,torch.Tensor([index + 3,index + 0,index + 2])),0)
                    elif corner == 0.5: #is a right down
                        feature = torch.cat((feature,torch.Tensor([index + 0,index + 1,index + 3])),0)
                    else:               # is a left down
                        feature = torch.cat((feature,torch.Tensor([index + 1,index + 0,index + 2])),0)     
        feature = torch.reshape(feature,(B,3))                
        return feature    
          
            
        
'''
test = cnc_input.main(['-i', 'right_chip.json'])
print(test[9][0])
plt.imshow(test[9][0], cmap=plt.get_cmap('gray'))
A_walkonchip(test)
print(package_points(test))
'''
