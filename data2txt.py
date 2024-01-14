import numpy as np
import openpyxl
from matplotlib import pyplot as plt
def arr_to_excel(data,sheet):
    for row in data:
        sheet.append(row)
def list_to_excel(data,sheet):
    for item in data:
        sheet.append([item])

# 格式化输出--excel
def dim2_ouput_format_excel(index_x_y_cat,Tri):
    """
    get excel's file

    Args:
        index_x_y_cat (_type_): numpy array of four columns
        Tri (_type_): triangulation's results
    """
    # Todo  下标从1开始
    part1_points=index_x_y_cat[:,0:-1].tolist()
    part1_points=np.array(part1_points)
    part1_points[:,0]+=1
    print(part1_points.shape[0])
    z=np.zeros((part1_points.shape[0],1))
    # Todo 加一个z坐标

    part1_points=np.hstack((part1_points,z))
    part2_tri=[]
    # Todo 下标从1开始
    for i,tri in enumerate(Tri):
        new_tri=np.insert(tri, 0, i+1, axis=0)
        new_tri[1:] += 1
        new_tri=list(map(int, new_tri))
        part2_tri.append(new_tri)
    # if_inside=index_x_y_cat[:,-1]==2
    # if_outside=index_x_y_cat[:,-1]==1
    # if_border=index_x_y_cat[:,-1]==0
    # inside_p=index_x_y_cat[if_inside][:,0].tolist()
    # outside_p=index_x_y_cat[if_outside][:,0].tolist()
    # border_p=index_x_y_cat[if_border][:,0].tolist()
    # fixed_node=[]
    # load_node=[]
    # define the load and support 
    # for data in index_x_y_cat:
    #     if data[1]==3:
    #         if data[2]<=-0.8:
    #              load_node.append(int(data[0]))
    # for data in  index_x_y_cat:
    #     if data[1]==-1:
    #         if data[3]==1:
    #             fixed_node.append(int(data[0]))
            
    # load_node=[i+1 for i in load_node]
    # fixed_node=[i+1 for i in fixed_node]



    # part2_title=[7,'inside',int(len(inside_p))]
    # part3_title=[7,'outside',int(len(outside_p))]
    # part4_title=[7,'border',int(len(border_p))]

    tleft_nodes=[]
    top_nodes=[]
    tright_nodes=[]
    left_nodes=[]
    right_nodes=[]
    bleft_nodes=[]
    bottom_nodes=[]
    bright_nodes=[]
    pcd=index_x_y_cat[:,1:]
    # # 
    # for i,p in enumerate(pcd):
    #     j=i+1
    #     if p[1]==125 and abs(p[0])!=250:	
    #         top_nodes.append(j)
    #     if p[1]==125 and  p[0]==-250:
	#         tleft_nodes.append(j)
    #     if p[1]==125 and  p[0]==250:
	#         tright_nodes.append(j)
    #     if p[0]==-250 and abs(p[1])!=125:
	#         left_nodes.append(j)
    #     if p[0]==250 and abs(p[1])!=125:
	#         right_nodes.append(j)
    #     if p[1]==-125 and abs(p[0])!=250:
	#         bottom_nodes.append(j)
    #     if p[1]==-125 and p[0]==-250:
	#         bleft_nodes.append(j)
    #     if p[1]==-125 and p[0]==250:
	#         bright_nodes.append(j)
    # ========================fun3
    near=1e-5
    # for i,data in enumerate(pcd):
    #     if data[0]<=-0.8:
    #         outputs[i]=0.0
    #     if data[1]<=data[0]+0.3-near:
    #         outputs[i]=1.
    #     if data[1]>data[0]+0.3+near and data[1]<=data[0]+0.6+near:
    #         outputs[i]=0.0
    #     if data[0]>-0.2 and data[1]>data[0]+0.6-near:
    #         outputs[i]=1.
    min_lb=9999
    max_rt=-9999
    for p in pcd:
        if p[0]==-250 and p[-1]==0:
            if min_lb>p[1]:
                min_lb=p[1]
        if  p[1]==250 and p[-1]==0:
            if max_rt<p[0]:
                max_rt=p[0]
    for i,p in enumerate(pcd):
        j=i+1
        if p[0]>-250 and p[1]-p[0]-90>0 and (p[-1]==0 or (p[1]==250 and p[-1]==1)):
            top_nodes.append(j)
        if p[1]==250 and p[0]==-250:
	        tleft_nodes.append(j)
        if p[1]==250 and p[0]==max_rt:
            bright_nodes.append(j)
        if p[0]==-250 and p[1]>=min_lb and p[1]!=250:
            left_nodes.append(j)
        if p[0]==-250 and p[1]==min_lb:
            bleft_nodes.append(j)
        if (p[1]-p[0]-90<=0 and  p[0]!=-250 and p[0]!=max(pcd[:,0]) and p[-1]==0 and p[0]!=max_rt):
            bottom_nodes.append(j)
    top_nodes = np.array(top_nodes)
    tleft_nodes = np.array(tleft_nodes)
    bright_nodes = np.array(bright_nodes)
    left_nodes = np.array(left_nodes)
    bleft_nodes = np.array(bleft_nodes)
    bottom_nodes = np.array(bottom_nodes)
    # plt.figure()
    # plt.scatter(pcd[top_nodes-1, 0], pcd[top_nodes-1, 1], color='r', label='Top Nodes')
    # plt.scatter(pcd[tleft_nodes-1, 0], pcd[tleft_nodes-1, 1], color='g', label='Top Left Nodes')
    # plt.scatter(pcd[bright_nodes-1, 0], pcd[bright_nodes-1, 1], color='b', label='Bottom Right Nodes')
    # plt.scatter(pcd[left_nodes-1, 0], pcd[left_nodes-1, 1], color='c', label='Left Nodes')
    # plt.scatter(pcd[bleft_nodes-1, 0], pcd[bleft_nodes-1, 1], color='m', label='Bottom Left Nodes')
    # plt.scatter(pcd[bottom_nodes-1, 0], pcd[bottom_nodes-1, 1], color='y', label='Bottom Nodes')
    # plt.savefig("./contour.png")
    top_nodes = [[x] for x in top_nodes]
    tleft_nodes = [[x] for x in tleft_nodes]
    tright_nodes = [[x] for x in tright_nodes]
    left_nodes = [[x] for x in left_nodes]
    right_nodes = [[x] for x in right_nodes]
    bottom_nodes = [[x] for x in bottom_nodes]
    bleft_nodes = [[x] for x in bleft_nodes]
    bright_nodes = [[x] for x in bright_nodes]
    
    part2_title=['7',"top_nodes",int(len(top_nodes))]
    part3_title=['7',"tleft_nodes",int(len(tleft_nodes))]
    part4_title=['7',"tright_nodes",int(len(tright_nodes))]
    part5_title=['7',"left_nodes",int(len(left_nodes))]
    part6_title=['7',"right_nodes",int(len(right_nodes))]
    part7_title=['7',"bottom_nodes",int(len(bottom_nodes))]
    part8_title=['7',"bleft_nodes",int(len(bleft_nodes))]
    part9_title=['7',"bright_nodes",int(len(bright_nodes))]
    # part_fixed=[7,'fixed_node',int(len(fixed_node))]
    # part_load=[7,'load',int(len(load_node))]
    inside_tri=[]
    outside_tri=[]
    for i,tri in enumerate(Tri):
        flag=2
        index_tri=i+1
        for node in tri:
            # Todo 下标回去
            if(index_x_y_cat[int(node),-1]==2):
                inside_tri.append(index_tri)
                flag=1
                break
        if flag==2:
            outside_tri.append(index_tri)
    part1_title=[2,3,index_x_y_cat.shape[0],len(outside_tri)]
    outside_tri_i=[ part2_tri[i-1] for i in outside_tri]
    # part5_title=[8,"inside_tri",int(len(inside_tri))]
    # part6_title=[8,"outside_tri",int(len(outside_tri))]
    # 创建一个新的 Excel 工作簿
    workbook = openpyxl.Workbook()
    # 选择默认的工作表
    sheet = workbook.active
    sheet.append(part1_title)
    arr_to_excel(part1_points.tolist(),sheet)
    arr_to_excel(outside_tri_i,sheet)
    sheet.append(part2_title)
    arr_to_excel(top_nodes,sheet)
    sheet.append(part3_title)
    arr_to_excel(tleft_nodes,sheet)
    # sheet.append(part4_title)
    # arr_to_excel(tright_nodes,sheet)
    sheet.append(part5_title)
    arr_to_excel(left_nodes,sheet)
    # sheet.append(part6_title)
    # arr_to_excel(right_nodes,sheet)
    sheet.append(part7_title)
    arr_to_excel(bottom_nodes,sheet)
    sheet.append(part8_title)
    arr_to_excel(bleft_nodes,sheet)
    sheet.append(part9_title)
    arr_to_excel(bright_nodes,sheet)
    # sheet.append(part2_title)
    # list_to_excel(inside_p,sheet)
    # sheet.append(part3_title)
    # list_to_excel(outside_p,sheet)
    # sheet.append(part4_title)
    # list_to_excel(border_p,sheet)
    # sheet.append(part_fixed)
    # list_to_excel(fixed_node, sheet)
    # sheet.append(part_load)
    # list_to_excel(load_node, sheet)
    # sheet.append(part5_title)
    # list_to_excel(inside_tri,sheet)
    # sheet.append(part6_title)
    # list_to_excel(outside_tri,sheet)
    workbook.save('output_sheet.xlsx')
    excel2txt(sheet)
    correct_txt()

def excel2txt(sheet):
    txt_file = "./data/output.txt"  # 替换为您想要保存的文本文件路径
    with open(txt_file, "w") as txt_file:
        for row in sheet.iter_rows(values_only=True):
            row_values = [str(int(value)) if i==0  else str(value) for i,value in enumerate(row)]
            txt_file.write(' '.join(row_values) + '\n')
def correct_txt(path=None):
    txt_file = f"./data/output.txt"  # 替换为您想要保存的文本文件路径
    dat_file = f"./data/output.dat"
    old_txt=' None'
    new_txt=''
    with open(txt_file, "r") as tf:
        file_content = tf.read()
    
    modified_content = file_content.replace(old_txt, new_txt)
    
    with open(dat_file, "w") as tf:
        tf.write(modified_content)