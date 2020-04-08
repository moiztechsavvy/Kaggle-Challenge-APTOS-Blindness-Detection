# TF records was a No go because of the Addtional memory space that it required.
#path_tfrecords_train = os.path.join(train_image_path, "train.tfrecords")
# print(path_tfrecords_train)


# defining global variable path
# train_image_path = 'data/preprocessed_train'
# test_image_path = 'data/test_preprocessed'
#Load Images as Paths so They Can Be accessed on the Go.
# def loadImages(path):
#     image_files = sorted([os.path.join(path, file) for file in os.listdir(path) if file.endswith('.png')])
#     return image_files

# train_image_path_array = loadImages(train_image_path)
# test_data_image_path_array = loadImages(test_image_path)


# import csv
# import random

# with open ('aptostrain.csv', mode='r') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
    
#     list0 = []
#     list1 = []
#     list2 = []
#     list3 = []
#     list4 = []
    
#     #take in values from csv file and sort into lists based on value in second column
#     for row in csv_reader:
#         if line_count == 0:
#             #print(", ".join(row))
#             line_count += 1
#         else:
#             val0 = row[0]
#             val = row[1]
#             if (val == "0"):
#                 list0.append(val0)
#             elif (val == "1"):
#                 list1.append(val0)
#             elif (val == "2"):
#                 list2.append(val0)
#             elif (val == "3"):
#                 list3.append(val0)
#             elif (val == "4"):
#                 list4.append(val0)
#             line_count += 1
            
            
#     #take in user input to determine number of items to choose from each list
#     #need to add checks here
#     num_items = int(input("Enter number of items to choose (193 or less): "))
    
#     #choose a random sample of the number of random items chosen by user from each list
#     l_0 = random.sample(list0, num_items)
#     l_1 = random.sample(list1, num_items)
#     l_2 = random.sample(list2, num_items)
#     l_3 = random.sample(list3, num_items)
#     l_4 = random.sample(list4, num_items)
    
#     with open('list.csv', 'w') as file:
#         while True:
#             if l_0 != []: file.write(','.join([l_0.pop(), '0\n']))
#             if l_1 != []: file.write(','.join([l_1.pop(), '1\n']))
#             if l_2 != []: file.write(','.join([l_2.pop(), '2\n']))
#             if l_3 != []: file.write(','.join([l_3.pop(), '3\n']))
#             if l_4 != []: file.write(','.join([l_4.pop(), '4\n']))
#             if l_0 == [] and l_1 == [] and l_2 == [] and l_3 == [] and l_4 == []:
#                 break

#     print("Line count: " + str(line_count))


#for Some Reason Image shows Up as Blue.
# input_image = cv2.imread(path_to_image)
# plt.imshow(input_image)
# path_to_image = "data/train_images/{}.png".format(list1[2])

# np.random.seed(43)

# plt.figure(figsize=(10,10))
# for i in range(25):
#   file_path = "data/train_images/{}.png".format(train_df.iloc[0,i])
#   input_image = cv2.imread(file_path,1)
#   plt.subplot(5,5,i+1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.grid(True)
#   plt.imshow(np.arange(input_image).reshape(input_image.shape[0], input_image.shape[1])).all()
#   # The CIFAR labels happen to be arrays, 
#   # which is why you need the extra index
#   plt.xlabel(df.iloc[1,i])
# plt.show()

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

# In[3]:


# def morph_func(img):
# 	#uint8 is an unsigned integer from 0 to 255
# 	img=img.astype(np.uint8)
	
# 	#axis=2 for 3d image
# 	findblack=np.sum(img,axis=2)
# 	print(np.min(findblack))
# 	findblack=findblack-np.min(findblack)
# 	findblack[findblack>np.mean(findblack)]=np.mean(findblack)
# 	findblack=findblack/np.max(findblack)
# 	borders=2+np.sum(findblack[:,:int(img.shape[1]/2)]<0.33,axis=1)
	
# 	for i in range(img.shape[0]):
# 		k=borders[i]
# 		img[i]=img[i,np.linspace(k,img.shape[1]-k,num=img.shape[1]).astype(int),:]
	
# 	return img

# def crop_image(img, resize_width=299, resize_height=299):
#     #Convert to black and gray and threshold
#     output = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret,gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

#     #RETR_EXTERNAL finds only extreme outer contours
#     #CHAIN_APPROX_SIMPLE compresses segments leaving only the end points
#     gray, contours,hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     #this catches any images that are too dark. I wasn't able to find any examples to test this though
#     if not contours:
#         # print('No contours! Image is too dark')
#         flag = 0
#         return img, flag
#     #find the largest contour
#     cnt = max(contours, key = cv2.contourArea)

#     #Get center of circle and radius
#     ((x, y), r) = cv2.minEnclosingCircle(cnt)
#     x = int(x)
#     y = int(y)
#     r = int(r)

#     #Get height and width of original image and divide by 2
#     height = int(np.size(img, 0)/2)
#     width = int(np.size(img, 1)/2)

#     #if the circle is bigger than the image, return resized original. else crop and then resize
#     dim = (resize_width,resize_height)
#     r=int(r * 0.8)
#     if(r > width and r > height):
#         #output = increase_brightness(output)
#         return cv2.resize(output, dim, interpolation=cv2.INTER_AREA)

#     else:
#         if(r > height):
#             output = output[:,max(x-r,0):x+r]
#         elif(r > width):
#             output = output[max(y-r,0):y+r,:]
#         else:
#             output = output[max(y-r,0):y+r,max(x-r,0):x+r]
#         #output = increase_brightness(output)
#         return cv2.resize(output, dim, interpolation=cv2.INTER_AREA)



# train_df = pd.read_csv('data/train.csv')
# test_df = pd.read_csv('data/testLabels15.csv')
# print("Size of Training Dataset",train_df.shape)
# print("Size of Test Dataset",test_df.shape)


# num_classes = train_df['diagnosis'].unique().size
# print("Total Number of classes =", num_classes)

# def display_samples(df, columns=4, rows=3):
#     fig=plt.figure(figsize=(5*columns, 4*rows))

#     for i in range(columns*rows):
#         image_path = df.loc[i,'id_code']
#         image_id = df.loc[i,'diagnosis']
#         img = cv2.imread('./data/preprocessed/{}.png'.format(image_path))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #         img = crop_image(img)
        
#         fig.add_subplot(rows, columns, i+1)
#         plt.title(image_id)
#         plt.imshow(img)
    
#     plt.tight_layout()

# train_images = processing(image_path_array, 2048,100,100) 
# test_images =  processing(test_data_image_path_array, 200,100,100) 

# # b = image_path_array[:len(train_images)]
# # id_code = b[0][18:-4]
# # print(id_code)
# train_classes = train_df['diagnosis'][:2048]
# test_classes = test_df['level'][:200]

# print(len(train_classes))
