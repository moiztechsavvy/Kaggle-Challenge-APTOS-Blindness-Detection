


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

np.random.seed(43)

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