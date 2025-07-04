import csv
import os
import time

import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random

#Probe Tip Match
input_image = cv2.imread('result/nucleus_boundary/path_to_original_image.jpg')
tempate_image = cv2.imread('Probe.tif')
coordinate_filex = "box_image/probe_coordinates.csv"
#Get template matching results
result = cv2.matchTemplate(input_image,tempate_image,5)

#Parse results

#Minimum, Maximum, Location of Minimum, Location of Maximum
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
#template height, width, channel
h,w ,channels = tempate_image.shape
#Get probe tip center coordinates
cv2.rectangle(input_image,maxLoc,(maxLoc[0]+w,maxLoc[1]+h),(255,0,0),5)
centroid_x = int(maxLoc[0]+(w / 2))
centroid_y = int(maxLoc[1]+(h / 2))
centroid = (int(centroid_x), int(centroid_y))

coordinates = []

coordinates.append([0, centroid_x, centroid_y])
#Save coordinate information to CSV file
with open(coordinate_filex, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'X', 'Y'])  # header
    writer.writerows(coordinates)
coordinates_file = "box_image/coordinates.csv"

#define function, add smallest rectangle box, determine nucleus centroid coordinates and save
def add_min_rect_with_centroid(binary_image_path,target_image_path):

    binary_image = cv2.imread(binary_image_path, 0)
    target_image=cv2.imread(target_image_path)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coordinates = []

    i = 0
    for contour in contours:

        area = cv2.contourArea(contour)
        #Remove rectangular boxes smaller than 20x20 and filter out misrecognized interference
        if area < 10 * 10:
            continue
        if area > 400:
            i+=1
        #Calculate centroid coordinates
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            centroid = (centroid_x, centroid_y)
            # Draw the nucleus cross centroid

            # Draw horizontal lines

            cv2.line(target_image, (centroid[0] - 10, centroid[1]), (centroid[0] + 10, centroid[1]), (0, 255, 0), 5)
            #Draw vertical lines
            cv2.line(target_image, (centroid[0], centroid[1] - 10), (centroid[0], centroid[1] + 10), (0, 255, 0), 5)
            #Add coordinate point information to the list
            coordinates.append([i, centroid_x, centroid_y])

    cv2.waitKey(0)
    cv2.imwrite('box_image/box_image.tif', target_image)
    #Save coordinate information to CSV file
    with open(coordinates_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'X', 'Y'])  # 表头
        writer.writerows(coordinates)
    cv2.destroyAllWindows()


binary_image_path = "result/binary_image.jpg"
#Draw all the information on this picture
target_image_path = "result/nucleus_boundary/path_to_original_image.jpg"
#Label centroid coordinates for binary images


add_min_rect_with_centroid(binary_image_path,target_image_path)

"#############################  calculates position information  ###################################"

#Get every coordinate
def get_coordinates_from_file(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first line
        for row in reader:
            coordinate = [int(row[1]), int(row[2])]
            coordinates.append(coordinate)
    return coordinates
#Calculate coordinate differences
def calculate_interpolated_coordinate(coord1, coord2, factor):
    x_diff = int((coord2[0] - coord1[0])*factor)
    y_diff = int((coord2[1] - coord1[1])*factor)
    return x_diff, y_diff

image_path = 'box_image/box_image.tif'
#Tip table file path
file1_path = 'box_image/probe_coordinates.csv'
#Nucleus table file path
file2_path = 'box_image/coordinates.csv'

image = cv2.imread(image_path)
coords1 = get_coordinates_from_file(file1_path)
coords2 = get_coordinates_from_file(file2_path)

#Mark probe tip and nucleus center

for coord1 in coords1:

    cv2.circle(image, tuple(coord1), radius=10, color=(0, 255, 255), thickness=-1)
    cv2.line(image, (centroid[0] - 10, centroid[1]), (centroid[0] + 10, centroid[1]), (0, 255, 255), 5)
    cv2.line(image, (centroid[0], centroid[1] - 10), (centroid[0], centroid[1] + 10), (0, 255, 255), 5)

    # for coord2 in coords2:
    #
    #     interpolated_x, interpolated_y = calculate_interpolated_coordinate(coord1, coord2, -0.2)
    #     #Mark the coordinate difference above the cell coordinates
    #     cv2.putText(image, f"({interpolated_y}, {interpolated_x})um", (coord2[0]-50, coord2[1]-40),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    #
    #     image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(image_pil)
        # Loading Calibri fonts
        # font_path = '/full/path/to/calibri.ttf'
        # font = ImageFont.truetype(font_path, 40)
        #text = f"({interpolated_y}, {interpolated_x}) um"
        #position = (coord2[0] - 70, coord2[1] - 70)

cv2.waitKey(0)
cv2.imwrite('box_image/box_image_result.tif',image)
cv2.destroyAllWindows()

"#############################  Delineating the detectable range of probe tip  ###################################"

df = pd.read_csv('box_image/probe_coordinates.csv')
x, y = df.iloc[0]['X'], df.iloc[0]['Y']

image_box = cv2.imread('box_image/box_image_result.tif')

half_side = 250
top_left = (int(x - half_side), int(y - half_side))
bottom_right = (int(x + half_side), int(y + half_side))
#Draw detectable range rectangle on picture
cv2.rectangle(image_box, top_left, bottom_right, (0, 255, 255), 4)

#cv2.imshow('Image with rectangle', image_box)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('box_image/box_image_result.tif', image_box)


"####################  Save the distance coordinates of all cells within the detectable range  ############################"

df1 = pd.read_csv('box_image/probe_coordinates.csv')
df2 = pd.read_csv('box_image/coordinates.csv')
coordinate_count = 0
sumCell = 0
x1, y1 = df1.iloc[0]['X'], df1.iloc[0]['Y']

result = pd.DataFrame(columns=['X', 'Y'])

for index, row in df2.iterrows():
    x2, y2 = row['X'], row['Y']
    #Calculate the difference between abscissa and ordinate
    delta_x = x2 - x1
    delta_y = y2 - y1
    #Determine whether the absolute values of the differences between horizontal and vertical coordinates are less than 250
    if abs(delta_x) < 250 and abs(delta_y) < 250:
        if abs(delta_x) > 20 or abs(delta_y) > 20:
            #Calculate the actual distance from the center of each cell nucleus to the probe tip

            # Actual distance per pixel
            factor = -0.02e-5
            delta_x_new = float (delta_x * factor)
            delta_y_new = float (delta_y * factor)
            coordinate_count += 1
            sumCell += coordinate_count
            result = result.append({
                'X': delta_y_new,
                'Y': delta_x_new
            }, ignore_index=True)

result.to_csv('box_image/output_table.csv', index=False)
file_path = r"C:\Users\17105\Desktop\data-transmit\1.tif.tif"
if file_path :
    result.to_csv(r'C:\Users\17105\Desktop\data-transmit\output_table.csv', index=False)
    time.sleep(5)
    #os.remove(file_path)
    #print("The distance information of cells within the detectable range of the probe tip has been saved to 'D:/output_table.csv'")
#print("The distance information of cells within the detectable range of the probe tip has been saved to 'box_image/output_table.csv'")
#print(f"Total number of coordinates: {coordinate_count}")

# "#########################    Looking for the next detectable area   #################################"
#
# df3 = pd.read_csv('box_image/output_table.csv')
# x1 = df1['X'].values[0]
# y1 = df1['Y'].values[0]
# x2 = df2['X'].values
# y2 = df2['Y'].values
#
# candidates = []
# for i in range(len(x2)):
#     if x2[i]>0 and y2[i]>0:
#         #Check whether the absolute value of the difference between the abscissa and ordinate of the point in Table 1 is greater than 600
#         if 2000>abs(x2[i] - x1) > 300 or 2000>abs(y2[i] - y1) > 300:
#             #print(f"point({x2[i]}, {y2[i]}) meets the condition of distance to probe tip")
#             #Check if there are at least three other points around this point, the absolute value of the abscissa and ordinate difference is less than 250
#             count = 0
#             for k in range(len(x2)):
#                 if i != k and abs(x2[i] - x2[k]) < 250 and abs(y2[i] - y2[k]) < 250:
#                     count += 1
#
#             if count >= 3:  #There are at least 3 points nearby, plus at least 4 points on its own
#                 #print(f"Point({x2[i]}, {y2[i]})  has {count - 1} neighbors that match the criteria.")
#                 candidates.append(i)
#
# if candidates:
#     # Randomly select a point that matches the criteria
#     chosen_idx = random.choice(candidates)
#     chosen_x, chosen_y = x2[chosen_idx], y2[chosen_idx]
#
#     #Output selected points
#     print(f"The selected point coordinates are: ({chosen_x}, {chosen_y})")
#
#
#     factor = -0.02e-5
#     diff_x = float((chosen_x - x1) * factor)
#     diff_y = float((chosen_y - y1) * factor)
#     img = Image.open('box_image/box_image_result.tif')
#     plt.figure(figsize=(10, 10))
#
#     #Draw dots on pictures
#     plt.imshow(img)
#     plt.scatter(x2, y2, c='orange', label='Table 2 Points')
#     plt.scatter([chosen_x], [chosen_y], c='green', label='next area Point', s=100)
#     plt.legend()
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Chosen Point Visualization on Image')
#     plt.grid(True)
#     plt.axis('off')
#     plt.savefig('box_image/chosen_point_visualization.tif', bbox_inches='tight', pad_inches=0)
#     #plt.show()
#
#     #Save the calculated actual distance coordinate results to the table
#     #Add selected points to end of detectable table
#     new_row = pd.DataFrame({'X': [diff_y], 'Y': [diff_x]})
#     #Append new rows directly to specified DataFrame and save
#     df3 = df3.append(new_row, ignore_index=True)
#     df3.to_csv('box_image\output_table.csv', index=False)
#     df3.to_csv(r'C:\Users\17105\Desktop\数据传输\output_table.csv', index=False)
#     print("Calculation results have been added to the specified table and saved as box_image/output_table.csv")
# else:
#     print("No matching points found")
#
# print(f"The selected point coordinates are: ",sumCell)