import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("final_yolov3_training_last.weights", "yolov3_testing_2classes.cfg")
classes = ["plastic bottle","can"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
#cap = cv2.VideoCapture(0) #webcam
cap = cv2.VideoCapture('marijns_video3.mp4')

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
property_id = int(cv2.CAP_PROP_FRAME_COUNT)
length = int(cv2.VideoCapture.get(cap, property_id))
previous_time = time.time()
previous_frame =[[],[]]
second_previous_frame = [[],[]]
numbers =[0,0]
while True:
    if frame_id < length:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        _, frame = cap.read()
        frame_id += 10
        #print(frame_id)
        height, width, channels = frame.shape
    
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
        net.setInput(blob)
        outs = net.forward(output_layers)
    
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        x_y_center = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.01:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    x_y_center.append([center_x,center_y])
                    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
        current_frame=[[],[]]
        #print(previous_frame)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                print(y)
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
                [x_cor,y_cor] = x_y_center[i]
                #new_object = False  
                if y < 100:
                    if (len(previous_frame[class_ids[i]]) == 0):
                        #new_object = True    
                        numbers[class_ids[i]]+=1
                    elif (min(previous_frame[class_ids[i]])>y_cor):
                        numbers[class_ids[i]]+=1
                #for j in range(len(previous_frame)):
                    #if j[0] == class_ids[i] and j[2]
                current_frame[class_ids[i]].append(y_cor)
        #print('\n')
        
        current_time = time.time()
        elapsed_time = current_time-previous_time
        fps = 1 / elapsed_time
        
        '''
        # Define the codec and create VideoWriter object
        ratio = .5  # resize ratio
        image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
        width2, height2, channels = image.shape
        video = cv2.VideoWriter('plastic_counter.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)
        '''
        
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
        cv2.putText(frame, "bottle="+str(numbers[0]), (10, 75), font, 2, colors[0], 3)
        cv2.putText(frame, "cans="+str(numbers[1]), (10, 100), font, 2, colors[1], 3)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        previous_time = current_time
        previous_frame = current_frame
        if key == 27:   #Esc-knop
            break
cap.release()
cv2.destroyAllWindows()

#%%
