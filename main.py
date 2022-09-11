#importing the opencv module
import cv2


# defining a function to create a blob  (blob is the binary long object classification )
def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
     #(dnn is the deep neural network it will only work in the latest version of opencv if you are using a version below 3.2 then you won't able to access the dnn in opencv library )
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False) #inside this dnn present two functionality first one is a blobformimage and the blobformimages
    #setting the import in our blob
    faceNet.setInput(blob)
    #putting the blob into forward method
    detection=faceNet.forward()
    bboxs=[]
    #adding the bounding box
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            #creating the rectangle that will show around our face
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 2)
    return frame, bboxs


#added the facemodel and faceproto here with the path of the file
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

#added the ageProto and ageModel here with the path of the file
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

#added the genderProto and genderModel here with the path of the file
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

#you can find the models in the source file that you can download at the end of the article




#we created the variable for the detecting our face
faceNet=cv2.dnn.readNet(faceModel, faceProto)
#we created the variable for the detecting our age
ageNet=cv2.dnn.readNet(ageModel,ageProto)
#we created the variable for the detecting our gender
genderNet=cv2.dnn.readNet(genderModel,genderProto)

#adding the mean value
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
#adding the age list range you can collected this age list from google image
ageList = [ '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

#Here we created a video capture object and inside the parameter, it takes an argument so here we used 0 because we are using a laptop camera.
video=cv2.VideoCapture(0)

padding=20

#we created a final window look
while True:
    ret,frame=video.read()
    frame,bboxs=faceBox(faceNet,frame)
    #creating bounding boxs and getting results
    #and loop through all the values from here
    for bbox in bboxs:
        #extracting face from here
        #bbox 1-3 will give us the width and bbox 0-2 will give us height and finally give us the face
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        #now we need to pop the phase from our blob
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]
        #argmax is for the maximum value

        label="{},{}".format(gender,age)
        cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1)
        #adding the text for showing age and gender
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1,cv2.LINE_AA)

    #add window name here
    cv2.imshow("age and gender detection",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()