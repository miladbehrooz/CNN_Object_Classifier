import numpy as np
import cv2
import tensorflow.keras as keras
from tensorflow.keras.applications import mobilenet_v2

def predict_frame(image,model):
    
    # define the classes
    classes = ['tomato','apple','pen','highlighter','empty']
    
    # reverse color channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # image should be a numpy array here

    # reshape image to (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)
    #print(image.shape)
    
    # apply pre-processing
    image = mobilenet_v2.preprocess_input(image)
    
    # load pretrained model
    

    # make prediction 
    predictions_prob = model.predict(image)
    prob =  np.max(predictions_prob[0])
    prediction = classes[np.argmax(predictions_prob)]
    
    return prediction , prob
# load model
model = keras.models.load_model('./models/4clasess_object_recognizer_pretrained_model.h5')
# set some general parameters
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (600, 170)
# fontScale
fontScale = 1
# colors in BGR
color_red = (0, 0, 255)
color_green = (0, 255, 0)
color = color_red
# Line thickness of 2 px
thickness = 2
offset = 2
width = 224
x = 600
y = 200
count = 1
prediction = 'empty'
webcam = cv2.VideoCapture(0)
while True:
    if prediction != 'empty':
        color = color_green 
    else:
        color = color_red 
    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(img=frame, 
                    pt1=(x-offset,y-offset), 
                    pt2=(x+width+offset, y+width+offset), 
                    color=color, #bgr
                    thickness=2
    )
    if prediction != 'empty':
        cv2.putText(frame, f'{prediction}: {round(prob * 100,1) } %',org, font, 
                    fontScale, color, thickness, cv2.LINE_AA) 
    cv2.imshow('frame', frame) 
    
    if count % 10 ==0:
        image = frame[y:y+width, x:x+width, :]
        prediction, prob = predict_frame(image,model)

          
    count += 1 
     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
cv2.destroyAllWindows()
