from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
from channels.generic.websocket import AsyncJsonWebsocketConsumer
import json
import numpy as np
import cv2
import mediapipe as mp
import pickle
from PIL import Image
import io

model_dict = pickle.load(open('C:/Users/DELL/Documents/sih/learn_Django/core/home/model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

# def blob_to_mat(blob, width, height):
#     # Convert the BLOB (byte array) to a numpy array
#     nparr = np.frombuffer(blob, np.uint8)
    
#     # Reshape the numpy array to the desired dimensions
#     mat = nparr.reshape((height, width, 3))  # Assuming a 3-channel image (BGR)
    
#     return mat


def blob_to_image(blob):
    # Convert the BLOB (byte array) to a BytesIO object
    image_stream = io.BytesIO(blob)
    
    # Use PIL to open the image
    image = Image.open(image_stream)
    
    # Convert the PIL image to a numpy array (OpenCV format)
    image_cv = np.array(image)
    
    # Convert RGB to BGR (OpenCV uses BGR format)
    if image_cv.ndim == 3:  # Check if the image has color channels
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    return image_cv

class TestConsumer(WebsocketConsumer):

    def connect(self):
        self.room_name = "test_consumer"
        self.room_group_name = "test_consumer_group"
        # async_to_sync(self.channel_layer.group_add)(
        #     self.room_name, self.room_group_name
        # )
        async_to_sync(self.channel_layer.group_add)(
            self.room_group_name,  # Group name first
            self.channel_name      # Channel name second
        )
        self.accept()
        self.send(text_data=json.dumps({'status' : 'connected from django channels'}))

    def receive(self,text_data):
        print(text_data)
        self.send(text_data=json.dumps({'status' : 'we got you'}))

    def disconnect(self,close_code):
        print(f'disconnected with close code {close_code}')

    def send_notification(self,event):
        print('send notification')
        print(event)
        #self.send(text_data=json.dumps({'payload' : event.get('value')}))
        data = json.loads(event.get('value'))
        self.send(text_data=json.dumps({'payload' : data}))
        print('send notification')

class NewConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.room_name = "new_consumer"
        self.room_group_name = "new_consumer_group"

        await(self.channel_layer.group_add)(
            self.room_group_name,  # Group name first
            self.channel_name      # Channel name second
        )
        await self.accept()
        await self.send(text_data=json.dumps({'status' : 'connected from new async json consumer'}))

    async def receive(self,text_data):
        print(text_data)
        await self.send(text_data=json.dumps({'status' : 'we got you'}))

    async def disconnect(self,close_code):
         print(f'disconnected with close code {close_code}')

    async def send_notification(self,event):
      
        data = json.loads(event.get('value'))
        await self.send(text_data=json.dumps({'payload' : data}))
        
class VideoConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.room_name = "new_consumer"
        self.room_group_name = "new_consumer_group"

        await self.channel_layer.group_add(
            self.room_group_name,  # Group name first
            self.channel_name      # Channel name second
        )
        await self.accept()
        await self.send(text_data=json.dumps({'status': 'connected from new async json consumer'}))

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # Process the binary data (video frame)
            #print(f"Received video frame of size: {len(bytes_data)} bytes")
            # Here you can save the frame, process it, or send it to another group, etc.
            # For example, you could save it to a file or process it with OpenCV.
            #frame = blob_to_image(bytes_data)

            image_stream = io.BytesIO(bytes_data)
            
            # Use PIL to open the image
            image = Image.open(image_stream)
            
            # Convert the PIL image to a numpy array (OpenCV format)
            frame = np.array(image)
            
            # Convert RGB to BGR (OpenCV uses BGR format)
            #if frame.ndim == 3:  # Check if the image has color channels 
                #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) Need not to convert, it is causing problem as
                #  in the original code it is RGB not BGR

            data_aux = []
            x_ = []
            y_ = []

            H, W, _ = frame.shape

            #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame)
            #cv2.imshow('frame',frame)
            #cv2.waitKey(1)
            #print ("before for")
            if results.multi_hand_landmarks:
                # for hand_landmarks in results.multi_hand_landmarks:
                #     mp_drawing.draw_landmarks(
                #         frame,  # image to draw
                #         hand_landmarks,  # model output
                #         mp_hands.HAND_CONNECTIONS,  # hand connections
                #         mp_drawing_styles.get_default_hand_landmarks_style(),
                #         mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    #print ("in for")
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                
                if(len(data_aux)==42):
                    prediction = model.predict([np.asarray(data_aux)])

                    predicted_character = labels_dict[int(prediction[0])]
                    #print(predicted_character)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

            
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)
            # Optionally, send a response back to the client
            await self.send(text_data=json.dumps({'status': 'video frame received'}))
        elif text_data:
            # Handle text data if needed
            print(f"Received text data: {text_data}")
            await self.send(text_data=json.dumps({'status': 'text data received'}))
        
        else:
            print("something else is received")

    async def disconnect(self, close_code):
        print(f'disconnected with close code {close_code}')

    async def send_notification(self, event):
        data = json.loads(event.get('value'))
        await self.send(text_data=json.dumps({'payload': data}))