from djitellopy import Tello
from detector import ArduinoCarDetector
import cv2

detector = ArduinoCarDetector("models/arduino_car.pt", conf_threshold=0.5)

tello = Tello()
tello.connect()
tello.streamon()

while True:
    frame = tello.get_frame_read().frame   # 720p BGR frame
    frame = cv2.resize(frame, (960, 720))  # normalize size

    detections = detector.detect(frame)
    frame = detector.draw(frame, detections)

    # Print bboxes for downstream pipeline
    for d in detections:
        print(f"bbox={d['bbox']}  conf={d['confidence']:.3f}")

    cv2.imshow("Tello Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()
print ('end -- quit demo.\r\n')


#recvThread create
recvThread = threading.Thread(target=recv)
recvThread.start()

while True: 

    try:
        msg = input("");

        if not msg:
            break  

        if 'end' in msg:
            print ('...')
            sock.close()  
            break

        # Send data
        msg = msg.encode(encoding="utf-8") 
        sent = sock.sendto(msg, tello_address)
    except KeyboardInterrupt:
        print ('\n . . .\n')
        sock.close()  
        break




