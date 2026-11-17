import cv2
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

# เอา URL ของกล้องมาใส่
url = 'http://192.168.1.187/cam-hi.jpg'
cv2. namedWindow ("live Cam Testing", cv2. WINDOW_AUTOSIZE)

#  สร้างออบเจ็กต์สำหรับเชื่อมต่อกับกล้อง
cap = cv2. VideoCapture (url)

# ตรวจสอบว่าสามารถเชื่อมต่อกับกล้องไหม
if not cap.isOpened () :
    print ("Failed to open the IP camera stream")
    exit ()
    
# รับภาพและแสดงผลจากกล้องอย่างต่อเนื่อง
while True:
    # ดึงภาพจากกล้อง
    img_resp=urllib.request.urlopen (url)
    imgnp=np.array (bytearray (img_resp.read()) ,dtype=np.uint8)

    # แปลงข้อมูลภาพให้อยู่ในรูปแบบที่ OpenCV ใช้งานได้
    im = cv2. imdecode (imgnp, -1)

    # แสดงภาพจากกล้อง
    cv2. imshow ('live Cam Testing', im)
    key=cv2. waitKey (5)
    if key==ord('q'):
        break


cap. release ()
cv2.destroyAllWindows ()
