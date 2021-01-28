import cv2

cap = cv2.VideoCapture(0)
address = "http://192.168.1.3:8080/video"
cap.open(address)
with open("Resources/coco.names", "rt") as f:

    class_names = f.read().split("\n")

weight_file = "Resources/frozen_inference_graph.pb"
config_file = "Resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

net = cv2.dnn_DetectionModel(weight_file, config_file)
net.setInputSize(225, 225)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    class_ids, conf, bbox = net.detect(img, confThreshold=0.5)

    if len(class_ids) != 0:

        for cid, confid, box in zip(class_ids.flatten(), conf.flatten(), bbox):
            print('------------------------------------------------')
            print(class_ids, conf, bbox)
            print('Index:', cid-1)
            print("Class  Name:", class_names[cid - 1])
            print("Bounding Box:", box[0] + 10, box[1] + 20)

            cv2.rectangle(img, box, (255, 0, 0), 2)
            cv2.putText(img, class_names[cid - 1], (box[0] + 10, box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 0, 0), 1)
            cv2.putText(img, str(round(confid * 100, 2)) + "%", (box[0] + 10, box[1] + 40), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 0, 255), 1)
    cv2.imshow("Output", img)
    cv2.waitKey(1)
