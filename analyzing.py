import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# loading all the recognitions and detection models and weights
detect_cfg = 'static/models/yolov3_testing.cfg'
flower_detect_weights = 'static/models/yolov3_training_final.weights'
leaf_detect_weights = 'static/models/yolov3_leaves_training_final.weights'
flower_model = load_model('static/models/7-flowers-classifier.h5')
leaf_model = load_model('static/models/4-leaves-classifier_new.h5')


def detecting(img_path):
    img = cv2.imread(img_path)
    copy_img = cv2.imread(img_path)
    flower_objects, flowers_detect_img = detectObject(img, flower_detect_weights, class_name='flower',
                                                      img_path=img_path)
    leaf_objects, leaves_detect_img = detectObject(copy_img, leaf_detect_weights, class_name='leaf', img_path=img_path)
    img = cv2.addWeighted(flowers_detect_img, 0.5, leaves_detect_img, 0.5, 0)
    t = time.localtime()
    timestamp = time.strftime("%m-%d-%Y_%H%M%S", t)
    path = ('static/uploads/after_detection-' + timestamp + '.jpg')
    cv2.imwrite(path, img)
    return img, ('/' + path), flower_objects, leaf_objects


def analyzing(flower_obj, leaf_obj):
    flower_strs = flower_analyze(flower_obj)
    leaf_strs = leaf_analyze(leaf_obj)
    return flower_strs, leaf_strs


def flower_analyze(flower_objs):
    if len(flower_objs) > 0:
        pred_strs = []
        for flower_img in flower_objs:
            t = time.localtime()
            timestamp = time.strftime("%m-%d-%Y_%H%M%S", t)
            path = 'static/uploads/detect_img' + timestamp + '.jpg'
            cv2.imwrite(path, flower_img)
            fg_img = backgroundDetect(flower_img)
            cv2.imwrite('static/uploads/crop_img.jpg', fg_img)
            img_to_predict = image.load_img('static/uploads/crop_img.jpg', target_size=(132, 132))
            img_to_predict = image.img_to_array(img_to_predict)
            img_to_predict = np.expand_dims(img_to_predict, axis=0)
            prob = flower_model.predict(img_to_predict)
            prediction_class = np.argmax(prob)
            pred_str = (pred_flower(prediction_class, prob))
            pred_strs.append([pred_str, ('/'+path)])
        return pred_strs


def leaf_analyze(leaf_objs):
    if len(leaf_objs) > 0:
        pred_strs = []
        for leaf_img in leaf_objs:
            t = time.localtime()
            timestamp = time.strftime("%m-%d-%Y_%H%M%S", t)
            path = 'static/uploads/detect1_img' + timestamp + '.jpg'
            cv2.imwrite(path, leaf_img)
            fg_img = backgroundDetect(leaf_img)
            cv2.imwrite('static/uploads/crop_img.jpg', fg_img)
            image_to_pred = image.load_img('static/uploads/crop_img.jpg', target_size=(132, 132))
            image_to_pred = image.img_to_array(image_to_pred)
            image_to_pred = np.expand_dims(image_to_pred, axis=0)
            p = leaf_model.predict(image_to_pred)
            prediction_class = np.argmax(p)
            pred_str = pred_leaf(prediction_class, p)
            pred_strs.append([pred_str, ('/' + path)])
        return pred_strs


def detectObject(img, weights, class_name, img_path):
    if class_name == 'flower':
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)

    height, width, channels = img.shape
    copy_img = cv2.imread(img_path)
    small_imgs = []

    # Load Yolo
    net = cv2.dnn.readNet(weights, detect_cfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # creating rectangles around the detect objects, using coordinates
    confidences = []
    rectangles = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                rectangles.append([x, y, w, h])
                confidences.append(float(confidence))

        indexes = cv2.dnn.NMSBoxes(rectangles, confidences, 0.5, 0.4)
    for i in range(len(rectangles)):
        if i in indexes:
            x, y, w, h = rectangles[i]
            x = abs(x)
            y = abs(y)

            small_imgs.append(img[y:y + h, x:x + w, 0:3])
            cv2.rectangle(copy_img, (x, y), (x + w, y + h), color, 2)

    return small_imgs, copy_img


def backgroundDetect(img):
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    w, h, ch = img.shape
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (1, 1, w, h)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
    fg_img = img * mask2[:, :, np.newaxis]
    return fg_img


def pred_flower(pred_cls, p):
    pred_str = ""
    if p[0, pred_cls] < 0.5:
        pred_str = "the system couldn't recognize the flower"
    else:
        if pred_cls == 0:
            pred_str = "the flower is daisy"
        elif pred_cls == 1:
            pred_str = "the flower is dandelion"
        elif pred_cls == 2:
            pred_str = "the flower is iris"
        elif pred_cls == 3:
            pred_str = "the flower is a rose"
        elif pred_cls == 4:
            pred_str = "the flower is sunflower"
        elif pred_cls == 5:
            pred_str = "the flower is tulip"
        elif pred_cls == 6:
            pred_str = "the flower is water lily"

    return pred_str


def pred_leaf(pred_cls, p):
    pred_str = ''
    if p[0, pred_cls] < 0.5:
        pred_str = "the system couldn't recognize the leaf's health condition"
    else:
        if pred_cls == 0:
            pred_str = "the leaf suffers from bugs"
        elif pred_cls == 1:
            pred_str = "the leaf is dehydrate"
        elif pred_cls == 2:
            pred_str = "the leaf is healthy"
        elif pred_cls == 3:
            pred_str = "the leaf suffers from lack of magnesium"
    return pred_str
