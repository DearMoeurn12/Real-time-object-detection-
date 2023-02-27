# Real-time-object-detection

Real-time object detection using OpenCV and TensorFlow can be achieved by using pre-trained models from the TensorFlow Object Detection API Zoo.

The TensorFlow Object Detection API provides a collection of detection models pre-trained on the COCO dataset, which includes a large variety of common objects such as people, animals, and vehicles. These pre-trained models can be fine-tuned on custom datasets as well.

Here are the steps to perform real-time object detection using OpenCV and TensorFlow Zoo:

1. Install OpenCV and TensorFlow on your machine.
2. Download a pre-trained object detection model from the TensorFlow Zoo. The model can be downloaded as a tarball or a frozen graph.
3. Load the model in your Python script using TensorFlow's Object Detection API. The API provides functions to load the model and perform inference on input images.
4. Capture a video stream or individual images using OpenCV's VideoCapture class.
5. For each frame, preprocess the image by resizing it to the input size of the model and normalizing pixel values.
6. Pass the preprocessed image to the TensorFlow Object Detection API to obtain detection results, which include bounding box coordinates and class labels.
7. Draw the detected objects on the original image using OpenCV's drawing functions.
8. Display the resulting image or video stream using OpenCV's imshow function.
#### Here's an example code snippet that demonstrates real-time object detection using OpenCV and TensorFlow Zoo:
```
import cv2
import tensorflow as tf

# Load the pre-trained model from TensorFlow Zoo
model = tf.saved_model.load('path/to/saved_model')

# Initialize the OpenCV video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Preprocess the frame by resizing and normalizing pixel values
    frame_resized = cv2.resize(frame, (320, 320))
    frame_normalized = cv2.normalize(frame_resized, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # Perform object detection on the preprocessed frame using TensorFlow Object Detection API
    inputs = tf.constant(frame_normalized.numpy()[np.newaxis, ...], dtype=tf.float32)
    detections = model(inputs)

    # Extract the bounding box coordinates and class labels from the detection results
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)

    # Draw the detected objects on the original frame
    for i in range(len(boxes)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, str(classes[i]), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
```
