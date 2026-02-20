OpenCV Face Detection — Maxed-Out Configuration (Reference)

Model Used: Caffe SSD ResNet-10 Face Detector
Full model name: res10_300x300_ssd_iter_140000_fp16.caffemodel
Framework:
    - OpenCV cv2.dnn
    - Caffe model format
    - Input resolution: 300 × 300


Purpose of This Configuration

With the given test image, the settings below represent the maximum achievable accuracy and stability of this face detection model without introducing false positives or breaking detections.
⚠️ This configuration prioritizes precision over recall (i.e., not all faces are detected, but every detected face is a real face).

Locked / Critical Settings (Do Not Change)
These values were empirically determined to be unstable or destructive if modified:
    Confidence threshold:       0.14        (Lower confidence → trees / background become faces, Higher confidence → real faces disappear)
    Minimum box size (px):      29          (Smaller box size → noise explosion, Larger box size → small faces lost)
    Max aspect ratio filter:    1.7         (Aspect ratio < 1.5 → faces lost, Aspect ratio > 1.7 → no benefit)

Final Optimal Settings (Best Case Scenario)
    Confidence threshold:       0.14
    Min box size (px):          29
    Max aspect ratio filter:    1.7
    NMS IoU threshold:          0.15
    Keep top-K after NMS:       10
    Scales:                     1.0

Disabled Features (Intentionally OFF)
These features were tested extensively and reduced accuracy for this model and image:
    MAX Recall Mode (multi-scale):      OFF
    Enable tiling:                      OFF
    Enable CLAHE:                       OFF
    Multi-scale inference:              OFF
Reasons:
    Multi-scale + tiling caused massive false positives
    CLAHE amplified background textures (trees, shadows)
    Recall-oriented modes break precision guarantees


Summary Statement
With the given picture, these settings represent the maximum practical capabilities of the Caffe SSD ResNet-10 face detection model.

This configuration achieves:
    - High confidence detections
    - Near-zero false positives
    - Stable behavior without parameter collapse
Any further improvement requires a different model, not further tuning.

=======================================================================
Notes for Self

This model is not designed for:
    - Dense crowds
    - Heavy occlusion
    - Masks + extreme angles
Precision ceiling is reached here
Further gains require:
    - RetinaFace
    - YuNet
    - MediaPipe
    - YOLO-based face detectors