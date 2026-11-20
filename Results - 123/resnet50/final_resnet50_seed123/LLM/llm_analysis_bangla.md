Sure, I'll generate the complete report now. Here is the comprehensive report combining all the files and their details:

---

### **1. ResNet50 Model Details**

**Model Name:** resnet50

**Run Name:** final_resnet50_seed123

**Timestamp:** 2026-11-16T11:08:15.222006+06:00

**Device:** cuda:0

**GPU:** NVIDIA GeForce RTX 3060 Laptop GPU

**Image Size:** 224x224

**Classes:** 7

**Model Parameters:**
- Total parameters: 23,522,375
- Trainable parameters: 23,522,375
- Frozen parameters: 0
- Parameter storage: 89.731 MB
- State dict storage: 89.934 MB
- Best checkpoint size: 269.642 MB

**Computational Complexity:**
- MACs/image: 4,087,150,592
- GMACs/image: 4.087151
- FLOPs/image: 8,174,301,184
- GFLOPs/image: 8.174301

**Note:** MAC/FLOP counts are estimates from supported module hooks. See JSON for excluded operations.

**Training Information:**
- Total training duration: 00:43:53
- Training duration seconds: 2633.106674899991

**Inference Performance:**
- Batch size: 1
  - Status: success
  - Mean latency: 13.8286 ms
  - Median latency: 13.6704 ms
  - P95 latency: 16.4422 ms
  - Per-image latency: 13.8286 ms
  - Throughput: 71.7132 images/sec
  - GPU peak allocated: 100.6533203125 MB
  - GPU peak reserved: 122.0 MB

- Batch size: 8
  - Status: success
  - Mean latency: 15.1042 ms
  - Median latency: 14.9847 ms
  - P95 latency: 15.1960 ms
  - Per-image latency: 1.8880 ms
  - Throughput: 528.7473 images/sec
  - GPU peak allocated: 137.6865234375 MB
  - GPU peak reserved: 184.0 MB

- Batch size: 16
  - Status: success
  - Mean latency: 26.3690 ms
  - Median latency: 26.3552 ms
  - P95 latency: 26.3890 ms
  - Per-image latency: 1.6481 ms
  - Throughput: 606.5038 images/sec
  - GPU peak allocated: 185.1552734375 MB
  - GPU peak reserved: 236.0 MB

---

### **2. ResNet50 Image Classification Results**

**Dataset:** maize_leaf_classification

**Sample Size:** 7

**Actual vs. Predicted Results:**

1. **Actual / Predicted:** Magnesium Deficiency
   - **Fall Armyworm (Insect) Damage:** 0
   - **Healthy Leaf:** 0
   - **Leaf Spot:** 0
   - **Magnesium Deficiency:** 83
   - **Nitrogen Deficiency:** 0
   - **Northern Leaf Blight:** 0
   - **Phosphorus Deficiency:** 0

2. **Actual / Predicted:** Nitrogen Deficiency
   - **Fall Armyworm (Insect) Damage:** 1
   - **Healthy Leaf:** 0
   - **Leaf Spot:** 1
   - **Magnesium Deficiency:** 0
   - **Nitrogen Deficiency:** 62
   - **Northern Leaf Blight:** 0
   - **Phosphorus Deficiency:** 0

3. **Actual / Predicted:** Northern Leaf Blight
   - **Fall Armyworm (Insect) Damage:** 1
   - **Healthy Leaf:** 0
   - **Leaf Spot:** 8
   - **Magnesium Deficiency:** 0
   - **Nitrogen Deficiency:** 1
   - **Northern Leaf Blight:** 70
   - **Phosphorus Deficiency:** 0

4. **Actual / Predicted:** Phosphorus Deficiency
   - **Fall Armyworm (Insect) Damage:** 0
   - **Healthy Leaf:** 0
   - **Leaf Spot:** 2
   - **Magnesium Deficiency:** 0
   - **Nitrogen Deficiency:** 0
   - **Northern Leaf Blight:** 0
   - **Phosphorus Deficiency:** 73

**Actual vs. Predicted Confusion Matrix:**

- **Actual / Predicted:** Magnesium Deficiency
  - **Fall Armyworm (Insect) Damage:** 113
  - **Healthy Leaf:** 6
  - **Leaf Spot:** 3
  - **Magnesium Deficiency:** 0
  - **Nitrogen Deficiency:** 0
  - **Northern Leaf Blight:** 4
  - **Phosphorus Deficiency:** 0

- **Actual / Predicted:** Healthy Leaf
  - **Fall Armyworm (Insect) Damage:** 0
  - **Healthy Leaf:** 74
  - **Leaf Spot:** 1
  - **Magnesium Deficiency:** 0
  - **Nitrogen Deficiency:** 0
  - **Northern Leaf Blight:** 0
  - **Phosphorus Deficiency:** 0

- **Actual / Predicted:** Leaf Spot
  - **Fall Armyworm (Insect) Damage:** 0
  - **Healthy Leaf:** 1
  - **Leaf Spot:** 59
  - **Magnesium Deficiency:** 0
  - **Nitrogen Deficiency:** 1
  - **Northern Leaf Blight:** 0
  - **Phosphorus Deficiency:** 0

- **Actual / Predicted:** Magnesium Deficiency
  - **Fall Armyworm (Insect) Damage:** 0
  - **Healthy Leaf:** 0
  - **Leaf Spot:** 0
  - **Magnesium Deficiency:** 83
  - **Nitrogen Deficiency:** 0
  - **Northern Leaf Blight:** 0
  - **Phosphorus Deficiency:** 0

- **Actual / Predicted:** Nitrogen Deficiency
  - **Fall Armyworm (Insect) Damage:** 1
  - **Healthy Leaf:** 0
  - **Leaf Spot:** 1
  - **Magnesium Deficiency:** 0
  - **Nitrogen Deficiency:** 62
  - **Northern Leaf Blight:** 0
  - **Phosphorus Deficiency:** 0

- **Actual / Predicted:** Northern Leaf Blight
  - **Fall Armyworm (Insect) Damage:** 1
  - **Healthy Leaf:** 0
  - **Leaf Spot:** 8
  - **Magnesium Deficiency:** 0
  - **Nitrogen Deficiency:** 1
  - **Northern Leaf Blight:** 70
  - **Phosphorus Deficiency:** 0

- **Actual / Predicted:** Phosphorus Deficiency
  - **Fall Armyworm (Insect) Damage:** 0
  - **Healthy Leaf:** 0
  - **Leaf Spot:** 2
  - **Magnesium Deficiency:** 0
  - **Nitrogen Deficiency:** 0
  - **Northern Leaf Blight:** 0
  - **Phosphorus Deficiency:** 73

**Original Characters:**
- **Model Parameters:** 23,522,375
- **Computational Complexity:** 4,087,150,592 MACs/image, 4.087151 GMACs/image, 8,174,301,184 FLOPs/image, 8.174301 GFLOPs/image
- **Training Information:** Total training duration: 00:43:53, Training duration seconds: 2633.106674899991
- **Inference Performance:** 
  - Batch size: 1, Status: success, Mean latency: 13.8286 ms, Median latency: 13.6704 ms, P95 latency: 16.4422 ms, Per-image latency: 13.8286 ms, Throughput: 71.7132 images/sec, GPU peak allocated: 100.6533203125 MB, GPU peak reserved: 122.0 MB
  - Batch size: 8, Status: success, Mean latency: 15.1042 ms, Median latency: 14.9847 ms, P95 latency: 15.1960 ms, Per-image latency: 1.8880 ms, Throughput: 528.7473 images/sec, GPU peak allocated: 137.6865234375 MB, GPU peak reserved: 184.0 MB
  - Batch size: 16, Status: success, Mean latency: 26.3690 ms, Median latency: 26.3552 ms, P95 latency: 26.3890 ms, Per-image latency: 1.6481 ms, Throughput: 606.5038 images/sec, GPU peak allocated: 185.1552734375 MB, GPU peak reserved: 236.0 MB

---

### **3. ResNet50 Image Classification Results (Detailed Report)**

The detailed results of the model on the maize leaf classification dataset are provided. The dataset contains 7 samples, and the model's predictions are compared against the actual labels to determine the accuracy and precision of the model.

#### **Detailed Predictions and Results:**
- **Magnesium Deficiency:** The model predicts this condition for 113 samples but is incorrect for 4 samples.
- **Healthy Leaf:** The model correctly identifies this condition for 74 samples.
- **Leaf Spot:** The model correctly identifies this condition for 59 samples.
- **Nitrogen Deficiency:** The model correctly identifies this condition for 62 samples.
- **Northern Leaf Blight:** The model correctly identifies this condition for 70 samples.
- **Phosphorus Deficiency:** The model correctly identifies this condition for 73 samples.

#### **Confusion Matrix:**
The confusion matrix provides a clear overview of how the model performed on each condition. The confusion matrix indicates that the model correctly identified the condition in 60 out of the 7 samples, but there are discrepancies in the predictions for the remaining samples.

#### **Original Characters:**
The original characters represent the total number of parameters, computational complexity, and training information, which are essential for understanding the model's structure and training process.

---

### **4. ResNet50 Image Classification Results (Additional Information)**

- **Dataset:** maize_leaf_classification
- **Sample Size:** 7
- **Model Name:** resnet50
- **Run Name:** final_resnet50_seed123
- **Timestamp:** 2026-11-16T11:08:15.222006+06:00
- **Device:** cuda:0
- **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU
- **Image Size:** 224x224
- **Classes:** 7

---

This comprehensive report covers all the details of the ResNet50 model, including its parameters, computational complexity, training information, and detailed results on the maize leaf classification dataset. The confusion matrix and original characters provide a clear picture of the model's performance.

---

If you need any further assistance or have additional questions, feel free to ask!
