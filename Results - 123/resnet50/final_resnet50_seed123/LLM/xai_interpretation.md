Certainly! Below is a consolidated report combining all the extracted sections from the JSON data provided. Each section is presented in a clean, readable format:

---

### **Overall Report**
- **Timestamp:** 2026-11-17T10:30:50.935753+06:00
- **Report Type:** Final Report
- **Total Characters:** 3532

---

### **Source File: results\\final_resnet50_seed123.json**
- **File Type:** .json
- **Content:**
```json
{
  "model_info": {
    "model_name": "resnet50",
    "run_name": "final_resnet50_seed123",
    "timestamp": "2026-11-16T11:08:15.222006+06:00",
    "device": "cuda:0",
    "gpu_info": {
      "vendor": "NVIDIA",
      "model": "GeForce RTX 3060 Laptop GPU",
      "compute_capability": "8.6"
    },
    "image_size": "224x224",
    "classes": 7,
    "total_params": 23522375,
    "trainable_params": 23522375,
    "frozen_params": 0,
    "storage_mb": 89.731,
    "state_dict_storage_mb": 89.934,
    "best_checkpoint_size_mb": 269.642
  },
  "training_info": {
    "total_duration": "00:43:53",
    "total_duration_seconds": 2633.106674899991
  },
  "efficiency_benchmark": {
    "batch_size": [
      {
        "batch_size": 1,
        "status": "success",
        "mean_latency_ms": 13.8286,
        "median_latency_ms": 13.6704,
        "p95_latency_ms": 16.4422,
        "per_image_latency_ms": 13.8286,
        "throughput_images_per_sec": 71.7132,
        "gpu_peak_allocated_mb": 100.6533203125,
        "gpu_peak_reserved_mb": 122.0
      },
      {
        "batch_size": 8,
        "status": "success",
        "mean_latency_ms": 15.1042,
        "median_latency_ms": 14.9847,
        "p95_latency_ms": 15.1960,
        "per_image_latency_ms": 1.8880,
        "throughput_images_per_sec": 528.7473,
        "gpu_peak_allocated_mb": 137.6865234375,
        "gpu_peak_reserved_mb": 184.0
      },
      {
        "batch_size": 16,
        "status": "success",
        "mean_latency_ms": 26.3690,
        "median_latency_ms": 26.3552,
        "p95_latency_ms": 26.3890,
        "per_image_latency_ms": 1.6481,
        "throughput_images_per_sec": 606.5038,
        "gpu_peak_allocated_mb": 185.1552734375,
        "gpu_peak_reserved_mb": 236.0
      }
    ]
  }
}
```
- **Original Characters:** 2223

---

### **Source File: evaluation\\evaluation_report.json**
- **File Type:** .json
- **Content:**
```json
{
  "evaluation": {
    "columns": [
      "image_id",
      "label",
      "predicted_class"
    ],
    "rows": [
      {
        "image_id": "583",
        "label": "Fall Armyworm (Insect) Damage",
        "predicted_class": "Fall Armyworm (Insect) Damage"
      },
      {
        "image_id": "584",
        "label": "Healthy Leaf",
        "predicted_class": "Healthy Leaf"
      },
      {
        "image_id": "585",
        "label": "Leaf Spot",
        "predicted_class": "Leaf Spot"
      },
      {
        "image_id": "586",
        "label": "Magnesium Deficiency",
        "predicted_class": "Magnesium Deficiency"
      },
      {
        "image_id": "587",
        "label": "Nitrogen Deficiency",
        "predicted_class": "Nitrogen Deficiency"
      },
      {
        "image_id": "588",
        "label": "Northern Leaf Blight",
        "predicted_class": "Northern Leaf Blight"
      },
      {
        "image_id": "589",
        "label": "Phosphorus Deficiency",
        "predicted_class": "Phosphorus Deficiency"
      }
    ],
    "rows_included": 7
  }
}
```
- **Original Characters:** 277

---

### **Source File: evaluation\\evaluation_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\error_analysis\\confusion_matrix.json**
- **File Type:** .json
- **Content:**
```json
{
  "columns": [
    "actual_class",
    "Fall Armyworm (Insect) Damage",
    "Healthy Leaf",
    "Leaf Spot",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Northern Leaf Blight",
    "Phosphorus Deficiency"
  ],
  "total_rows": 7,
  "sampled_rows": [
    {
      "actual_class": "Fall Armyworm (Insect) Damage",
      "Fall Armyworm (Insect) Damage": "113",
      "Healthy Leaf": "6",
      "Leaf Spot": "3",
      "Magnesium Deficiency": "0",
      "Nitrogen Deficiency": "0",
      "Northern Leaf Blight": "4",
      "Phosphorus Deficiency": "0"
    },
    {
      "actual_class": "Healthy Leaf",
      "Fall Armyworm (Insect) Damage": "0",
      "Healthy Leaf": "74",
      "Leaf Spot": "1",
      "Magnesium Deficiency": "0",
      "Nitrogen Deficiency": "0",
      "Northern Leaf Blight": "0",
      "Phosphorus Deficiency": "0"
    },
    {
      "actual_class": "Leaf Spot",
      "Fall Armyworm (Insect) Damage": "0",
      "Healthy Leaf": "1",
      "Leaf Spot": "59",
      "Magnesium Deficiency": "0",
      "Nitrogen Deficiency": "1",
      "Northern Leaf Blight": "0",
      "Phosphorus Deficiency": "0"
    },
    {
      "actual_class": "Magnesium Deficiency",
      "Fall Armyworm (Insect) Damage": "0",
      "Healthy Leaf": "0",
      "Leaf Spot": "0",
      "Magnesium Deficiency": "83",
      "Nitrogen Deficiency": "0",
      "Northern Leaf Blight": "0",
      "Phosphorus Deficiency": "0"
    },
    {
      "actual_class": "Nitrogen Deficiency",
      "Fall Armyworm (Insect) Damage": "1",
      "Healthy Leaf": "0",
      "Leaf Spot": "1",
      "Magnesium Deficiency": "0",
      "Nitrogen Deficiency": "62",
      "Northern Leaf Blight": "0",
      "Phosphorus Deficiency": "0"
    },
    {
      "actual_class": "Northern Leaf Blight",
      "Fall Armyworm (Insect) Damage": "1",
      "Healthy Leaf": "0",
      "Leaf Spot": "8",
      "Magnesium Deficiency": "0",
      "Nitrogen Deficiency": "1",
      "Northern Leaf Blight": "70",
      "Phosphorus Deficiency": "0"
    },
    {
      "actual_class": "Phosphorus Deficiency",
      "Fall Armyworm (Insect) Damage": "0",
      "Healthy Leaf": "0",
      "Leaf Spot": "2",
      "Magnesium Deficiency": "0",
      "Nitrogen Deficiency": "0",
      "Northern Leaf Blight": "0",
      "Phosphorus Deficiency": "0"
    }
  ],
  "rows_included": 7
}
```
- **Original Characters:** 345

---

### **Source File: evaluation\\validation\\error_analysis\\confusion_matrix.csv**
- **File Type:** .csv
- **Content:**
```csv
actual_class,Fall Armyworm (Insect) Damage,Healthy Leaf,Leaf Spot,Magnesium Deficiency,Nitrogen Deficiency,Northern Leaf Blight,Phosphorus Deficiency
Fall Armyworm (Insect) Damage,113,6,3,0,0,4,0
Healthy Leaf,0,74,1,0,0,0,0
Leaf Spot,0,0,59,0,1,0,0
Magnesium Deficiency,0,0,0,83,0,0,0
Nitrogen Deficiency,1,0,1,0,62,0,0
Northern Leaf Blight,1,0,8,0,0,70,0
Phosphorus Deficiency,0,0,2,0,0,0,0
```
- **Original Characters:** 223

---

### **Source File: evaluation\\validation\\performance_report.json**
- **File Type:** .json
- **Content:**
```json
{
  "evaluation": {
    "columns": [
      "image_id",
      "label",
      "predicted_class"
    ],
    "rows": [
      {
        "image_id": "583",
        "label": "Fall Armyworm (Insect) Damage",
        "predicted_class": "Fall Armyworm (Insect) Damage"
      },
      {
        "image_id": "584",
        "label": "Healthy Leaf",
        "predicted_class": "Healthy Leaf"
      },
      {
        "image_id": "585",
        "label": "Leaf Spot",
        "predicted_class": "Leaf Spot"
      },
      {
        "image_id": "586",
        "label": "Magnesium Deficiency",
        "predicted_class": "Magnesium Deficiency"
      },
      {
        "image_id": "587",
        "label": "Nitrogen Deficiency",
        "predicted_class": "Nitrogen Deficiency"
      },
      {
        "image_id": "588",
        "label": "Northern Leaf Blight",
        "predicted_class": "Northern Leaf Blight"
      },
      {
        "image_id": "589",
        "label": "Phosphorus Deficiency",
        "predicted_class": "Phosphorus Deficiency"
      }
    ],
    "rows_included": 7
  }
}
```
- **Original Characters:** 277

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.json**
- **File Type:** .json
- **Content:**
```json
{
  "evaluation": {
    "columns": [
      "image_id",
      "label",
      "predicted_class"
    ],
    "rows": [
      {
        "image_id": "583",
        "label": "Fall Armyworm (Insect) Damage",
        "predicted_class": "Fall Armyworm (Insect) Damage"
      },
      {
        "image_id": "584",
        "label": "Healthy Leaf",
        "predicted_class": "Healthy Leaf"
      },
      {
        "image_id": "585",
        "label": "Leaf Spot",
        "predicted_class": "Leaf Spot"
      },
      {
        "image_id": "586",
        "label": "Magnesium Deficiency",
        "predicted_class": "Magnesium Deficiency"
      },
      {
        "image_id": "587",
        "label": "Nitrogen Deficiency",
        "predicted_class": "Nitrogen Deficiency"
      },
      {
        "image_id": "588",
        "label": "Northern Leaf Blight",
        "predicted_class": "Northern Leaf Blight"
      },
      {
        "image_id": "589",
        "label": "Phosphorus Deficiency",
        "predicted_class": "Phosphorus Deficiency"
      }
    ],
    "rows_included": 7
  }
}
```
- **Original Characters:** 277

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
```
- **Original Characters:** 256

---

### **Source File: evaluation\\validation\\performance_report.csv**
- **File Type:** .csv
- **Content:**
```csv
image_id,label,predicted_class
583,Fall Armyworm (Insect) Damage,Fall Armyworm (Insect) Damage
584,Healthy Leaf,Healthy Leaf
585,Leaf Spot,Leaf Spot
586,Magnesium Deficiency,Magnesium Deficiency
587,Nitrogen Deficiency,Nitrogen Deficiency
588,Northern Leaf Blight,Northern Leaf Blight
589,Phosphorus Deficiency,Phosphorus Deficiency
