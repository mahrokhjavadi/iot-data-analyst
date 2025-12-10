# Smart Energy IoT – GitHub README (Professional Version)

## 🔋 Project Overview
This repository contains a fully automated **Smart Energy IoT Data Analytics Pipeline**, designed to process, clean, analyze, cluster, and visualize large-scale smart grid data.

The pipeline follows these stages:

1. **Import**
2. **Cleaning**
3. **EDA & Correlation**
4. **Clustering**
5. **Interactive Dashboard**

All modules are modular, reproducible, and optimized for data-driven energy analysis.

---

## 📂 Project Folder Structure

project_root/
│
├── data/
│   └── raw/
│       └── smart_grid_dataset_city_modified.csv   ← User input file
│
├── outputs/
│   ├── 1_result_import/
│   ├── 2_result_cleaning/
│   ├── 3_result_correlation/
│   ├── 4_result_clustering/
│   └── dashboard/
│
├── notebooks/
│   ├── import.ipynb
│   ├── clean.ipynb
│   ├── correlation.ipynb
│   ├── clustering.ipynb
│   └── Dashboard.ipynb
│
└── README.md

---

## ✅ User Input Requirement
Place the following file in the correct folder:

```
data/raw/smart_grid_dataset_city_modified.csv
```

Once placed correctly, the entire pipeline runs without modification.

---

## ⚙️ Pipeline Steps

### STEP 1 — Import  
Loads dataset, checks structure, creates initial summaries.  
**Outputs saved to:**  
`outputs/1_result_import/`

---

### STEP 2 — Cleaning  
Handles missing values, timestamps, outliers, and standardizes variables.  
**Outputs saved to:**  
`outputs/2_result_cleaning/`

---

### STEP 3 — Correlation Analysis  
Generates heatmaps, statistical summaries, feature relationships.  
**Outputs saved to:**  
`outputs/3_result_correlation/`

---

### STEP 4 — Clustering  
Creates consumption behavior groups using K-Means or Hierarchical clustering.  
**Outputs saved to:**  
`outputs/4_result_clustering/`  
Dashboard input file:  
`clustering_results.csv`

---

### STEP 5 — Dashboard (Interactive)  
Visual exploration of:
- Consumption patterns  
- Hourly/seasonal trends  
- Cluster behaviors  

Dashboard reads:  
`outputs/4_result_clustering/clustering_results.csv`

Run via notebook:  
`Dashboard.ipynb`

---

## 📦 Installation & Dependencies

Install all required libraries:

```
pip install pandas numpy dash dash-bootstrap-components plotly scikit-learn seaborn
```

---

## ▶️ How to Run the Project

1. Place the input file into:
```
data/raw/smart_grid_dataset_city_modified.csv
```

2. Run notebooks in order:

```
1. import.ipynb
2. clean.ipynb
3. correlation.ipynb
4. clustering.ipynb
5. Dashboard.ipynb
```

---

## 📤 Outputs Summary

| Module       | Output Folder               | Description                               |
|--------------|-----------------------------|-------------------------------------------|
| Import       | 1_result_import             | Import stats & validation                 |
| Cleaning     | 2_result_cleaning           | Cleaned dataset + logs                    |
| Correlation  | 3_result_correlation        | Heatmaps & statistical analysis           |
| Clustering   | 4_result_clustering         | Cluster labels + final dataset            |
| Dashboard    | dashboard/                  | Interactive visualization                 |

---

## 👩‍💻 Author
**Dr. Mahrokh Javadi**  
**Shima Maheronnaghsh**
**Mahshid Pournajar** 
