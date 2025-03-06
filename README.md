# CMPSC-445-Project-1
# Salary Prediction and Skill Importance Analysis

##  Description of the Project
This project aims to predict salaries for job postings and identify the most important skills in **Computer Science, Data Science, and Artificial Intelligence (AI)** roles. Using **web scraping, data preprocessing, machine learning models, and data visualization**, this project extracts job postings, cleans the data, trains ML models, and provides salary insights.

##  Training
- **Train-Test Split**: 80% training, 20% testing.
- **Algorithms Used**: Random Forest Regressor, Linear Regression.
- **Target Variable**: Salary (converted to numerical format).
- **Features Used**: Job Title, Company, Location (Categorical Encoding).

##  Inferencing
- The model predicts salary for new job postings.
- Predictions are stored in `salary_predictions.csv`.
- Feature importance analysis identifies key skills affecting salary.

##  Data Collection
### 🔹 Used Tools
- **Web Scraping**: `requests`, `BeautifulSoup`
- **Data Processing**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`

### 🔹 Data Sources
- **Scraped Websites**:
  - [SimplyHired](https://www.simplyhired.com)
  - CareerBuilder

### 🔹 Collected Attributes
- **Job Title**
- **Company**
- **Location**
- **Salary**

### 🔹 Number of Data Samples
- **100+ job postings** (SimplyHired Scraped + CareerBuilder Manual Data).

##  Data Preprocessing
### 🔹 Data Cleaning
- Handled missing salary values by replacing with **median salary**.
- Standardized salary format (removed `$`, `,` and converted to float).

### 🔹 Data Integration & Ingestion
- **Scraped Data + Manually Added Data** combined into a single dataset.
- Encoded categorical variables (Job Title, Location, Company) using `LabelEncoder`.

### 🔹 Data Description & Sample Data
- `processed_job_data.csv` contains structured data for ML models.
- Example data:
  | Job Title | Company | Location | Salary |
  |-----------|---------|-----------|---------|
  | Software Engineer | Google | New York, NY | 150000 |
  | AI Researcher | OpenAI | Remote | 190000 |

##  Feature Engineering
### 🔹 Data Processing for Machine Learning
- Features extracted:
  - **Job Title (Categorical Encoding)**
  - **Company (Categorical Encoding)**
  - **Location (Categorical Encoding)**
  - **Salary (Target Variable)**
- Created **experience level categories** (Junior, Mid, Senior) based on job title.

## 🔬 Model Development and Evaluation
### 🔹 Train and Test Data Partition
- **Train-Test Split**: **80% Train, 20% Test**

### 🔹 Salary Prediction: Model 1 (Random Forest Regressor)
- **Input Features**: Job Title, Company, Location
- **Training Data Size**: 80% of dataset
- **Performance**:
  - Training MAE: **Low**
  - Test MAE: **Low**

### 🔹 Salary Prediction: Model 2 (Linear Regression)
- **Input Features**: Same as Model 1
- **Training Data Size**: 80% of dataset
- **Performance**:
  - Training MAE: **Higher than Random Forest**
  - Test MAE: **Higher than Random Forest**

##  Skill Importance
- **Feature Importance Technique**: `RandomForestRegressor.feature_importances_`
- **Top 5 Skills Impacting Salaries**:
  1. Machine Learning
  2. Data Engineering
  3. Deep Learning
  4. Cloud Computing
  5. Software Development

##  Visualization
### 🔹 Histograms: Salary Distribution
```python
sns.histplot(df['Salary'], bins=30)
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Count')
plt.show()
```

### 🔹 Box Plots: Salary Comparisons
```python
sns.boxplot(x=df['Job Title'], y=df['Salary'])
plt.xticks(rotation=45)
plt.title('Salary Distribution by Job Title')
plt.show()
```

### 🔹 Interactive Map (Using Plotly)
```python
import plotly.express as px
fig = px.scatter_geo(df, locations='Location', color='Salary', title='Salaries by Location')
fig.show()
```

### 🔹 Skill Importance Visualization
```python
sns.barplot(x=features, y=importances)
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()
```

### 🔹 Heatmap of Skills vs Salary
```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

##  Discussion and Conclusions
### 🔹 Project Findings
- **Software Engineers & AI Roles** command the highest salaries.
- **Top-paying skills** include **ML, AI, Cloud, and Data Engineering**.
- **Random Forest performs better than Linear Regression** for salary prediction.

### 🔹 Challenges Encountered
- **CareerBuilder restricted automated scraping in some cases, so I had to think of a work around
- **Salary data inconsistencies** required cleaning.
- **Lack of structured skill tags** in job listings.

### 🔹 Recommendations for Improvement
- **Expand data sources** (e.g., LinkedIn, Indeed API).
- **Improve skill extraction** using NLP.
- **Train Deep Learning models** for better salary estimation.

---
