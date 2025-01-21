# Shop&Go reviews topic identification Repository

This repository contains the initial analysis and topic identification done for shop&go reviews.

## Repository Contents

### Notebooks

1. **`analysis.ipynb`**  
   - Description: Analyzes the reviews dataset by applying filters:
     - `answer_translated` is not null.
     - `rating` is not equal to 0.
   - Use case: Filtered analysis for the reviews considered in the notebooks.

2. **`analysis_complete.ipynb`**  
   - Description: Provides a comprehensive analysis of the entire reviews dataset without any filtering.
   - Use case: Complete dataset analysis.

3. **`BERTopicHierarchical.ipynb`**  
   - Description: Contains the main code to extract topics using BERTopic with the hierarchical method.  
   - Features:  
     - Select which query to use to get the reviews.
     - Extracts topics with hierarchical clustering.

4. **`BERTopicTop8.ipynb`**  
   - Description: Similar to `BERTopicHierarchical.ipynb`, but utilizes an alternative method for topic extraction, focusing on the top 8 topics.  
   - Features:  
     - Select which query to use to get the reviews.
     - Obtain topics using the alternative top 8 method.

### Scripts

1. **`app.py`**  
   - Description: A Streamlit app that provides an interactive demo for visualizing and exploring the topic filtering an AI summaries.
   - Features:  
     - Run automatically at the end of the notebooks with the command `!streamlit run app.py`.
     - Summaries and insights are displayed in an intuitive interface.

2. **`summaries.py`**  
   - Description: Code for generating AI-based summaries of the reviews.
   - Features:  
     - Run automatically when the buttons 'Short summary' or 'Detailed summary' are clicked in the demo.
     - Generates concise or detailed summaries of the data.

---

## Usage

### 0. Installing the requirements
- Installing Python 3.9.6
- pip install -r requirements.txt

### 1. Running the Notebooks
- Use Jupyter Notebook or a compatible platform to open and execute the `.ipynb` files.
- Follow the instructions within each notebook to customize queries and getting the topics.
