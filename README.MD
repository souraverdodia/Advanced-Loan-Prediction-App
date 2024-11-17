# Loan Approval Prediction App

This is a machine learning project developed by Monali and Jayash Bhardwaj as a final group project for our internship at Hoping Minds. The project aims to predict loan approval based on various applicant features using a Random Forest Classifier.

## Features

The project offers the following features:

* **Loan Approval Prediction:** Predicts the likelihood of loan approval based on user-provided input.
* **Interactive Data Exploration:** Allows users to explore and visualize the loan application data through interactive charts and graphs.
* **User-Friendly Interface:** Provides a simple and intuitive interface built with Streamlit for easy interaction and understanding.
* **Data Preprocessing:** Includes data cleaning, handling missing values, and feature engineering for optimal model performance.
* **Model Training and Evaluation:** Utilizes a Random Forest Classifier trained on a preprocessed dataset for accurate predictions.
* **Visualization of Prediction Probability:** Displays the approval probability using an interactive gauge chart for better visualization.

## Installation

1. Clone the repository: `git clone https://github.com/jayash1973/Advanced-Loan-Predictor`
2. Install the required packages: `pip install -r requirements.txt`

## Usage

1. Run the app: `streamlit run app.py`
2. Navigate to the app in your web browser (usually `http://localhost:8501`).
3. Use the sidebar to choose between "Predict" and "Explore Data" modes.

**Predict Mode:**

* Fill in the required applicant information in the provided fields.
* Click the "Predict" button to get the loan approval prediction and probability.

**Explore Data Mode:**

* Explore various visualizations and insights from the loan application data, including:
    * Loan Status Distribution
    * Correlation Heatmap
    * Loan Amount Distribution
    * Applicant Income vs Loan Amount
    * Loan Status by Education and Credit History

    ![Explore Data 1 Ui](images/explore-data-1.jpg)

    ![Explore Data 1 Ui](images/explore-data-2.jpg)

    ![Explore Data 1 Ui](images/explore-data-3.jpg)

## Data

The project uses a loan application dataset (`train.csv`) for training and prediction. The dataset contains various features such as gender, marital status, education, income, loan amount, and credit history.

## Model

The prediction model is a Random Forest Classifier, chosen for its robustness and ability to handle complex relationships within the data.

## Live Demo

You can test this project live at: [https://huggingface.co/spaces/Johan713/Advanced-Loan-Predictor](https://huggingface.co/spaces/Johan713/Advanced-Loan-Predictor)

***screenshots***
![Streamlit Ui](images/pre-prediction-result.jpg)
![Streamlit Ui](images/prediction-results.jpg)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Authors

* **Monali**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=for-the-badge&logo=github)](https://github.com/Monali-0830)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/monali-verma-6b6b1028a/)

* **Jayash Bhardwaj**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=for-the-badge&logo=github)](https://github.com/jayash1973)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/jayash-bhardwaj)

## Acknowledgments

* Hoping Minds for providing the opportunity and mentorship for this project.

## License

This project is licensed under the MIT License.

## Contact Details

**Jayash Bhardwaj**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=for-the-badge&logo=github)](https://github.com/jayash1973)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/jayash-bhardwaj)

**Monali**
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=for-the-badge&logo=github)](https://github.com/jayash1973)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/monali-verma-6b6b1028a/)