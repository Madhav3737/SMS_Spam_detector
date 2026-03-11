📩 SMS Spam Detection Web Application

A Machine Learning + NLP based web application that detects whether an SMS message is Spam or Ham (Not Spam).
The system uses trained ML models and NLP preprocessing to classify messages and visualize prediction probabilities.

🚀 Features

  - Detects Spam / Ham SMS messages

  - Uses Natural Language Processing (NLP) for text preprocessing

  - Multiple trained models included

       - Naive Bayes

       - Random Forest

       - Support Vector Classifier

  - Displays probability visualization using Pie Charts

  - Simple Flask Web Interface

  - Bootstrap based UI

🧠 Machine Learning Models Used

The project includes the following trained models:

  - Model	File
  - Multinomial Naive Bayes	model_mnb.pkl
  - Random Forest	model_rf.pkl
  - Support Vector Classifier	model_svc.pkl

The text data is converted into numerical features using:

TF-IDF Vectorizer

File used:

vectorizer.pkl


🗂️ Project Structure
sms-spam-detection/
│
├── app.py                     # Flask application
├── preprocess_m.py            # Message preprocessing functions
├── generate_pie_chart.py      # Generates prediction probability charts
│
├── model_mnb.pkl              # Naive Bayes trained model
├── model_rf.pkl               # Random Forest trained model
├── model_svc.pkl              # Support Vector Classifier model
├── vectorizer.pkl             # TF-IDF vectorizer
│
├── spam.csv                   # SMS dataset
├── spam_ds2.csv               # Additional dataset
├── ham_call_messages.csv      # Ham message dataset
├── gathered_data.csv          # Collected message data
│
├── sms_spam.ipynb             # Jupyter Notebook for model training
│
├── nltk_data/                 # NLTK resources
├── nltk_setup.py              # Script to setup nltk resources
│
├── templates/                 # HTML templates
├── static/                    # CSS, JS, images
│
├── requirements.txt           # Required Python libraries
└── README.md                  # Project documentation

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/sms-spam-detection.git
cd sms-spam-detection

2️⃣ Install Required Libraries
pip install -r requirements.txt

3️⃣ Setup NLTK Resources
python nltk_setup.py

4️⃣ Run the Application
python app.py

5️⃣ Open in Browser
http://127.0.0.1:5000

🧾 Example

Input Message:

Congratulations! You have won a $1000 Walmart gift card.
Click the link to claim your prize.

Output:

Prediction: Spam
Probability Chart: Spam vs Ham
📊 Dataset

The model is trained using SMS datasets containing labeled messages:

Spam messages
Legitimate (Ham) messages
Datasets used:
spam.csv
spam_ds2.csv
ham_call_messages.csv

🛠 Technologies Used
Python
Flask
Scikit-learn
NLTK
Pandas
NumPy
Bootstrap
Matplotlib

📈 Workflow

1️⃣ User enters a message in the web interface
2️⃣ Message is sent to the Flask backend
3️⃣ NLP preprocessing is applied
4️⃣ Text is transformed using TF-IDF Vectorizer
5️⃣ Model predicts Spam / Ham
6️⃣ Probability pie chart is generated
7️⃣ Result is displayed to the user

🎯 Future Improvements

Add Deep Learning (LSTM / BERT) models

Deploy on Cloud (AWS / Heroku / Render)

Create REST API version

Add Real-time message filtering

👨‍💻 Author

Madhav Appana

📜 License

This project is for educational purposes.
