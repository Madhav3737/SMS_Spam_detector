from flask import Flask,render_template,request
import preprocess_m
import nltk_setup
import joblib
app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizor = joblib.load('vectorizer.pkl')

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('index.html')

@app.route('/result',methods = ['POST','GET'])
def result():
    msg = request.form['msg']
    print(msg,'\n')
    cleaned_msg = preprocess_m.preprocess_msg(msg)
    X = vectorizor.transform([cleaned_msg])
    output = model.predict(X)
    op = 'HAM' if output[0]==0 else 'SPAM' if output[0]==1 else 'ERROR Occured'
    print(op,'\n')
    return render_template('result.html',output = op)

if __name__ == '__main__':
    nltk_setup.ensure_nltk_resources()
    app.run(debug=True)