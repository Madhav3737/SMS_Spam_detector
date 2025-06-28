from flask import Flask,render_template,request
import preprocess_m
import nltk_setup
import joblib
import numpy as np 
import generate_pie_chart

app = Flask(__name__)

mnb = joblib.load('model_mnb.pkl')
rf = joblib.load('model_rf.pkl')
svc = joblib.load('model_svc.pkl')
vectorizor = joblib.load('vectorizer.pkl')

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('index.html')

@app.route('/result',methods = ['POST','GET'])
def result():
    msg = request.form['msg']
    # print(msg,'\n')
    cleaned_msg = preprocess_m.preprocess_msg(msg)
    input_msg_v = vectorizor.transform([cleaned_msg])
    models = [mnb,rf,svc]
    op_list = []
    probabs = []
    for model in models:
        op_list.append(model.predict(input_msg_v)[0])
        probabs.append(model.predict_proba(input_msg_v)[0])
    print("Three model predictions:",op_list)
    output = 0 if op_list.count(0)>op_list.count(1) else 1
    # print(output)
    # return op_list
    op = 'HAM' if output==0 else 'SPAM' if output == 1 else 'ERROR Occured'
    print(op,'\n')
    ham_probab = np.mean([x[0] for x in probabs])
    ham_percent = ham_probab*100
    generate_pie_chart.gen_pie(ham_percent)
    return render_template('result.html',output = op)

if __name__ == '__main__':
    nltk_setup.ensure_nltk_resources()
    app.run(debug=True)