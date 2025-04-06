from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pf
from sklearn.preprocessing import StandardScaler






application=Flask(__name__)
app=application












## import ridgrer regressor and standoard scaler pickel

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))


## Creating the home page :
@app.route('/')
def index():
    return  render_template('index.html')


@app.route('/predict_datapoint',methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        Tempeature=float(request.form.get('Temperature'))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("WS"))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        
        new_data_scaled=standard_scaler.transform([[Tempeature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        # print(new_data_scaled)
        result=ridge_model.predict(new_data_scaled)
        # print(result)
        
        return render_template('home.html',results=result[0])
        
        
    else:
        return render_template('home.html')






if __name__=='__main__':
    app.run(host='0.0.0.0')









