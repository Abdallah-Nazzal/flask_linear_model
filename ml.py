from flask import Flask,render_template,request
import json
import pandas as pd
import numpy as np
import pickle
import os



app = Flask(__name__)



@app.route('/')
def hello():
    return 'Hello, World!'


       
@app.route('/predict', methods = ["GET", "POST"])
def predict_house_price():
 
    #   if the request is POST
    if request.method == 'POST':


        #   get the features as dictionary 
        features = request.get_json()
      
        #   load the linear model
        loaded_model = pickle.load(open('./model.sav', 'rb'))
        
        #  use the loaded model to predict  
        x = np.array(features["features"]).reshape(1,6)
        result = loaded_model.predict(x)

        if x.shape[0] == 1:
            result = result[0]

        print(result)

        #   return price as a string 
        return json.dumps(result)



    else:
        
        return 'hello world'

         

    
    


if __name__ == '__main__':
    

    app.run(debug=True, use_reloader=False,host ='0.0.0.0', port = int(os.environ.get('PORT', 8000))

