from flask import Flask,render_template,request
import json
import pandas as pd

theta0 = None
thetas = None



app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'


       
@app.route('/predict', methods = ["GET", "POST"])
def predict_house_price():
 
    #if the request is POST
    if request.method == 'POST':

        # get the features as dictionary 
        features = request.get_json()
        price = 0
        
        # for loop to calculate the price by multiply feature and thetas
        for index, feature in enumerate(features["features"]):
            price += float(feature) * float(thetas[index])

        price += theta0    

        # return price as a string
        return json.dumps(price)



    else:
        return 'hello world'

         

    
    


if __name__ == '__main__':

    # read the parameters from the csv file
    parameters = pd.read_csv('lm_parameters.csv')

    
    thetas = parameters.columns 
    theta0 = pd.read_csv('lm_parameters.csv').iloc[0,0]

    app.run(debug=True, use_reloader=False)

