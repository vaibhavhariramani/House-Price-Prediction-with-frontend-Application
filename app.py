import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
    

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    for x in request.form.values():
        print(x)
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]

    features_name = ['UNDER_CONSTRUCTION','RERA','BHK_NO.','RESALE','LONGITUDE','LATITUDE','area','POSTED_BY Dealer','POSTED_BY Owner','BHK_OR_RK','city_tier tier2','city_tier tier3']
    df = pd.DataFrame(final_features, columns=features_name)
    print(df.head())
    output = model.predict(df)

    output1 = round(output[0], 7)

    return render_template('index.html', prediction_text='House price should be $ {}'.format(output1))


if __name__ == "__main__":
    app.run(debug=True)
