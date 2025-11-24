from flask import Flask, request, render_template, jsonify
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set the Flask application
application = Flask(
    __name__, 
    template_folder='C://Users//Amreet//Desktop//Bagging_Technique//Template'
)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')

    else:
        # Create a CustomData object with form inputs
        data = CustomData(
            Item_Weight = float(request.form.get('Item_Weight')),
            Item_Visibility = float(request.form.get('Item_Visibility')),
            Item_MRP = float(request.form.get('Item_MRP')),
            Outlet_Establishment_Year = int(request.form.get('Outlet_Establishment_Year')),
            Item_Fat_Content = request.form.get('Item_Fat_Content'),
            Item_Type = request.form.get('Item_Type'),
            Outlet_Size = request.form.get('Outlet_Size'),
            Outlet_Location_Type = request.form.get('Outlet_Location_Type'),
            Outlet_Type = request.form.get('Outlet_Type')
        )

        # Convert to dataframe
        final_new_data = data.get_data_as_dataframe()

        # Predict using pipeline
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0], 2)

        return render_template('form.html', final_result=results)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
