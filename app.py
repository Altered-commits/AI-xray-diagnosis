from flask import Flask, request, render_template, redirect, url_for
from backend.ml.model_runner import run_model
from keras.api.models import load_model
import os

app = Flask(__name__)

model = load_model(r'backend\models\best_model.keras')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']

        img_path = './temp.jpg'
        file.save(img_path)

        prediction = run_model(model, img_path)

        if(os.path.exists(img_path)):
            os.remove(img_path)

        return redirect(url_for('home', prediction=prediction))

    prediction = request.args.get('prediction')

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
