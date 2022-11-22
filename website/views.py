from tracemalloc import start
from flask import Blueprint, render_template, request, flash, jsonify, Flask
from flask_login import login_required, current_user
from sqlalchemy import false
from tensorboard import summary
import json
import os
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, HiddenField, TextAreaField, SubmitField
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from wtforms.validators import InputRequired
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, RepeatVector
from sklearn.metrics import mean_squared_error
from math import sqrt
from flask_wtf.csrf import CSRFProtect
import io
from .models import Note
from . import db

app = Flask(__name__)
views = Blueprint('views', __name__)

app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['SECRET_KEY'] = 'supersikreto'

class UploadFileForm(FlaskForm):
    file=FileField("File")
    submit = SubmitField("Upload File")
@views.route('/', methods=['GET', 'POST'])
@views.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    mse = 0
    rmse = 0
    mean = 0
    form = UploadFileForm()
    araw = ''
    model_summaryData = ''
    if form.validate_on_submit():
        file = form.file.data
        if file:
            print('hinahanap',file)
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename('Burat.csv')))
        else:
            print('hinahanap wala laman',file)        
        df = pd.read_csv('website/static/files/Burat.csv', encoding = 'latin-1')
        startDate = request.form['startDate']
        endDate = request.form['endDate']
        if startDate and endDate:     
            try:       
                araw = startDate + ' - ' + endDate
                df2 = df.drop(['ADDRESS','TIME COMMITTED'], axis = 'columns')
                df2['DATE'] = pd.to_datetime(df2['DATE COMMITTED'], infer_datetime_format=True)
                df2.drop(['DATE COMMITTED'], axis = 1, inplace = True)
                df2['CRIME'].unique().tolist()
                dates = pd.pivot_table(df2, index=['DATE'], aggfunc='count')
                idx = pd.date_range(startDate, endDate)
                print(df2)
                print(len(df2['CRIME'].unique().tolist()))
                print(dates)
                print(len(idx))
                dates = dates.reindex(idx, fill_value = 0)
                print(dates)
                plt.figure()
                dates.plot(figsize=(20,5))    
                plt.tight_layout(pad=0.5)
                plt.savefig('website/static/pic2.png',label='xxx')
                plt.figure()
                results = seasonal_decompose(dates['CRIME'])
                results.plot()
                plt.tight_layout(pad=0.5)
                plt.savefig('website/static/pic.png',label='xxx')
                train = dates.iloc[:296]
                test = dates.iloc[296:]    
                # tests(train,test, dates)
                scaler = MinMaxScaler()
                scaler.fit(train)
                scaled_train = scaler.transform(train)
                scaled_test = scaler.transform(test)
                scaled_train[:10]
                n_input = 3
                n_features = 1
                generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
                X,y = generator[0]
                print(f'Given the Array: \n{X.flatten()}')
                print(f'Predict this y: \n {y}')
                X.shape
                n_input = 12
                generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
                model = Sequential()
                model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_input, n_features)))
                model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_input, n_features)))
                model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_input, n_features)))
                model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_input, n_features)))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Flatten())
                model.add(RepeatVector(1))
                model.add(LSTM(64, activation='relu', return_sequences=True))
                model.add(Flatten())
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')
                model.summary()
                s = io.StringIO()
                model.summary(print_fn=lambda x: s.write(x + '\n'))
                model_summary = s.getvalue()
                print(model_summary)
                plt.figure()
                plt.tight_layout(pad=0.5)
                plt.text(0, 0, model_summary)
                plt.axis('off')
                plt.savefig("out.png", bbox_inches = "tight")
                
                model.fit(generator,epochs=50)
                loss_per_epoch = model.history.history['loss']
                plt.figure()
                plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
                plt.tight_layout(pad=0.5)
                plt.savefig('website/static/pic3.png')
                last_train_batch = scaled_train[-12:]

                last_train_batch = last_train_batch.reshape((1, n_input, n_features))

                model.predict(last_train_batch)
                scaled_test[0]
                test_predictions = []

                first_eval_batch = scaled_train[-n_input:]
                current_batch = first_eval_batch.reshape((1, n_input, n_features))

                for i in range(len(test)):
                    
                    # first batch
                    current_pred = model.predict(current_batch)[0]
                    
                    # prediction into the array
                    test_predictions.append(current_pred) 
                    
                    # prediction to update the batch and remove the first value
                    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
                true_predictions = scaler.inverse_transform(test_predictions)

                test['Predictions'] = true_predictions
                plt.figure()
                test.plot(figsize=(14,5))
                plt.tight_layout(pad=0.5)
                plt.savefig('website/static/pic4.png')
                rmse=sqrt(mean_squared_error(test['CRIME'],test['Predictions']))
                mse = rmse ** 2
                mean = dates['CRIME'].mean()
                print(f'MSE Error: {mse}\nRMSE Error: {rmse}\nMean: {mean}')                
                print('araw',araw)
            except:
                print('error')
        else:
            print('No data')
    df = pd.read_csv('website/static/files/Burat.csv', encoding = 'latin-1')
    #print(df.to_json)
    return render_template("new.html", user=current_user, form=form, tables=[df.to_html()],titles=[''], mseData = mse, rmseData = rmse, meanData = mean, araw=araw)
    # return render_template("new.html", user=current_user)

@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data)
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})
