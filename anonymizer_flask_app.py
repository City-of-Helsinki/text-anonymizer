from flask import Flask, render_template, request, session, send_file
from flask_session import Session
import pandas as pd
from text_anonymizer import TextAnonymizer
from text_anonymizer.default_settings import DEFAULT_SETTINGS
from werkzeug.utils import secure_filename
import io
import os
import logging

logging.getLogger().setLevel(logging.WARN)

app = Flask(__name__, template_folder='flask/templates', static_folder='flask/static')
app.logger.setLevel(logging.INFO)

app.secret_key = os.getenv("SECRET_KEY", "@I{&33dy647GyIwP74qzq6(j0'CXX1o{")
app.config['SESSION_TYPE'] = 'filesystem'
app.config["SESSION_PERMANENT"] = False
Session(app)

app.config['UPLOAD_FOLDER'] = 'uploads'  # Define a folder to store uploaded files
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size, here set to 16MB

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

LANGUAGES = ['fi', 'en']

# Init anonymizer as singleton
text_anonymizer = TextAnonymizer(languages=LANGUAGES, debug_mode=False)

# User recognizers
recognizer_options = list(DEFAULT_SETTINGS.mask_mapppings.keys())


@app.route("/", methods=["GET"])
def index():
    # Pääsivu, jossa on linkit tai napit, jotka ohjaavat käyttäjän oikeaan lomakkeeseen
    return render_template("index.html")

@app.route("/plain_text", methods=["GET", "POST"])
def plain_text():
    if request.method == "POST":
        return handle_text_anonymization(request)
    else:
        return render_template("plain_text.html", languages=LANGUAGES, recognizer_options=recognizer_options)

@app.route("/text_file", methods=["GET", "POST"])
def text_file():
    if request.method == "POST":
        if "file" in request.files:
            return handle_text_file_anonymization(request)
    return render_template("text_file.html", languages=LANGUAGES, recognizer_options=recognizer_options)

@app.route("/csv", methods=["GET", "POST"])
def csv():
    if request.method == "POST":
        if "file" in request.files:
            return handle_csv_upload(request)
        else:
            return handle_csv_anonymization(request)
    else:
        return render_template("csv.html", recognizer_options=recognizer_options, phase="upload")

def handle_csv_upload(request):
    
    uploaded_file = None
    if 'file' in request.files:
        uploaded_file = request.files['file']
    # Handle uploaded file
    
    # Check if there is a file and it has a CSV filename
    if uploaded_file and uploaded_file.filename.endswith('.csv'):
        try:
            separator = request.form.get('separator', ',')
            encoding = request.form.get('encoding', 'utf-8')
            filename = secure_filename(uploaded_file.filename)

            # Read the contents of the file into a DataFrame without saving it to disk
            file_stream = io.StringIO(uploaded_file.stream.read().decode(encoding), newline=None)
            dataframe = pd.read_csv(file_stream, sep=separator, dtype=str, encoding=encoding, index_col=False)

            # Store the DataFrame in the session after converting it to JSON
            session['dataframe'] = dataframe.to_json(orient='split', date_format='iso')
            session['filename'] = filename
            columns = dataframe.columns.tolist()

            return render_template('csv.html',
                                   columns=columns,
                                   languages=LANGUAGES,
                                   recognizer_options=recognizer_options,
                                   phase="column_selection")
        except Exception as e:
            app.logger.exception('Csv upload failed: %s', str(e))

    return render_template("csv.html",
                           error="Tiedoston lukeminen ei onnistunut. Tarkista erotinmerkki ja merkistö.",
                           phase="upload")

def handle_csv_anonymization(request):
    # If no columns selected, return to page with column selection
    column_selection = None
    if request.form:
        column_selection = request.form.getlist('columns')

    
    if not column_selection or len(column_selection) == 0:
        
        if 'dataframe' in session:
            app.logger.info("Dataframe not in session. Forward to column selection page.")
            dataframe_json = session['dataframe']
            dataframe = pd.read_json(dataframe_json, orient='split')
            columns = dataframe.columns.tolist()
            
            return render_template('csv.html',
                                   columns=columns,
                                   error="Valitse sarakkeet, jotka haluat anonymisoida.",
                                   recognizer_options=recognizer_options,
                                   phase="column_selection")
        else:
            # Return an error message or redirect if there is no dataframe in the session
            return render_template("csv.html", error="Sessio on vanhentunut.",
                                   phase="upload")

    else:
        # If columns selected and data is in session, anonymize them and return the anonymized file
        if 'dataframe' in session:
            app.logger.info("Dataframe found in session. Anonymizing...")
            try:
                recognizers = request.form.getlist('recognizers')
                dataframe_json = session['dataframe']
                dataframe = pd.read_json(io.StringIO(dataframe_json), orient='split')
                encoding = request.form.get('encoding', 'utf-8')
                user_languages = request.form.getlist('user_languages')
                app.logger.info(f"Got dataframe in json format, encoding is {encoding}, user_languages: {user_languages}")
                # If columns selected, anonymize them
                for column in column_selection:
                    app.logger.info(f"Anonymizing column {column[0:1]}", )
                    dataframe[column] = dataframe[column].apply(
                        lambda x: text_anonymizer.anonymize(x, user_recognizers=recognizers, user_languages=user_languages).anonymized_text
                    )

                resp = io.StringIO()
                dataframe.to_csv(resp, encoding=encoding, index=False)
                resp.seek(0)

                # add _anonymized to original filename
                filename = secure_filename(session['filename'])
                filename = filename.replace('.csv', '_anonymized.csv')
                app.logger.info("Anonymization done. Returning anonymized csv.")

                return send_file(
                    io.BytesIO(resp.getvalue().encode(encoding)),
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name=filename
                )
            except Exception as e:
                app.logger.exception('Csv anonymization failed with exception: %s', str(e))
                return render_template("csv.html", error="Tiedoston anonymisointi ei onnistunut.",
                                       phase="upload")
        else:
            return render_template("csv.html", error="Ei tiedostoa",
                                  phase="upload")

def handle_text_file_anonymization(request):
    uploaded_file = None
    if 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file and uploaded_file.filename.endswith('.txt'):
            try:
                encoding = request.form.get('encoding', 'utf-8')
                input_text = uploaded_file.stream.read().decode(encoding)
                recognizers = request.form.getlist('recognizers')
                user_languages = request.form.getlist('user_languages')
                anonymized_str = text_anonymizer.anonymize(input_text, user_recognizers=recognizers,
                                                           user_languages=user_languages).anonymized_text
                # add _anonymized to original filename

                filename = secure_filename(uploaded_file.filename)
                filename = filename.replace('.txt', '_anonymized.txt')
                app.logger.info(f"Text file uploaded, encoding is {encoding}, user_languages: {user_languages}")
                return send_file(
                    io.BytesIO(anonymized_str.encode(encoding)),
                    mimetype='plain/text',
                    as_attachment=True,
                    download_name=filename
                )

            except Exception as e:
                app.logger.exception('Error handling txt file: %s', str(e))
                return render_template("text_file.html",
                                       languages=LANGUAGES,
                                       error="Tiedoston anonymisointi ei onnistunut. Tarkista onko merkistö oikein.",
                                   recognizer_options=recognizer_options)

    return render_template("text_file.html", recognizer_options=recognizer_options)


def handle_text_anonymization(request):
    text = request.form['text']
    recognizers = request.form.getlist('recognizers')
    user_languages = request.form.getlist('user_languages')
    app.logger.info(f"Text form uploaded, user_languages: {user_languages}")
    anonymized_text = text_anonymizer.anonymize(text, user_recognizers=recognizers, user_languages=user_languages).anonymized_text.strip()
    return render_template("plain_text.html", anonymized_text=anonymized_text,
                           languages=LANGUAGES,
                           recognizer_options=recognizer_options, text=text)


if __name__ == "__main__":

    app.run(debug=False)