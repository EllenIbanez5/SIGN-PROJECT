from flask import Flask, render_template, request, redirect, url_for, flash
from model import model, predd, severity

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Pastikan severity memiliki kolom 'Symptom'
symptoms = severity['Symptom'].tolist() if 'Symptom' in severity.columns else []

@app.route('/')
def landing():
    # Pastikan landing.html ada di folder templates
    return render_template('landing.html')

@app.route('/about')
def about():
    # Pastikan about.html ada di folder templates
    return render_template('about.html')

@app.route('/check', methods=['GET', 'POST'])
def check():
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        
        # Validasi jumlah gejala
        if len(selected_symptoms) < 3:
            flash('Please select at least 3 symptoms.')
            return redirect(url_for('check'))
        elif len(selected_symptoms) > 17:
            flash('You can select a maximum of 17 symptoms.')
            return redirect(url_for('check'))

        # Isi gejala hingga 17 jika kurang
        inputs = selected_symptoms + [0] * (17 - len(selected_symptoms))
        
        # Pastikan predd mengembalikan 3 nilai yang benar
        try:
            disease, description, precautions = predd(model, *inputs)
            return render_template('result.html', disease=disease, description=description, precautions=precautions)
        except Exception as e:
            flash(f"An error occurred during prediction: {str(e)}")
            return redirect(url_for('check'))

    # Pastikan check.html ada di folder templates
    return render_template('check.html', symptoms=symptoms)

if __name__ == '__main__':
    app.run(debug=True, port=1111)
