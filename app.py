from flask import Flask, render_template, Response
import pandas as pd
import risk_profiling_be as rpb
import output_integration_be as oib
import strategic_analysis_be as sab
# ------------------------------ Library Corner -----------------------------

app = Flask(__name__)

# -------------------------------- App Functions --------------------------------
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/<employee_dataset>/Raw', methods=['GET', 'POST'])
def show_raw(employee_dataset):
    raw_dataset_name = employee_dataset + '.csv'
    raw_dataset_directory = 'test_csv/' + raw_dataset_name
    raw_dataset = pd.read_csv(raw_dataset_directory)
    return raw_dataset.sample(7).to_html()

@app.route('/<employee_dataset>/Dashboard', methods=['GET' ,'POST'])
def show_dashboard(employee_dataset):
    dashboard = rpb.sample_predict(employee_dataset)
    return Response(dashboard.to_html(), mimetype='text/html')

@app.route('/<employee_dataset>/ViewInsights', methods=['GET', 'POST'])
def show_view_insights(employee_dataset):
    insights = rpb.combined_dataset(employee_dataset)
    return Response(insights.to_html(), mimetype='text/html')

# --------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)