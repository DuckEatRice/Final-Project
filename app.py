import numpy as np
import pandas as pd

#from google.colab import drive
#drive.mount('/content/drive')

#df= pd.read_csv("/content/drive/My Drive/Stress_level/Teen_Mental_Health_Dataset.csv")
df = pd.read_csv("Teen_Mental_Health_Dataset.csv")
df.head()

df.info()

df.isnull().sum()

df.tail()

df.columns

import pandas as pd
import plotly.express as px

fig = px.bar(
    df['platform_usage'].value_counts().reset_index(),
    x='platform_usage',
    y='count',
    title='Most Used Platforms'
)

fig.show()

fig = px.pie(
    df,
    names='age',
    title='Age Distribution'
)

fig.show()

gender_usage = df.groupby('gender')['daily_social_media_hours'].mean().reset_index()

fig = px.bar(
    gender_usage,
    x='gender',
    y='daily_social_media_hours',
    title='Average Social Media Usage by Gender'
)

fig.show()

fig = px.scatter(
    df,
    x='screen_time_before_sleep',
    y='sleep_hours',
    title='Screen Time Before Sleep vs Sleep Hours'
)

fig.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


df_model = df.drop(columns=['anxiety_level', 'addiction_level']).copy()
df_model.columns = df_model.columns.str.strip()


# encode categorcal columns
le_gender   = LabelEncoder()
le_platform = LabelEncoder()
le_social   = LabelEncoder()

df_model['gender_enc']   = le_gender.fit_transform(df_model['gender'])
df_model['platform_enc'] = le_platform.fit_transform(df_model['platform_usage'])
df_model['social_enc']   = le_social.fit_transform(df_model['social_interaction_level'])

print('Gender encodings:  ', dict(zip(le_gender.classes_,   le_gender.transform(le_gender.classes_))))
print('Platform encodings:', dict(zip(le_platform.classes_, le_platform.transform(le_platform.classes_))))
print('Social encodings:  ', dict(zip(le_social.classes_,   le_social.transform(le_social.classes_))))



# this function rescales a value to 0-1
# for example, values  in the sleep_hours column range from 4-9. A value of 4 would return 0 and a value of 9 would return 1.
def minmax(series):
    return (series - series.min()) / (series.max() - series.min())

stress_score = (
    (1 - minmax(df_model['sleep_hours']))           * 2.5 +
    minmax(df_model['daily_social_media_hours'])     * 2.0 +
    minmax(df_model['screen_time_before_sleep'])     * 1.5 +
    (1 - minmax(df_model['academic_performance']))   * 1.5 +
    df_model['depression_label']                     * 1.5 +
    (1 - minmax(df_model['physical_activity']))      * 1.0
)

# same thing. rescales to a value from 0-1
stress_score = 1 + 9 * (stress_score - stress_score.min()) / (stress_score.max() - stress_score.min())
df_model['stress_level'] = stress_score.round().astype(int).clip(1, 10)

print('Engineered stress_level distribution:')
print(df_model['stress_level'].value_counts().sort_index())

fig = px.histogram(df_model, x='stress_level', nbins=10,
                   title='Engineered Stress Level Distribution')
fig.show()

feature_cols = [
    'age', 'gender_enc', 'daily_social_media_hours', 'platform_enc',
    'sleep_hours', 'screen_time_before_sleep', 'academic_performance',
    'physical_activity', 'social_enc', 'depression_label'
]

X = df_model[feature_cols]
y = df_model['stress_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Training samples : {len(X_train)}')
print(f'Test samples     : {len(X_test)}')

models = {
    'Linear Regression':  LinearRegression(),
    'Random Forest':      RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
    'Gradient Boosting':  GradientBoostingRegressor(n_estimators=200, random_state=42,
                                                     max_depth=5, learning_rate=0.05),
}

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

results = {}
for name, model in models.items():
    X_tr = X_train_sc if name == 'Linear Regression' else X_train
    X_te = X_test_sc  if name == 'Linear Regression' else X_test
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    results[name] = {
        'MAE':   mean_absolute_error(y_test, preds),
        'RMSE':  np.sqrt(mean_squared_error(y_test, preds)),
        'R²':    r2_score(y_test, preds),
        'preds': preds
    }
    print(f'{name:25s}  MAE={results[name]["MAE"]:.3f}  '
          f'RMSE={results[name]["RMSE"]:.3f}  R²={results[name]["R²"]:.3f}')

best_name = max(results, key=lambda m: results[m]['R²'])
best_model = models[best_name]
print(f'\nBest model: {best_name}')

metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE':   [results[m]['MAE']  for m in results],
    'RMSE':  [results[m]['RMSE'] for m in results],
    'R²':    [results[m]['R²']   for m in results],
})

fig = px.bar(
    metrics_df.melt(id_vars='Model', value_vars=['MAE', 'RMSE', 'R²'],
                    var_name='Metric', value_name='Score'),
    x='Model', y='Score', color='Metric', barmode='group',
    title='Model Comparison – Regression Metrics'
)
fig.show()
metrics_df

best_preds = results[best_name]['preds']
scatter_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': best_preds})

fig2 = px.scatter(
    scatter_df, x='Actual', y='Predicted',
    title=f'{best_name} – Actual vs Predicted Stress Level',
    opacity=0.5
)
fig2.add_trace(go.Scatter(
    x=[1, 10], y=[1, 10], mode='lines', name='Perfect fit',
    line=dict(color='red', dash='dash')
))
fig2.show()

tree_name = 'Gradient Boosting' if results['Gradient Boosting']['R²'] > results['Random Forest']['R²'] else 'Random Forest'
importance_df = pd.DataFrame({
    'Feature':    feature_cols,
    'Importance': models[tree_name].feature_importances_
}).sort_values('Importance', ascending=False)

fig3 = px.bar(
    importance_df, x='Importance', y='Feature', orientation='h',
    title=f'Feature Importance ({tree_name})'
)
fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
fig3.show()

#!pip install gradio -q

import gradio as gr

gender_map   = dict(zip(le_gender.classes_,   le_gender.transform(le_gender.classes_).tolist()))
platform_map = dict(zip(le_platform.classes_, le_platform.transform(le_platform.classes_).tolist()))
social_map   = dict(zip(le_social.classes_,   le_social.transform(le_social.classes_).tolist()))

def predict_stress(
    age, gender, daily_social_media_hours, platform_usage,
    sleep_hours, screen_time_before_sleep, academic_performance,
    physical_activity, social_interaction_level, depression_label
):
    input_data = pd.DataFrame([{
        'age':                      int(age),
        'gender_enc':               gender_map[gender],
        'daily_social_media_hours': float(daily_social_media_hours),
        'platform_enc':             platform_map[platform_usage],
        'sleep_hours':              float(sleep_hours),
        'screen_time_before_sleep': float(screen_time_before_sleep),
        'academic_performance':     float(academic_performance),
        'physical_activity':        float(physical_activity),
        'social_enc':               social_map[social_interaction_level],
        'depression_label':         int(depression_label)
    }])

    prediction = float(best_model.predict(input_data[feature_cols])[0])
    prediction = round(max(1.0, min(10.0, prediction)), 2)

    if prediction <= 3:
        label = '🟢 Low'
    elif prediction <= 6:
        label = '🟡 Moderate'
    else:
        label = '🔴 High'

    return f"{prediction} / 10  ({label} stress)"


genders       = list(gender_map.keys())
platforms     = list(platform_map.keys())
social_levels = list(social_map.keys())

with gr.Blocks(title='Teen Stress Level Predictor') as demo:
    gr.Markdown('## 🧠 Teen Stress Level Predictor')
    gr.Markdown(f'Model: **{best_name}**  |  R² = {results[best_name]["R²"]:.3f}  |  MAE = {results[best_name]["MAE"]:.3f}')

    with gr.Row():
        with gr.Column():
            gr.Markdown('### Demographics')
            age    = gr.Slider(13, 19, value=16, step=1, label='Age')
            gender = gr.Radio(genders, label='Gender', value=genders[0])

        with gr.Column():
            gr.Markdown('### Social Media')
            daily_social_media_hours = gr.Slider(1.0, 8.0, value=4.5, step=0.1,
                                                  label='Daily Social Media Hours')
            platform_usage = gr.Dropdown(platforms, label='Platform', value=platforms[0])

    with gr.Row():
        with gr.Column():
            gr.Markdown('### Sleep & Screen Time')
            sleep_hours               = gr.Slider(4.0, 9.0, value=7.0, step=0.1,
                                                   label='Sleep Hours')
            screen_time_before_sleep  = gr.Slider(0.5, 3.0, value=1.5, step=0.1,
                                                   label='Screen Time Before Sleep (hrs)')

        with gr.Column():
            gr.Markdown('### Lifestyle')
            academic_performance      = gr.Slider(2.0, 4.0, value=3.0, step=0.01,
                                                   label='Academic Performance (GPA)')
            physical_activity         = gr.Slider(0.0, 2.0, value=1.0, step=0.1,
                                                   label='Physical Activity (hrs/day)')
            social_interaction_level  = gr.Radio(social_levels, label='Social Interaction Level',
                                                  value=social_levels[0])

    with gr.Row():
        with gr.Column():
            gr.Markdown('### Mental Health')
            depression_label = gr.Radio([0, 1], label='Depression (0 = No, 1 = Yes)', value=0)

    predict_btn = gr.Button('Predict Stress Level', variant='primary')
    output      = gr.Textbox(label='Predicted Stress Level', interactive=False)

    predict_btn.click(
        fn=predict_stress,
        inputs=[
            age, gender, daily_social_media_hours, platform_usage,
            sleep_hours, screen_time_before_sleep, academic_performance,
            physical_activity, social_interaction_level, depression_label
        ],
        outputs=output
    )

demo.launch(share=True)
