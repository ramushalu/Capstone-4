Applied AI & ML Essentials Capstone
Part 3 — Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline
Student Name: Aashika R
-------------------------------------------------------------------------
from google.colab import files
# files.download("best_model.pkl")
-------------------------------------------------------------------------
from google.colab import files

print("Please select 'best_model.pkl' to upload:")
uploaded = files.upload()

if 'best_model.pkl' in uploaded:
  print('best_model.pkl uploaded successfully.')
else:
  print('best_model.pkl was not uploaded.') 
  
Output:

Please select 'best_model.pkl' to upload:
No file chosen Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving best_model.pkl to best_model.pkl
best_model.pkl uploaded successfully.
-------------------------------------------------------------------------------------------------------
import os

# List files in the current directory
print(os.listdir('.')) 

Output:
['.config', 'best_model.pkl', 'sample_data']
------------------------------------------------------------------------------------------------------
# PART 4 - TASK 1
# Install Required Libraries

!pip install openai requests jsonschema joblib pandas scikit-learn -q

# Import Required Libraries

import os
import json
import re
import requests
import joblib
import pandas as pd

from jsonschema import validate, ValidationError

url = "https://api.openai.com/v1/chat/completions"

def call_llm(
    system_prompt,
    user_prompt,
    temperature=0.0,
    max_tokens=512
):
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {os.environ.get('LLM_API_KEY')}",
        "Content-Type": "application/json"
    }

    # Debug print statement to see what headers are being sent
    print("DEBUG: Headers being sent:", headers)

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        print("Error Status Code:", response.status_code)
        print(response.text)
        return None

    result = response.json()

    return result["choices"][0]["message"]["content"]

# Load Best Model

best_model = joblib.load("best_model.pkl")

print("="*60)
print("BEST MODEL LOADED SUCCESSFULLY")
print("="*60)

print(best_model)

Output:
============================================================
BEST MODEL LOADED SUCCESSFULLY
============================================================
Pipeline(steps=[('simpleimputer', SimpleImputer(strategy='median')),
                ('standardscaler', StandardScaler()),
                ('randomforestclassifier',
                 RandomForestClassifier(max_depth=10, n_estimators=200,
                                        random_state=42))])
------------------------------------------------------------------------------------
# Store API Key

from google.colab import userdata

# Retrieve the API key from Colab secrets
api_key = userdata.get('OPENAI_API_KEY')

# Set the environment variable
os.environ["LLM_API_KEY"] = api_key

print("LLM_API_KEY environment variable set.")

Output:
LLM_API_KEY environment variable set.
----------------------------------------------------------------------------------------
import os

print(f"LLM_API_KEY: {os.environ.get('LLM_API_KEY')}")
----------------------------------------------------------------------------------------
from google.colab import userdata
userdata.get('OPENAI_API_KEY')
----------------------------------------------------------------------------------------
# PART 4 - TASK 5
# Test API

system_prompt = "You are a helpful assistant."

user_prompt = "Reply with only the word: hello"

response = call_llm(
    system_prompt,
    user_prompt,
    temperature=0,
    max_tokens=20
)

print("="*60)
print("LLM TEST")
print("="*60)

print(response)

Output:
============================================================
LLM TEST
============================================================

hello
---------------------------------------------------------------------
# PART 4 - TASK 6
# PII Guardrail

import re

def has_pii(text):

    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'

    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'

    return bool(
        re.search(email_pattern, text) or
        re.search(phone_pattern, text)
    )
---------------------------------------------------------------------------------
# Test 1 (Should be blocked)

test_input1 = "My email is student@gmail.com"

print("="*60)
print("PII TEST 1")
print("="*60)

if has_pii(test_input1):
    print("Input blocked: PII detected.")
else:
    print("Safe input")

Output:
============================================================
PII TEST 1
============================================================
Input blocked: PII detected.
-------------------------------------------------------------------------------
# Test 2 (Should pass)

test_input2 = "Age is 45 and BMI is 28.4"

print("="*60)
print("PII TEST 2")
print("="*60)

if has_pii(test_input2):
    print("Input blocked: PII detected.")
else:
    print("Safe input")

Output:
============================================================
PII TEST 2
============================================================
Safe input
-------------------------------------------------------------------------------
# PART 4 - TASK 7
# JSON Schema

schema = {

    "type": "object",

    "properties": {

        "prediction_label": {
            "type": "string"
        },

        "confidence_level": {
            "type": "string"
        },

        "top_reason": {
            "type": "string"
        },

        "second_reason": {
            "type": "string"
        },

        "next_step": {
            "type": "string"
        }

    },

    "required": [

        "prediction_label",

        "confidence_level",

        "top_reason",

        "second_reason",

        "next_step"

    ]

}

print("="*60)
print("JSON SCHEMA CREATED")
print("="*60)

Output:
============================================================
JSON SCHEMA CREATED
============================================================
--------------------------------------------------------------------
# Convert Inputs

sample_df = pd.DataFrame(sample_inputs)

print(sample_df)

Output:
  age   bmi  children  sex_male  smoker_yes  region_northwest  \
0   25  24.5         0         1           0                 0   
1   55  36.8         2         0           1                 0   
2   40  30.2         1         1           0                 1   

   region_southeast  region_southwest  
0                 0                 1  
1                 1                 0  
2                 0                 0  
---------------------------------------------------------------------------
# Model Prediction

predictions = best_model.predict(sample_df)

probabilities = best_model.predict_proba(sample_df)[:,1] 
---------------------------------------------------------------------------
# System Prompt

system_prompt = """
You are an AI model explanation assistant.

Given:

1. Feature values
2. Predicted class
3. Prediction probability

Return ONLY valid JSON.

Required format:

{
"prediction_label":"",
"confidence_level":"",
"top_reason":"",
"second_reason":"",
"next_step":""
}
"""
-----------------------------------------------------------------
# Generate Explanations

print("="*70)
print("MODEL PREDICTION EXPLANATIONS")
print("="*70)

for i in range(len(sample_inputs)):

    feature_text = json.dumps(sample_inputs[i], indent=2)

    user_prompt = f"""

Feature Values:

{feature_text}

Predicted Class:

{predictions[i]}

Prediction Probability:

{probabilities[i]:.4f}

Explain the prediction.
Return ONLY JSON.
"""

    # Guardrail

    if has_pii(user_prompt):

        print("Input blocked: PII detected.")
        continue

    response = call_llm(

        system_prompt,

        user_prompt,

        temperature=0,

        max_tokens=300

    )

    print("\n")
    print("="*60)
    print(f"INPUT {i+1}")
    print("="*60)

    print(feature_text)

    print("\nPredicted Class :", predictions[i])

    print("Probability :", round(probabilities[i],4))

    print("\nRAW LLM RESPONSE")

    print(response)

Output:
======================================================================
MODEL PREDICTION EXPLANATIONS
======================================================================


============================================================
INPUT 1
============================================================
{
  "age": 25,
  "bmi": 24.5,
  "children": 0,
  "sex_male": 1,
  "smoker_yes": 0,
  "region_northwest": 0,
  "region_southeast": 0,
  "region_southwest": 1
}

Predicted Class : 0
Probability : 0.1425

RAW LLM RESPONSE

{
  "prediction_label": "Low Medical Cost",
  "confidence_level": "High",
  "top_reason": "The individual is a non-smoker with a healthy BMI.",
  "second_reason": "Young age and no dependent children reduce expected medical costs.",
  "next_step": "Continue maintaining a healthy lifestyle."
}

============================================================
INPUT 2
============================================================
{
  "age": 55,
  "bmi": 36.8,
  "children": 2,
  "sex_male": 0,
  "smoker_yes": 1,
  "region_northwest": 0,
  "region_southeast": 1,
  "region_southwest": 0
}

Predicted Class : 1
Probability : 0.9834

RAW LLM RESPONSE

{
  "prediction_label": "High Medical Cost",
  "confidence_level": "Very High",
  "top_reason": "Smoking status is the strongest factor contributing to higher predicted medical costs.",
  "second_reason": "Higher BMI and older age further increase the prediction.",
  "next_step": "Consider preventive healthcare and regular medical check-ups."
}
============================================================
INPUT 3
============================================================
{
  "age": 40,
  "bmi": 30.2,
  "children": 1,
  "sex_male": 1,
  "smoker_yes": 0,
  "region_northwest": 1,
  "region_southeast": 0,
  "region_southwest": 0
}

Predicted Class : 0
Probability : 0.3517

RAW LLM RESPONSE

{
  "prediction_label": "Moderate Medical Cost",
  "confidence_level": "Medium",
  "top_reason": "The person is a non-smoker, which lowers predicted medical costs.",
  "second_reason": "BMI is moderately high but age is not in the highest-risk category.",
  "next_step": "Maintaining a healthy weight may help reduce future health risks."
}

------------------------------------------------------------------------
# PART 4 - TASK 9
# JSON Validation

import json
from jsonschema import validate, ValidationError

print("="*70)
print("JSON VALIDATION")
print("="*70)

# FIX: Redefine call_llm with the correct model name for OpenAI API
# This redefinition will apply within this cell and subsequent cells.
# The original call_llm in GKMlwa5Q0hNh used "openai/gpt-4o-mini" with api.openai.com,
# which is an invalid model ID for that endpoint.
def call_llm(system_prompt, user_prompt, model_name="gpt-4o-mini", temperature=0.0, max_tokens=512):
    # Ensure the correct OpenAI API URL is used, as defined in GKMlwa5Q0hNh
    url = "https://api.openai.com/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {os.environ.get('LLM_API_KEY')}",
        "Content-Type": "application/json"
    }

    # Debug print statement to see what headers are being sent
    print("DEBUG: Headers being sent:", headers)

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        print("Error Status Code:", response.status_code)
        print(response.text)
        return None

    result = response.json()

    return result["choices"][0]["message"]["content"]


for i in range(len(sample_inputs)):

    feature_text = json.dumps(sample_inputs[i], indent=2)

    user_prompt = f"""
Feature Values:
{feature_text}

Predicted Class:
{predictions[i]}

Prediction Probability:
{probabilities[i]:.4f}

Explain the prediction.
Return ONLY JSON.
"""

    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        continue

    response = call_llm(
        system_prompt,
        user_prompt,
        temperature=0,
        max_tokens=300
    )

    print("\n")
    print("="*60)
    print(f"INPUT {i+1}")
    print("="*60)

    print("Raw Response:")
    print(response)

    try:

        parsed = json.loads(response.strip())

        validate(instance=parsed, schema=schema)

        print("\nValidation Status : PASS")

    except json.JSONDecodeError:

        print("\nValidation Status : FAIL")
        print("Reason : Invalid JSON")

        parsed = {
            "prediction_label": None,
            "confidence_level": None,
            "top_reason": None,
            "second_reason": None,
            "next_step": None
        }

    except ValidationError as e:

        print("\nValidation Status : FAIL")
        print(e.message)

        parsed = {
            "prediction_label": None,
            "confidence_level": None,
            "top_reason": None,
            "second_reason": None,
            "next_step": None
        }

    print("\nValidated Output")
    print(parsed)
  
Output:
======================================================================
JSON VALIDATION
======================================================================

============================================================
INPUT 1
============================================================
Raw Response:

{
  "prediction_label": "Low Medical Cost",
  "confidence_level": "High",
  "top_reason": "The individual is a non-smoker with a healthy BMI.",
  "second_reason": "Young age and absence of dependent children reduce expected medical costs.",
  "next_step": "Continue maintaining a healthy lifestyle."
}

Validation Status : PASS

Validated Output

{
  'prediction_label': 'Low Medical Cost',
  'confidence_level': 'High',
  'top_reason': 'The individual is a non-smoker with a healthy BMI.',
  'second_reason': 'Young age and absence of dependent children reduce expected medical costs.',
  'next_step': 'Continue maintaining a healthy lifestyle.'
} 

============================================================
INPUT 2
============================================================
Raw Response:

{
  "prediction_label": "High Medical Cost",
  "confidence_level": "Very High",
  "top_reason": "Smoking is the strongest contributor to increased medical costs.",
  "second_reason": "Higher BMI and older age further increase the predicted risk.",
  "next_step": "Regular health checkups and lifestyle improvements are recommended."
}

Validation Status : PASS

Validated Output

{
  'prediction_label': 'High Medical Cost',
  'confidence_level': 'Very High',
  'top_reason': 'Smoking is the strongest contributor to increased medical costs.',
  'second_reason': 'Higher BMI and older age further increase the predicted risk.',
  'next_step': 'Regular health checkups and lifestyle improvements are recommended.'
} 

============================================================
INPUT 3
============================================================
Raw Response:

{
  "prediction_label": "Moderate Medical Cost",
  "confidence_level": "Medium",
  "top_reason": "The individual is a non-smoker, reducing the likelihood of very high medical costs.",
  "second_reason": "BMI is moderately elevated, contributing to some health risk.",
  "next_step": "Maintaining a healthy weight and regular exercise is recommended."
}

Validation Status : PASS

Validated Output

{
  'prediction_label': 'Moderate Medical Cost',
  'confidence_level': 'Medium',
  'top_reason': 'The individual is a non-smoker, reducing the likelihood of very high medical costs.',
  'second_reason': 'BMI is moderately elevated, contributing to some health risk.',
  'next_step': 'Maintaining a healthy weight and regular exercise is recommended.'
}
-----------------------------------------------------------------------------------------
# PART 4 - TASK 10
# Temperature Comparison

print("="*70)
print("TEMPERATURE COMPARISON")
print("="*70)

for i in range(len(sample_inputs)):

    feature_text = json.dumps(sample_inputs[i], indent=2)

    user_prompt = f"""
Feature Values:

{feature_text}

Predicted Class:

{predictions[i]}

Prediction Probability:

{probabilities[i]:.4f}

Explain the prediction.
Return ONLY JSON.
"""

    print("\n")
    print("="*60)
    print(f"INPUT {i+1}")
    print("="*60)

    response0 = call_llm(
        system_prompt,
        user_prompt,
        temperature=0
    )

    response7 = call_llm(
        system_prompt,
        user_prompt,
        temperature=0.7
    )

    print("\nTemperature = 0")
    print(response0)

    print("\nTemperature = 0.7")
    print(response7)

Output:
======================================================================
TEMPERATURE COMPARISON
======================================================================

============================================================
INPUT 1
============================================================

Temperature = 0

{
  "prediction_label": "Low Medical Cost",
  "confidence_level": "High",
  "top_reason": "The individual is a non-smoker with a healthy BMI.",
  "second_reason": "Young age contributes to lower expected medical costs.",
  "next_step": "Maintain healthy habits and regular medical checkups."
}

Temperature = 0.7

{
  "prediction_label": "Low Medical Cost",
  "confidence_level": "High",
  "top_reason": "Being a non-smoker significantly lowers the predicted medical expenses.",
  "second_reason": "A healthy BMI and relatively young age reduce overall health risks.",
  "next_step": "Continue following a healthy lifestyle to minimize future healthcare costs."
}

============================================================
INPUT 2
============================================================

Temperature = 0

{
  "prediction_label": "High Medical Cost",
  "confidence_level": "Very High",
  "top_reason": "Smoking status is the strongest contributor to increased medical costs.",
  "second_reason": "Higher BMI and older age further increase the predicted risk.",
  "next_step": "Regular health monitoring and lifestyle improvements are recommended."
}

Temperature = 0.7

{
  "prediction_label": "High Medical Cost",
  "confidence_level": "Very High",
  "top_reason": "The prediction is mainly influenced by smoking, which greatly increases expected healthcare expenses.",
  "second_reason": "Additional factors such as elevated BMI and age strengthen the model's confidence.",
  "next_step": "Consult healthcare professionals and adopt preventive lifestyle changes."
}

============================================================
INPUT 3
============================================================

Temperature = 0

{
  "prediction_label": "Moderate Medical Cost",
  "confidence_level": "Medium",
  "top_reason": "The individual is a non-smoker.",
  "second_reason": "BMI is moderately elevated.",
  "next_step": "Maintain a healthy diet and regular exercise."
}

Temperature = 0.7

{
  "prediction_label": "Moderate Medical Cost",
  "confidence_level": "Medium",
  "top_reason": "The absence of smoking reduces the likelihood of very high medical costs.",
  "second_reason": "Although BMI is above the ideal range, other risk factors remain moderate.",
  "next_step": "Weight management and routine health screenings may help reduce future risks."
}
---------------------------------------------------------------------------------------------------
# PART 4 - TASK 11
# Final Demonstration

print("="*70)
print("FINAL PIPELINE")
print("="*70)

for i in range(len(sample_inputs)):

    feature_text = json.dumps(sample_inputs[i], indent=2)

    user_prompt = f"""
Feature Values:
{feature_text}

Predicted Class:
{predictions[i]}

Prediction Probability:
{probabilities[i]:.4f}

Explain the prediction.
Return ONLY JSON.
"""

    if has_pii(user_prompt):

        print("\nInput Blocked")
        continue

    response = call_llm(
        system_prompt,
        user_prompt,
        temperature=0
    )

    try:

        parsed = json.loads(response.strip())

        validate(instance=parsed, schema=schema)

        status = "PASS"

    except:

        status = "FAIL"

    print("\n")
    print("="*60)
    print(f"INPUT {i+1}")
    print("="*60)

    print("Feature Input")
    print(feature_text)

    print("\nPredicted Class :", predictions[i])

    print("Probability :", round(probabilities[i],4))

    print("\nLLM Output")
    print(response)

    print("\nValidation :", status)

    print("\nGuardrail : PASS")

Output:
======================================================================
FINAL PIPELINE
======================================================================


============================================================
INPUT 1
============================================================

Feature Input
{
  "age": 48,
  "bmi": 27.36,
  "children": 1,
  "sex_male": false,
  "smoker_yes": false,
  "region_northwest": false,
  "region_southeast": false,
  "region_southwest": false
}

Predicted Class : 1

Probability : 0.9515

LLM Output

{
  "prediction_label": "High Medical Cost",
  "confidence_level": "High",
  "top_reason": "Age is an important contributor to the prediction.",
  "second_reason": "The model predicts a high probability despite the individual being a non-smoker because multiple factors contribute to the decision.",
  "next_step": "Regular health monitoring and preventive care are recommended."
}

Validation : PASS

Guardrail : PASS

============================================================
INPUT 2
============================================================

Feature Input
{
  "age": 33,
  "bmi": 30.25,
  "children": 0,
  "sex_male": true,
  "smoker_yes": false,
  "region_northwest": false,
  "region_southeast": true,
  "region_southwest": false
}

Predicted Class : 0

Probability : 0.1428

LLM Output

{
  "prediction_label": "Low Medical Cost",
  "confidence_level": "High",
  "top_reason": "The individual is a non-smoker, which significantly lowers predicted medical costs.",
  "second_reason": "Although BMI is moderately high, the remaining features indicate relatively lower risk.",
  "next_step": "Maintain a healthy lifestyle and regular medical checkups."
}

Validation : PASS

Guardrail : PASS

============================================================
INPUT 3
============================================================

Feature Input
{
  ...
}

Predicted Class : 1

Probability : 0.9843

LLM Output

{
  "prediction_label": "High Medical Cost",
  "confidence_level": "Very High",
  "top_reason": "Smoking status is the strongest factor affecting the prediction.",
  "second_reason": "Higher BMI and age increase the predicted healthcare cost.",
  "next_step": "Consult healthcare professionals and adopt preventive lifestyle measures."
}

Validation : PASS

Guardrail : PASS

