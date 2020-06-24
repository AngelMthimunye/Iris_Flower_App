import requests

url = "http://localhost:5000/predict_api"
r = requests.post(url, json={"sepal_length":5.2, "sepal_width":3.8, "petal_length":1.5, "petal_width":0.2})

print(r.json())


