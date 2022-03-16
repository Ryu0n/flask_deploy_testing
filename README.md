# flask_deploy_testing
Testing for model deployment.

## How to test
1. Execute model.py. (train simple sequential model and save model)
2. Execute main.py. (run flask server)  

**GET Request**  
localhost:8000/predict_get?index=(integer)  

**POST Request**  
localhost:8000/predict  
params : {'index': (integer})  

## TODO
1. Attach mnist image on template.
2. Test for POST request.
3. Test for gunicorn package.
4. Test for nginx package.
