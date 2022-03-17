# flask_deploy_testing
Testing for scikit-leaern model deployment.

## How to test  
- Execute model.py (train simple sequential model and save model)
- Execute main.py (run flask server)  

**GET Request**  
localhost:8000/show_static  
localhost:8000/show_dynamic?index=(integer)  
localhost:8000/predict_get?index=(integer)  
localhost:8000/predict_real_get?index=(integer)  

**POST Request**  
localhost:8000/predict  
params : {'index': (integer})  

## TODO
1. Test for POST request.
2. Test for gunicorn package.
3. Test for nginx package.
