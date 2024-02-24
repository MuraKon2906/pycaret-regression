from pycaret.regression import * 


from pycaret.datasets import get_data
insurance_data=get_data('insurance')

pipline=setup(insurance_data,target='charges',session_id=123)

predictor=create_model('lr')

plot_model(predictor,plot='residuals')

save_model(predictor,model_name='D:\MuraKon\pycaret-deployment\predictor')