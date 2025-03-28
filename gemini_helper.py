# import google.genai as genai
# from pathlib import Path
# import os
# 
# project_id =''
# location = ''
# # Configure Gemini
# client = genai.Client(
#         vertexai=True, project=project_id, location=location
#     )
# # Load the model
# model_id = genai.GenerativeModel('gemini-2.0-flash')
# 
# def load_context():
#     """Load context from documentation"""
#     context_path = Path("docs/model_documentation.pdf")
#     return """
#     Model: BigQuery ARIMA_PLUS_XREG
#     Features used: 
#     - Indicator_1: Historical revenue data
#     - Indicator_2: Market demand index
#     - Indicator_3: Seasonal factors
#     - Indicator_4: Economic indicators
#     - Indicator_5: Previous period performance
#     - Indicator_6: Rolling average metrics
#     
#     Attribution scores represent the relative importance of each feature
#     in making the forecast prediction. Higher scores indicate stronger
#     influence on the forecast.
#     
#     The model was trained on 5 years of historical data and uses
#     both time series components and external regressors to make
#     predictions.
#     """
# 
# # Initialize context
# model_context = load_context()
# 
# def get_gemini_response(question, context):
#     """Get response from Gemini"""
#     prompt = f"""
#     You are an AI assistant helping users understand a forecasting model and its outputs.
#     Use the following context to answer the question:
#     
#     {context}
#     
#     Question: {question}
#     
#     Provide a clear, concise answer focusing on the specific aspects asked about.
#     If you're not sure about something, say so rather than making assumptions.
#     """
#     
#     try:
#         response = client.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"I apologize, but I encountered an error: {str(e)}" 