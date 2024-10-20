from model import ML
import streamlit as st
import pandas as pd

Ml = ML("model.pkl","label_encoders.pkl")

def collect_user_input():
    # Collect input from the user based on unique values provided
    job = st.selectbox("Job", ['housemaid', 'services', 'admin.', 'blue-collar', 
                               'technician', 'retired', 'management', 'unemployed', 
                               'self-employed', 'unknown', 'entrepreneur', 'student'])
    
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
    
    education = st.selectbox("Education", ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 
                                           'professional.course', 'unknown', 'university.degree', 'illiterate'])
    
    default = st.selectbox("Has Credit in Default?", ['no', 'unknown', 'yes'])
    
    housing = st.selectbox("Has Housing Loan?", ['no', 'yes', 'unknown'])
    
    loan = st.selectbox("Has Personal Loan?", ['no', 'yes', 'unknown'])
    
    contact = st.selectbox("Contact Communication Type", ['telephone', 'cellular'])
    
    month = st.selectbox("Last Contact Month", ['may', 'jun', 'jul', 'aug', 'oct', 
                                                'nov', 'dec', 'mar', 'apr', 'sep'])
    
    day_of_week = st.selectbox("Day of the Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
    
    poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'])
    
    age = st.slider("Age", min_value=17, max_value=98, value=32)
    
    campaign = st.slider("Number of Contacts Per Campaign", min_value=1, max_value=56, value=1)
    
    pdays = st.slider("Days Passed Since Last Contact", min_value=0, max_value=999, value=999)
    
    previous = st.slider("Number of Previous Contacts", min_value=0, max_value=7, value=0)
    
    emp_var_rate = st.slider("Employment Variation Rate", min_value=-3.4, max_value=1.4, step=0.1, value=1.1)
    
    cons_price_idx = st.slider("Consumer Price Index", min_value=92.2, max_value=94.8, step=0.001, value=93.994)
    
    cons_conf_idx = st.slider("Consumer Confidence Index", min_value=-50.8, max_value=-26.9, step=0.1, value=-36.4)
    
    euribor3m = st.slider("Euribor 3-Month Rate", min_value=0.63, max_value=5.045, step=0.001, value=4.857)
    
    nr_employed = st.slider("Number of Employees", min_value=4963.6, max_value=5228.1, step=0.1, value=5191.0)

    # Create a DataFrame
    data = {
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'poutcome': [poutcome],
        'age': [age],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'emp.var.rate': [emp_var_rate],
        'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        'nr.employed': [nr_employed]
    }
    
    df = pd.DataFrame(data)
    
    return df


def main():

    st.title("Customer Data Input Form")

    df = collect_user_input()

    if st.button("Submit"):
        processed_df = Ml.transform(df)
        result = Ml.predict(processed_df)
        st.write("Outcome:")
        st.write("The customer will purchase the subscription" if result == 1 else "The campaign is a failure")

if __name__ == "__main__":
    main()