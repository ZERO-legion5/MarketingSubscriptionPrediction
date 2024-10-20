import pickle

class ML():

    def __init__(self, model_path, label_encoder_path):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.load()

    def load(self):
        # Load the model
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load the separate label encoders
        with open(self.label_encoder_path, "rb") as f:
            self.label_encoders = pickle.load(f)

    def transform(self, df):
        # Apply the correct label encoder for each categorical attribute
        df['job'] = self.label_encoders['job'].transform(df['job'])
        df['marital'] = self.label_encoders['marital'].transform(df['marital'])
        df['education'] = self.label_encoders['education'].transform(df['education'])
        df['default'] = self.label_encoders['default'].transform(df['default'])
        df['housing'] = self.label_encoders['housing'].transform(df['housing'])
        df['loan'] = self.label_encoders['loan'].transform(df['loan'])
        df['contact'] = self.label_encoders['contact'].transform(df['contact'])
        df['month'] = self.label_encoders['month'].transform(df['month'])
        df['day_of_week'] = self.label_encoders['day_of_week'].transform(df['day_of_week'])

        # Replace 'poutcome' manually (no LabelEncoder needed)
        df['poutcome'].replace(['nonexistent', 'failure', 'success'], [1, 2, 3], inplace=True)

        # Apply transformation to 'age' by binning the values into groups
        df.loc[df['age'] <= 32, 'age'] = 1
        df.loc[(df['age'] > 32) & (df['age'] <= 47), 'age'] = 2
        df.loc[(df['age'] > 47) & (df['age'] <= 70), 'age'] = 3
        df.loc[(df['age'] > 70) & (df['age'] <= 98), 'age'] = 4

        # Return the transformed DataFrame with the selected columns
        return df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                   'contact', 'month', 'day_of_week', 'emp.var.rate', 'cons.price.idx',
                   'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 
                   'previous', 'poutcome']]

    def predict(self, df):
        # Predict the output using the model and return the result for the first sample
        val = self.model.predict(df.iloc[[0]])
        return val
