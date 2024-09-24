import matplotlib.pyplot as plt

def plot_data(predicted_df):
    predicted_df.plot(x="Date", y="Predicted_Price", kind="line")
    plt.show()