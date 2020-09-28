import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use('ggplot')


def plot_performance(file):
    df = pd.read_csv(file)
    df['epoch'] = range(0, len(df))
    sns.regplot(x='epoch', y='epoch_steps', data=df)

    plt.title('Total Steps by Epoch')
    plt.ylim([0, df['epoch_steps'].max() + 10])
    plt.show()


if __name__ == '__main__':
    plot_performance('log/per_092520.csv')
