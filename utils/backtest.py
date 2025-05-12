import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from utils import config


class Backtest:
    def __init__(self, weight_df):
        """
        Initialize the Backtest class with weight data
        """

        self.price_df = pd.read_csv('data/price.csv', index_col='Date', parse_dates=True).ffill()
        self.benchmark = pd.read_csv('data/benchmark.csv', index_col='Date', parse_dates=True).squeeze()
        self.return_df = self.price_df.pct_change(fill_method=None).fillna(0) + 1
        self.weight_df = weight_df
        self.capital = None

    def run(self, transaction_cost=0.001, outsample=False):
        """
        Run the backtest simulation
        """
        return_df = self.return_df
        if not outsample:
            return_df = return_df[return_df.index < pd.to_datetime(config.split_date)]
        else:
            return_df = return_df[return_df.index >= pd.to_datetime(config.split_date)]

        dates = []
        capital_list = []
        holdings = None
        for index in return_df.index:
            if index in self.weight_df.index:
                holdings = self.weight_df.loc[index] * (capital_list[-1] if capital_list else 1) * (1 - transaction_cost) * return_df.loc[index]
            elif holdings is not None:
                holdings = holdings * return_df.loc[index]
            else:
                continue
            capital_list.append(sum(holdings))
            dates.append(index)
            
        self.capital = pd.Series(capital_list, index=dates, name='Capital')

    def show(self):
        """
        Print performance metrics and plot the portfolio value over time.
        """

        if self.capital is None:
            raise ValueError("Run backtest first")

        # Calculate metrics
        aligned_index = self.capital.index.intersection(self.benchmark.index)
        capital_returns = self.capital.loc[aligned_index].pct_change().dropna()
        benchmark_returns = self.benchmark.loc[aligned_index].pct_change().dropna()

        metrics = {
            'Total Return': f"{(self.capital.iloc[-1] / self.capital.iloc[0] - 1) * 100:.2f}%",
            'Annualized Return': f"{((1 + (self.capital.iloc[-1] / self.capital.iloc[0] - 1)) ** (252 / len(self.capital)) - 1) * 100:.2f}%",
            'Portfolio Volatility': f"{capital_returns.std() * np.sqrt(252) * 100:.2f}%",
            'Benchmark Volatility': f"{benchmark_returns.std() * np.sqrt(252) * 100:.2f}%",
            'Sharpe Ratio': f"{(((1 + (self.capital.iloc[-1] / self.capital.iloc[0] - 1)) ** (252 / len(self.capital)) - 1) - 0.01) / (capital_returns.std() * np.sqrt(252)):.2f}",
            'Beta': f"{np.cov(capital_returns, benchmark_returns)[0][1] / np.var(benchmark_returns):.2f}",
            'Max Drawdown': f"{((self.capital - self.capital.cummax()) / self.capital.cummax()).min() * 100:.2f}%"
        }

        # Plot the portfolio value
        benchmark = self.benchmark.loc[self.capital.index]
        benchmark = benchmark / benchmark.iloc[0]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(self.capital.index, self.capital, color='#cc1452')
        ax.plot(benchmark.index, benchmark, color='#006f9b')
        
        fig.patch.set_facecolor("#fffcfa")
        ax.set_facecolor("#fffcfa")
        ax.grid(axis='y', color='#999189', linestyle='-', linewidth=.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title('Portfolio Value Over Time', color='#555555', pad=15)
        ax.set_xlabel('Date', color='#555555')
        ax.set_ylabel('Value', color='#555555')
        ax.tick_params(axis='x', colors='#555555', labelrotation=0, direction='inout')
        ax.tick_params(axis='y', colors='#555555', length=0)
        ax.text(self.capital.index[-1] + pd.Timedelta(days=15), self.capital.iloc[-1], 'Strategy', color='#555555', va='center', ha='left', fontsize=9)
        ax.text(benchmark.index[-1] + pd.Timedelta(days=15), benchmark.iloc[-1], self.benchmark.name, color='#555555', va='center', ha='left', fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Plot metrics table
        table = plt.table(
            cellText=list(metrics.items()),
            cellLoc='left',
            loc='right',
            bbox=[1.15, 0.2, 0.45, 0.6]
        )

        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(.5)        
            cell.set_edgecolor('#555555') 
            cell.set_text_props(color='#555555')
            cell.PAD = 0.05
            if col == 0:
                cell.set_width(0.8) 
            elif col == 1:
                cell.set_width(0.4)  

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        plt.subplots_adjust(right=0.65)
        
        plt.show()
