import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RollMean:
    attrs = ['aveSpeed']
    up = pd.read_csv('data/up/ave_up(replaceMissingData).csv', encoding='utf-8')
    down = pd.read_csv('data/down/ave_down(replaceMissingData).csv', encoding='utf-8')
    left = pd.read_csv('data/left/ave_left(replaceMissingData).csv', encoding='utf-8')
    right = pd.read_csv('data/right/ave_right(replaceMissingData).csv', encoding='utf-8')

    up_data = pd.DataFrame(up[attrs].values)
    up_data = np.array(up_data.rolling(window=6).mean())
    down_data = pd.DataFrame(down[attrs].values)
    down_data = np.array(down_data.rolling(window=6).mean())
    left_data = pd.DataFrame(left[attrs].values)
    left_data = np.array(left_data.rolling(window=6).mean())
    right_data = pd.DataFrame(right[attrs].values)
    right_data = np.array(right_data.rolling(window=6).mean())

    # up_data = up[attrs].values[8346:, :]
    # df_up = pd.DataFrame(up_data, columns=['aveSpeed'])
    # df_up['roll_mean'] = df_up.rolling(window=6).mean()
    pd.DataFrame(up_data[8352:, :]).plot(subplots=True, figsize=(9, 5), grid=True)
    plt.show()
    print(up_data[8352:, :])
