# --coding:utf-8
import numpy as np
import pandas as pd
import scipy.signal as signal

FREQ = ('5m', '30m', 'd', 'w')


class TsBlock:

    def __init__(self, name):
        self.instrument_name = name
        self.__extremes = dict.fromkeys(FREQ)
        self.__segments = dict.fromkeys(FREQ)
        self.__blocks = dict.fromkeys(FREQ)
        self.__current = pd.DataFrame(index=FREQ, columns=['trend', 'No.', 'block_status', 'segment_num'])

    def get_extremes(self, start=None, end=None, freq='d'):
        extreme = self.__extremes[freq]

        if extreme:
            if start and end:
                pass
            elif start:
                pass
            elif end:
                pass
            else:
                return extreme
        else:
            df = self.get_history_hq(start=start, end=end, freq=freq)
            extreme = self.__extreme(df.high, df.low)
            self.__extremes[freq] = extreme

        return extreme

    def get_segments(self, start=None, end=None, freq='d'):
        return self.__segments[freq]

    def get_blocks(self, start=None, end=None, freq='d'):
        return self.__blocks[freq]

    def get_current(self, start=None, end=None, freq='d'):
        return self.__current.loc[freq]

    def get_history_hq(self, start=None, end=None, freq='d'):
        """
        get history bar from external api
        :param start: datetime
        :param end: datetime
        :param freq: '5m', '30m', 'd', 'w'
        :return: pd.Dataframe columns=['high', 'low']
        """
        df = pd.DataFrame(columns=['high', 'low'])
        return df

    @staticmethod
    def __extreme(high, low):
        """
        calculate the extreme values of high and low
        :param high: pd.Series, index is datetime
        :param low:  pd.Series, index is datetime
        :return: [pd.Series, pd.Series]
        """
        higher = high.iloc[signal.argrelextrema(high.values, np.greater)]
        lower = low.iloc[signal.argrelextrema(-low.values, np.greater)]

        return [higher, lower]

    def segment(self, higher, lower):
        df = pd.concat([higher, lower], axis=1, join='outer')
        # 比较前后高低点
        df1 = df.diff()
        df2 = df.diff(-1)

        # 需要删除的高低点，连续高低点中的较低高点和较高低点
        index = [df1['high'] < 0, df2['high'] < 0, df1['low'] > 0, df2['low'] > 0]
        flag = [x.any() for x in index]

        if not (flag[0] or flag[1] or flag[2] or flag[3]):
            return [higher, lower]

        # 处理连续的高低点中，高点比低点低的情况

        # 删除无效的高低点
        if flag[0]:
            df.loc[index[0], 'high'] = np.nan  # 向后删除较低高点
        if flag[1]:
            df.loc[index[1], 'high'] = np.nan  # 向前删除较低高点
        if flag[2]:
            df.loc[index[2], 'low'] = np.nan
        if flag[3]:
            df.loc[index[3], 'low'] = np.nan
        if flag[0] or flag[1]:
            higher = df['high'].dropna()
        if flag[2] or flag[3]:
            lower = df['low'].dropna()

        # 合并高低点后再处理一次
        return self.segment(higher, lower)

    def block_identify(self, higher, lower):
        # 前向寻找Block
        gd_df = pd.concat([higher, lower], axis=1, join='outer')
        df = gd_df.sort_index(ascending=False).fillna(0)

        # init current block
        block_high = higher[-1]
        block_low = lower[-1]
        start_dt = df.index[1]
        end_dt = df.index[0]
        segment_num = 1
        current_dt = start_dt
        # 初始化block表
        block_df = pd.DataFrame(
            columns=['start_dt', 'end_dt', 'block_high', 'block_low', 'block_highest', 'block_lowest', 'segment_num'])

        for row in df[2:].itertuples():
            # print(row.Index)
            # print([current_dt, start_dt, end_dt, block_high,block_low,block_highest, block_lowest,segment_num])
            if segment_num < 2:  # 一上一下2根线段必定有交集,不需要判断是否是新的block
                current_dt = row.Index
                segment_num = segment_num + 1
                if row.high > row.low:  # 顶
                    block_high = min(block_high, row.high)
                else:
                    block_low = max(block_low, row.low)
            else:
                if row.high > row.low:  # 顶
                    if row.high < block_low:  # 第三类卖点，新的中枢开始
                        start_index = gd_df.index.get_loc(current_dt) + 1
                        end_index = gd_df.index.get_loc(end_dt)
                        block_highest = gd_df.high[start_index: end_index].max()
                        block_lowest = gd_df.low[start_index: end_index].min()

                        insert_row = pd.DataFrame(
                            [[current_dt, end_dt, block_high, block_low, block_highest, block_lowest, segment_num]],
                            columns=['start_dt', 'end_dt', 'block_high', 'block_low', 'block_highest', 'block_lowest',
                                     'segment_num'])
                        block_df = block_df.append(insert_row, ignore_index=True)

                        end_dt = start_dt
                        segment_num = 2
                        block_high = row.high
                        block_low = lower[current_dt]
                    else:
                        segment_num = segment_num + 1
                        block_high = min(block_high, row.high)
                else:
                    if row.low > block_high:  # 第三类买点，新的中枢开始
                        start_index = gd_df.index.get_loc(current_dt) + 1
                        end_index = gd_df.index.get_loc(end_dt)
                        block_highest = gd_df.high[start_index: end_index].max()
                        block_lowest = gd_df.low[start_index: end_index].min()

                        insert_row = pd.DataFrame(
                            [[current_dt, end_dt, block_high, block_low, block_highest, block_lowest, segment_num]],
                            columns=['start_dt', 'end_dt', 'block_high', 'block_low', 'block_highest', 'block_lowest',
                                     'segment_num'])
                        block_df = block_df.append(insert_row, ignore_index=True)

                        end_dt = start_dt
                        segment_num = 2
                        block_low = row.low
                        block_high = higher[current_dt]
                    else:
                        segment_num = segment_num + 1
                        block_low = max(block_low, row.low)
                start_dt = current_dt
                current_dt = row.Index
        return block_df.set_index('start_dt')

    def block_relation(self, block_df):
        block_relation_df = block_df[block_df['segment_num'] > 3].diff(-1)[:-1]
        df = block_df.copy(deep=True)
        df['block_flag'] = '-'
        df['block_hl_flag'] = '-'
        df['top_bottom_flag'] = '-'

        for row in block_relation_df.itertuples():
            current_dt = row.Index
            # prev_index = block_relation_df.index.get_loc(current_dt) - 1

            if row.block_high > 0 and row.block_low > 0:
                block_flag = 'up'
            elif row.block_high < 0 and row.block_low < 0:
                block_flag = 'down'
            elif row.block_high > 0 and row.block_low < 0:
                block_flag = 'include'
            elif row.block_high < 0 and row.block_low > 0:
                block_flag = 'included'

            if row.block_highest > 0 and row.block_lowest > 0:
                block_hl_flag = 'up'
            elif row.block_highest < 0 and row.block_lowest < 0:
                block_hl_flag = 'down'
            elif row.block_highest > 0 and row.block_lowest < 0:
                block_hl_flag = 'include'
            elif row.block_highest < 0 and row.block_lowest > 0:
                block_hl_flag = 'included'

            df.block_flag[current_dt] = block_flag
            df.block_hl_flag[current_dt] = block_hl_flag

            if df.segment_num[current_dt] % 2 == 0:
                if block_flag == 'up':
                    df.top_bottom_flag[current_dt] = 'top'
                elif block_flag == 'down':
                    df.top_bottom_flag[current_dt] = 'bottom'

        return df
