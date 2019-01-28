# --coding:utf-8
import numpy as np
import pandas as pd
import scipy.signal as signal
from datetime import datetime, timedelta

FREQ = ('5m', '30m', 'd', 'w')


class TsBlock:

    def __init__(self, name):
        self.instrument_name = name
        self.__extremes = dict.fromkeys(FREQ)
        self.__segments = dict.fromkeys(FREQ)
        self.__blocks = dict.fromkeys(FREQ)
        self.__current = pd.DataFrame(index=FREQ, columns=['trend', 'No.', 'block_status', 'segment_num'])

    def get_extremes(self, start=None, end=None, freq='d'):
        """
        only append the data at the end,  ascending sort
        :param start: datetime
        :param end: datetime
        :param freq: ('5m', '30m', 'd', 'w')
        :return: [pd.Series, pd.Series]
        """
        extreme = self.__extremes[freq]

        if extreme:
            if start and end:
                return [extreme[0][start:end], extreme[1][start:end]]
            elif start:
                return [extreme[0][start:end], extreme[1][start:end]]
            elif end:
                return [extreme[0][start:end], extreme[1][start:end]]
            else:
                return extreme
        else:
            df = self.get_history_hq(start=start, end=end, freq=freq)
            extreme = self.__extreme(df.high, df.low)
            self.__extremes[freq] = extreme

            return extreme

    def get_segments(self, start=None, end=None, freq='d'):
        """
                only append the data at the end,  ascending sort
                :param start: datetime
                :param end: datetime
                :param freq: ('5m', '30m', 'd', 'w')
                :return: [pd.Series, pd.Series]
                """
        segment = self.__segments[freq]

        if segment:
            if start and end:
                return [segment[0][start:end], segment[1][start:end]]
            elif start:
                return [segment[0][start:end], segment[1][start:end]]
            elif end:
                return [segment[0][start:end], segment[1][start:end]]
            else:
                return segment
        else:
            higher, lower = self.get_extremes(start=start, end=end, freq=freq)
            segment = self.__segment(higher, lower)
            self.__segments[freq] = segment
            return segment

    def get_blocks(self, start=None, end=None, freq='d'):
        """
        only append the data at the end,  ascending sort
        :param start: datetime
        :param end: datetime
        :param freq: ('5m', '30m', 'd', 'w')
        :return: pd.Dataframe   columns=['end_dt', 'block_high', 'block_low', 'block_highest', 'block_lowest',
                                         'segment_num', 'block_flag', 'block_hl_flag', 'top_bottom_flag']
        """

        block = self.__blocks[freq]

        if block:
            if start and end:
                return [segment[0][start:end], segment[1][start:end]]
            elif start:
                return [segment[0][start:end], segment[1][start:end]]
            elif end:
                return [segment[0][start:end], segment[1][start:end]]
            else:
                return segment
        else:
            higher, lower = self.get_segments(start=start, end=end, freq=freq)
            block = self.__block_identify(higher, lower)
            self.__blocks[freq] = block
            return block


    def get_current(self, start=None, end=None, freq='d'):
        return self.__current.loc[freq]

    @staticmethod
    def get_history_hq(start=None, end=None, freq='d'):
        """
        get history bar from external api
        :param start: datetime
        :param end: datetime
        :param freq: '5m', '30m', 'd', 'w'
        :return: pd.Dataframe columns=['high', 'low']
        """
        from pytdx.reader import TdxExHqDailyBarReader
        reader = TdxExHqDailyBarReader()
        df = reader.get_df(r"H:\TDX\vipdoc\ds\lday\30#AGL9.day")

        return df.loc[:, ['high', 'low']]

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

    @staticmethod
    def __segment(higher, lower):
        """
        identify the segments from high/low extremes
        :param higher: pd.Series, index is datetime
        :param lower:  pd.Series, index is datetime
        :return: [pd.Series, pd.Series]
        """
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
        return TsBlock.__segment(higher, lower)

    @staticmethod
    def __block_identify(higher, lower):
        # 后向寻找Block
        gd_df = pd.concat([higher, lower], axis=1, join='outer')
        df = gd_df.fillna(0)

        # init current block
        block_high = higher[0]
        block_low = lower[0]
        start_dt = df.index[0]
        end_dt = df.index[1]
        segment_num = 1
        current_dt = start_dt
        # 初始化block表
        block_df = pd.DataFrame(
            columns=['start_dt', 'end_dt', 'block_high', 'block_low', 'block_highest', 'block_lowest',
                     'segment_num'])

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
                        start_index = gd_df.index.get_loc(start_dt) + 1
                        end_index = gd_df.index.get_loc(current_dt)
                        block_highest = gd_df.high[start_index: end_index].max()
                        block_lowest = gd_df.low[start_index: end_index].min()

                        insert_row = pd.DataFrame([[start_dt, current_dt, block_high, block_low, block_highest,
                                                    block_lowest, segment_num]],
                                                  columns=['start_dt', 'end_dt', 'block_high', 'block_low',
                                                           'block_highest', 'block_lowest', 'segment_num'])
                        block_df = block_df.append(insert_row, ignore_index=True)

                        start_dt = end_dt
                        segment_num = 2
                        block_high = row.high
                        block_low = lower[current_dt]
                    else:
                        segment_num = segment_num + 1
                        block_high = min(block_high, row.high)
                else:
                    if row.low > block_high:  # 第三类买点，新的中枢开始
                        start_index = gd_df.index.get_loc(start_dt) + 1
                        end_index = gd_df.index.get_loc(current_dt)
                        block_highest = gd_df.high[start_index: end_index].max()
                        block_lowest = gd_df.low[start_index: end_index].min()

                        insert_row = pd.DataFrame([[start_dt, current_dt, block_high, block_low, block_highest,
                                                    block_lowest, segment_num]],
                                                  columns=['start_dt', 'end_dt', 'block_high', 'block_low',
                                                           'block_highest', 'block_lowest', 'segment_num'])
                        block_df = block_df.append(insert_row, ignore_index=True)

                        start_dt = end_dt
                        segment_num = 2
                        block_low = row.low
                        block_high = higher[current_dt]
                    else:
                        segment_num = segment_num + 1
                        block_low = max(block_low, row.low)
                end_dt = current_dt
                current_dt = row.Index
        return block_df.set_index('start_dt')

    def __block_relation(self, block_df):
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


if __name__ == "__main__":
    block = TsBlock("AGL9")

    segment = block.get_segments()
    print("------init extreme-----------")
    print(segment[0].tail())
    print("-----------------")
    print(segment[1].tail())

    today = datetime.today()
    observation = 365
    start = today - timedelta(observation)
    end = today - timedelta(7)

    segment = block.get_segments(start=start, end=end)
    print("------get extreme from memory-----------")
    print(segment[0].tail())
    print("-----------------")
    print(segment[1].tail())

    segment = block.get_segments(end=end)
    print("------get extreme from memory-----------")
    print(segment[0].tail())
    print("-----------------")
    print(segment[1].tail())

    segment = block.get_segments(start=today - timedelta(60))
    print("------get extreme from memory-----------")
    print(segment[0].head())
    print("-----------------")
    print(segment[1].head())

    # extreme = block.get_extremes()
    # print("------init extreme-----------")
    # print(extreme[0].tail())
    # print("-----------------")
    # print(extreme[1].tail())
    #
    # today = datetime.today()
    # observation = 365
    # start = today - timedelta(observation)
    # end = today - timedelta(7)
    #
    # extreme = block.get_extremes(start=start, end=end)
    # print("------get extreme from memory-----------")
    # print(extreme[0].tail())
    # print("-----------------")
    # print(extreme[1].tail())
    #
    # extreme = block.get_extremes(end=end)
    # print("------get extreme from memory-----------")
    # print(extreme[0].tail())
    # print("-----------------")
    # print(extreme[1].tail())
    #
    # extreme = block.get_extremes(start=today - timedelta(60))
    # print("------get extreme from memory-----------")
    # print(extreme[0].head())
    # print("-----------------")
    # print(extreme[1].head())