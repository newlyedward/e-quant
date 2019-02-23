# -*- coding: utf-8 -*-
import os
import datetime as dt
import numpy as np
import pandas as pd

from log import LogHandler

from tdx.setting import tdx_dir, MARKET2TDX_CODE, MARKET_DIR, PERIOD_DIR, PERIOD_EXT
from .utils import int2date

log = LogHandler(__file__)


def _get_future_day_hq(file_handler):
    names = 'datetime', 'open', 'high', 'low', 'close', 'openInt', 'volume', 'comment'
    offsets = tuple(range(0, 31, 4))
    formats = 'i4', 'f4', 'f4', 'f4', 'f4', 'i4', 'i4', 'i4'

    dt_types = np.dtype({'names': names, 'offsets': offsets, 'formats': formats}, align=True)
    hq_day_df = pd.DataFrame(np.fromfile(file_handler, dt_types))
    hq_day_df.index = pd.to_datetime(hq_day_df['datetime'].astype('str'), errors='coerce')
    hq_day_df.pop('datetime')
    return hq_day_df


def _get_future_min_hq(self, file_handler):
    names = 'date', 'time', 'open', 'high', 'low', 'close', 'openInt', 'volume', 'comment'
    formats = 'u2', 'u2', 'f4', 'f4', 'f4', 'f4', 'i4', 'i4', 'i4'
    offsets = (0, 2) + tuple(range(4, 31, 4))

    dt_types = np.dtype({'names': names, 'offsets': offsets, 'formats': formats}, align=True)
    hq_min_df = pd.DataFrame(np.fromfile(file_handler, dt_types))

    hq_min_df.index = hq_min_df.date.transform(self._int2date) + pd.to_timedelta(hq_min_df.time,
                                                                                 unit='m')
    hq_min_df.pop('date')
    hq_min_df.pop('time')
    return hq_min_df


def get_future_day_hq(market, contractid, update=dt.datetime(1970, 1, 1)):
    """
    :param market: 交易市场
    :param contractid: IL8 主力合约 IL9 期货指数 I1801
    :param update: 最后更新日期
    :return: pd.DateFrame
    """

    tdx_hq_dir = os.path.join(tdx_dir, 'vipdoc', MARKET_DIR[market], PERIOD_DIR['d'])
    hq_filename = MARKET2TDX_CODE[market] + '#' + contractid.upper() + PERIOD_EXT['d']
    hq_path = os.path.join(tdx_hq_dir, hq_filename)

    if not os.path.exists(hq_path):
        return None

    f = open(hq_path, "rb")

    f.seek(0, 0)
    begin = np.fromfile(f, dtype=np.int32, count=1)
    begin = dt.datetime.strptime(begin.astype(str)[0], '%Y%m%d')

    f.seek(-32, 2)
    end = np.fromfile(f, dtype=np.int32, count=1)
    end = dt.datetime.strptime(end.astype(str)[0], '%Y%m%d')

    if update < begin:
        f.seek(0, 0)
        return _get_future_day_hq(f)
    elif update > end:
        return None

    delta = (end - update)
    factor = delta.days
    try:
        f.seek(-32 * factor, 2)
    except OSError:
        f.seek(0, 0)
        log.warning('%s trade recodes are few and factor = %d is too big.', contractid, factor)
    hq_day_df = _get_future_day_hq(f)
    return hq_day_df[hq_day_df.index > update]


def get_future_min_hq(market, contractid, update=dt.datetime(1970, 1, 1), period='5m'):
    """
    :param market: 交易市场
    :param contractid: IL8 主力合约 IL9 期货指数 I1801
    :param update: 最后更新时间
    :param period: 周期'1m'，'5m'
    :return: 返回
    """
    tdx_hq_dir = os.path.join(tdx_dir, 'vipdoc', MARKET_DIR[market], PERIOD_DIR[period])
    hq_filename = MARKET2TDX_CODE[market] + '#' + contractid.upper() + PERIOD_EXT[period]
    hq_path = os.path.join(tdx_hq_dir, hq_filename)

    if not os.path.exists(hq_path):
        return None

    f = open(hq_path, "rb")

    f.seek(0, 0)
    begin = np.fromfile(f, dtype=np.int16, count=1)
    begin = int2date(begin)

    f.seek(-32, 2)
    end = np.fromfile(f, dtype=np.int16, count=1)
    end = int2date(end)

    if update < begin:
        f.seek(0, 0)
        return _get_future_min_hq(f)
    elif update > end:
        return None

    k_num = 240
    if period == '5m':
        k_num = k_num / 5

    delta = (end - update)
    factor = delta.days * k_num

    while update < end:
        try:
            f.seek(-32 * factor, 2)
            end = np.fromfile(f, dtype=np.int16, count=1)
            f.seek(-32 * factor, 2)  # 数据读取后移位，文件指针要回到原来位置
            end = int2date(end)
            factor = factor * 2
        except OSError:
            f.seek(0, 0)
            log.warning('%s trade recodes are few and factor = %d is too big.', contractid, factor)
            break
        except TypeError:
            log.error('{} end date is null!'.format(contractid))
            # TODO 删除数据文件，文件指向开始
            return None

    hq_min_df = _get_future_min_hq(f)
    return hq_min_df[hq_min_df.index > update]