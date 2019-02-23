# -*- coding: utf-8 -*-
import datetime as dt


def int2date(x):
    year = int(x / 2048) + 2004
    month = int(x % 2048 / 100)
    day = x % 2048 % 100
    return dt.datetime(year, month, day)
