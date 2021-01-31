'''
Date: 2021-01-15 19:38:25
LastEditors: Rustle Karl
LastEditTime: 2021-01-16 13:06:13
'''
from threading import Thread
from time import sleep

import schedule

from deephoto import analysis_face, recommend_photo_fit

# 首次分析
# analysis_face()
recommend_photo_fit()

#
#
# def run_threaded(job_func):
#     job = Thread(target=job_func)
#     job.start()
#
#
# # schedule.every(8).hours.do(run_threaded, analysis_face)  # 首次之后定时分析
#
# while True:
#     schedule.run_pending()
#     sleep(1)
