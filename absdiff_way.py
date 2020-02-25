# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 00:18:15 2019

@author: User-abc
"""
import cv2
import numpy as np
import time


# 使用迴圈分析所有影片
time.sleep(1)
print('\n\n==============================================\n')
print('開始分析影片....\n')

cap = cv2.VideoCapture('lowlightvideo')  # 開啟影片檔
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 取得畫面尺寸-寬
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 取得畫面尺寸-高
area = width * height  # 計算畫面面積
ret, frame = cap.read()  # 初始化平均畫面
blrlen = 2  # 模糊處理長寬
mrphlen = 2  # 去除雜訊處理長寬
avg = cv2.blur(frame, (blrlen, blrlen))  # 平均畫面模糊處理
avg_float = np.float32(avg)  # 平均畫面轉浮點數
frameCnt = 0  # 影片畫格計數器
while cap.isOpened():
    ret, frame = cap.read()  # 讀取一幅影格
    frameCnt = frameCnt + 1
    if not ret:  # 若讀取至影片結尾，則跳出
        break
    blur = cv2.blur(frame, (blrlen, blrlen))  # 模糊處理
    # cv2.imshow('blur', blur)  # 解除註解可看畫面
    diff = cv2.absdiff(avg, blur)  # 計算目前影格與平均影像的差異值
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # 將圖片轉為灰階
    # cv2.imshow('gray', gray)  # 解除註解可看畫面
    ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # 篩選出變動程度大於門檻值的區域
    # cv2.imshow('thresh', thresh)  # 解除註解可看畫面
    kernel = np.ones((mrphlen, mrphlen), np.uint8)  # 使用型態轉換函數去除雜訊
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # cv2.imshow('thresh', thresh)  # 解除註解可看畫面
    cnts, cnsImg = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 產生等高線
    movecatch = [0, False]
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)  # 計算等高線的外框範圍
        if cv2.contourArea(c) < 30:  # 忽略太小的區域
            continue
        elif x + w < 500 and y + h < 55:  # 忽略監視器時間區域，米家(470,50)，海爾(500,55)
            continue
        else:
            movecatch[0] = movecatch[0] + 1
            movecatch[1] = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 篩選出的區域畫出方形外框
    cv2.drawContours(frame, cnts, -1, (255, 0, 255), 1)  # 畫出等高線(除錯用)
    # if frameCnt % 5 == 1:  # 解除註解可看畫面
    #     cv2.imshow('frame', frame)  # 顯示偵測結果影像
    if frameCnt % 500 == 0:
        print('>> 目前分析到影格[', frameCnt, '] <<', sep='')
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 若顯示過程鍵盤按下"q"鍵則終止while迴圈
        break
    cv2.accumulateWeighted(blur, avg_float, 0.1)  # 更新平均影像
    avg = cv2.convertScaleAbs(avg_float)
    # cv2.imshow('avg', avg)  # 解除註解可看畫面
print('\n')
time.sleep(0.1)
cap.release()  # 釋放所有讀取中影片
cv2.destroyAllWindows()  # 關閉所有預覽畫面
print('\n\n==============================================\n')
print('影片分析完成！\n')
