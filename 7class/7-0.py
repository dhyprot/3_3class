# [7.2 ransac]


# 変換行列の信頼度 = 「検出された点と変換された点の距離が十分に小さい点」の数を計算
def calcScore(H,w):
  # NAIVE
  score=0
  for i in range(len(w)):   # ** ここから 3 行は「for x,y,u,v in w:」でも良い
    x=w[i,0]; y=w[i,1]
    u=w[i,2]; v=w[i,3]
    du=0
    dv=0 # (du,dv) は， (x,y) を H で変換した座標 と (u,v) の差．
    if du*du+dv*dv<10: pass # w[i] が正しい場合の処理 (※1 閾値)
    else:              pass # w[i] が正しくない場合の処理
  return score

  # FAST ※2

def getH(w):

 TRIAL=200                  # ※3 試行回数
 MAX=len(w)

 # MAX 未満の非負整数が一回ずつ出現する配列を作る
 rndAry=np.arange(MAX)

 # RANSAC 試行を反復
 bestScore=0                # RANSAC の結果用
 for trial in range(TRIAL):
  chkTime(30)               # ※4 無限ループ対策（30秒で止まる）

  # 相異なる数値を4つ選び出す．
  #   rndAry の要素を乱数で並び変え，
  #   最初の 4 要素 rndAry[0], ..., rndAry[3] を使う．
  np.random.shuffle(rndAry)
  i0,i1,i2,i3=rndAry[:4]

  # 4組の対応候補で変換行列を計算する．
  H=calcHomography(w[[i0,i1,i2,i3]])

  # 変換行列 H を評価する．
  #   評価関数は別途実装のこと．
  score=calcScore(H,w)

  if 1:                  # TODO: 最大値を残すようにする
   bestScore=score       # 多数の試行で得られる score の最大値が bestScore
   bestH=H               # 現在までに見つかった最良の H が bestH
   print(i0,i1,i2,i3,'score=',score)

 eb.         stopWatch('getH')
 return bestH


def main():

  # ※5 TODO: 前回の課題で得られた特徴点対応候補リスト w を与える．
  w=np.array([
   234,529,483,536,
   294,560,656,475,
   336,512,585,525,
   685,481,635,478,
   355,506,606,520,
   296,496,545,507,
   615,439,349,226,
   384,464,365,407,
   113,398,444,525,
   317,494,542,567,
  ]).reshape(-1,4)  # 4要素毎に区切り，行列にする．
 
  eb.         stopWatch('start')
  H=getH(w)
  eb.         showTime()
  print('best matrix:\n',H)




#### UTIL

import time
_start_time=time.time()
def chkTime(_exec_time_limit):
 if time.time()-_start_time>_exec_time_limit:
  raise BaseException(f'!!!!!!!! Too long execution time (> {_exec_time_limit} sec) !!!!!!!!')
