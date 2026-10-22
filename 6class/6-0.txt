# [6.1 matcher]

def getFeatures(im,xy,W=7):
  N=len(xy)
  X=np.empty((N,15*15*3),'f')
  for i in range(N):
   x,y=xy[i]
   X[i]=im[y-W:y+W+1,x-W:x+W+1].reshape(-1)
  return X


def calcSSDtable(tex1, x1, tex2, x2):
  N1=len(x1)
  N2=len(x2)

  X=getFeatures(tex1,x1)
  Y=getFeatures(tex2,x2)
  eb.                    stopWatch('getFeatures')

## NAIVE
  mt=np.empty((N1,N2),'f')
  for i in range(N1):
    for j in range(N2):
      t=X[i]-Y[j]
      mt[i,j]=t@t

## FAST
# mt=np.sum(X*X,axis=1)[:,None]+\
#    np.sum(Y*Y,axis=1)[None,:]-\
#    2*(X@Y.T)

  eb.                    stopWatch('SSD-table')
  return mt


def matchMethod1(mt,x1,x2):
  N1=len(x1)
  N2=len(x2)
  N=min(N1,N2)
  w=np.empty((N,4),'i')
# mtorg=mt.copy() # 下記ループで mt が破壊されるため，必要であればコピーを保存しておく

  for i in range(N):
    ji=np.argmin(mt[i]) # 行中の最小値の場所
    sm=mt[i,ji]         # 最小値

    # 出力は対(i,ji)ではなく，
    # それらに対応する座標の組(x,y,u,v)とする
    w[i]=[ x1[i ,0],x1[i ,1],
           x2[ji,0],x2[ji,1] ]
    mt[:,ji]=np.inf     # 縦消去

  return w


def getW(im1,x1, im2,x2):
  mt=calcSSDtable(im1,x1,
                  im2,x2)
  print(mt)

  w=matchMethod1(mt,x1,x2) # 特徴点の対応付け
  eb.                    stopWatch('match')

  return w


def main():
  im1=eb.getImg('0.jpg') # 入力画像1 (変数名と画像名の番号の違いに注意)
  im2=eb.getImg('2.jpg') # 入力画像2

  x1=np.array([
    [234,529],[294,560],[336,512],[685,481],[355,506],
    [296,496],[615,439],[384,464],[113,398],[317,494]])
  x2=np.array([
    [483,536],[542,567],[585,525],[545,507],[606,520],
    [365,407],[635,478],[444,525],[349,226],[656,475]])
  # これらはデバッグ用としても点数が少なすぎる．
  # デバッグ時は第 5 回の結果を埋め込む（30-100点）．
  # 最終的には第 4-5 回の手法で配列を作る．

  eb.                    stopWatch('start')
  w=getW(im1,x1, im2,x2)

  print("Match Results:")
# print(w)
  for wi in w:
   print(f"{wi[0]},{wi[1]},{wi[2]},{wi[3]},")

  eb.showTime()
