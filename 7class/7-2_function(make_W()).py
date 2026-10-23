# [5.1 features]

def TKfilter(src):
  W=7
  hgt,wdt,_=src.shape

# (1) 勾配 Ix, Iy
  Ix=np.zeros((hgt,wdt),'f') # 作業領域
  Iy=np.zeros((hgt,wdt),'f')

  """
  for y in range(1,hgt-1):
   for x in range(1,wdt-1):
    Ix[y,x]=0
    Ix[y,x]=src[y,x+1  ,1] - src[y,x-1  ,1]
  # TODO: Iy を参考に x 方向導関数を実装（現状では Iyy しか正しく計算されない）
    Iy[y,x]=src[y+1,x  ,1] - src[y-1,x  ,1]
  """

  # 別の方法の偏微分
  Ix=conv2d(src[...,1], np.array([[1,0,-1]],'f'))
  Iy=conv2d(src[...,1], np.array([[1],[0],[-1]],'f'))

  print('※1 y方向偏微分 Iy (赤は正，緑は負)')
  # eb.imshow(Iy[...,None]*[3,-2,0])

  print('※1 x方向偏微分 Ix (赤は正，緑は負)')
  # eb.imshow(Ix[...,None]*[3,-2,0])
  eb.stopWatch('(1)')

  # ｘ，ｙで別々の状態

# (2) IxIx, IxIy, IyIy
  IxIx=np.zeros((hgt,wdt),'f')     # 作業領域
  IxIy=np.zeros((hgt,wdt),'f')
  IyIy=np.zeros((hgt,wdt),'f')
  """
  for y in range(1,hgt-1):
   for x in range(1,wdt-1):
    IxIx[y,x]=Ix[y,x]*Ix[y,x]
    IxIy[y,x]=Ix[y,x]*Iy[y,x]
    IyIy[y,x]=Iy[y,x]*Iy[y,x]
  """
  # for文を使わない表現
  IxIx=Ix*Ix
  IxIy=Ix*Iy
  IyIy=Iy*Iy

  print('※2 Iy*Iy')
  # eb.imshow(IyIy*3)
  eb.stopWatch('(2)')

# (3) 共分散行列
# blurでぼかせる（平滑化）
  Ixx=blur(IxIx)
  Ixy=blur(IxIy)
  Iyy=blur(IyIy)

  print('※3 Iy*Iy の平滑化結果')
  # eb.imshow(Iyy*8)
  eb.stopWatch('(3)')

# (4) 固有値
  dstf=np.zeros((hgt,wdt),'f') # 出力配列（濃淡画像）
  """
  for y in range(W+1,hgt-W-1):
   for x in range(W+1,wdt-W-1):
    ixx=Ixx[y,x]; ixy=Ixy[y,x]; iyy=Iyy[y,x]
    dstf[y,x]=(ixx+iyy-np.sqrt((ixx-iyy)*(ixx-iyy)-4*ixy*ixy))/2
 # TODO: 固有値の小さい方を計算
  """
  # for文を使わない表現（多分早くなる）
  dstf=(Ixx+Iyy-np.sqrt((Ixx-Iyy)*(Ixx-Iyy)+4*Ixy*Ixy))/2

  eb.stopWatch('(4)')

  return dstf


import time
_start_time=time.time()
def chkTime(_exec_time_limit):
 if time.time()-_start_time>_exec_time_limit:
  raise BaseException(f'!!!!!!!! Too long execution time (> {_exec_time_limit} sec) !!!!!!!!')

import scipy

def conv2d(im,k):
 chkTime(30) # 無限ループ対策
 return scipy.signal.convolve2d(im, k, mode='same')

def blur(im,W=7): # a.k.a. Box blur
 out=np.zeros_like(im)
 t=np.zeros_like(im)
 
 for v in range(-W,W+1):
   t[W:-W,:]+=im[v+W:v+im.shape[0]-W,:]
 t=t/(2*W+1)
 for u in range(-W,W+1):
  out[:,W:-W]+=t[:,u+W:u+t.shape[1]-W]
 return out/(2*W+1)



def LocalMax(src,W=7):
# (5) 近傍での最大値検出
  out=np.zeros_like(src)
  for v in range(-W,W+1):
   for u in range(-W,W+1):
    out[W:-W,W:-W]=np.maximum(out[W:-W,W:-W],
                              src[v+W:src.shape[0]+v-W,
                                  u+W:src.shape[1]+u-W])
  return out

def detectFeaturePoints(src,MAX):
  W=7

  tkf=TKfilter(src)
  eb.                stopWatch('TKfilter')
  out=LocalMax(tkf)
  out[    :W+1]=0    # 周辺 8 pixel は除去
  out[-W-1:   ]=0    # ※1
  out[:,    :W+1]=0
  out[:,-W-1:   ]=0
  eb.                stopWatch('Max')

# (6) 極大判定
## FAST:
  mask=(out==tkf)&(out>0) # (最大値 == 中心) かつ (最大値 > 0)
  y,x=np.where(mask)      # mask[y,x]!=0 であるような x,y のリスト
  ary=np.vstack((x,y,out[y,x])).T # [x,y,out[y,x]] を行とする行列

## NAIVE:
# hgt,wdt=tkf.shape
# ary=[]
# for y in range(W+1,hgt-W-1):
#  for x in range(W+1,wdt-W-1):
#   if out[y,x]==tkf[y,x] and out[y,x]>0:
#    ary+=[[x,y,out[y,x]]]
# ary=np.array(ary)

  eb.                stopWatch('ary')

  # (7) ary を第3列の降順に MAX 個残す

  ary = ary[np.argsort(ary[:, 2])[::-1]]
  ary=ary[:MAX].astype(int)

  eb.                stopWatch('sort')

  # (8) 画像で結果表示
  hgt,wdt=tkf.shape
  img=np.zeros((hgt,wdt,3),'f') # 出力画像
  img[:,:,0]=tkf*9 # 濃淡画像からカラー画像に変換
  img[:,:,1]=tkf*9 # Numpy tips: Broadcast を参照
  img[:,:,2]=tkf*9

  for x,y,v in ary:
   x=int(x)
   y=int(y)
   img[y-W:y+W+1,x-W:x+W+1]=(img[y-W:y+W+1,x-W:x+W+1]+[1,0,0])/2

  # eb.imshow(img)

  return ary


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


def matchMethod2(mt,x1,x2):
  N1=len(x1)
  N2=len(x2)
  N=min(N1,N2)
  w=np.empty((N,4),'i')
# mtorg=mt.copy() # 下記ループで mt が破壊されるため，必要であればコピーを保存しておく

  for i in range(N):
    ji=np.argmin(mt) # 行中の最小値の場所
    sx,sy=np.unravel_index(ji,mt.shape) 

    # 出力は対(i,ji)ではなく，
    # それらに対応する座標の組(x,y,u,v)とする
    w[i]=[ x1[sx ,0],x1[sx ,1],
           x2[sy,0],x2[sy,1] ]
    mt[:,sy]=np.inf     # 縦消去
    mt[sx,:]=np.inf     # 横消去

  return w


def getW(im1,x1, im2,x2):
  mt=calcSSDtable(im1,x1,
                  im2,x2)

  w=matchMethod2(mt,x1,x2) # 特徴点の対応付け
  eb.                    stopWatch('match')

  return w

def make_W(im1, im2, max):
#   im1=eb.getImg('0.jpg') # 入力画像1 (変数名と画像名の番号の違いに注意)
#   im2=eb.getImg('2.jpg') # 入力画像2
#   eb.                      stopWatch('start')
#   max=30
  x1=detectFeaturePoints(im1, max)[:,:2]
  x2=detectFeaturePoints(im2, max)[:,:2]
  w=getW(im1,x1, im2,x2)[:,:4]
  print(w)
  return w

# def main():
#   im1=eb.getImg('0.jpg') # 入力画像1 (変数名と画像名の番号の違いに注意)
#   im2=eb.getImg('2.jpg') # 入力画像2
#   eb.                      stopWatch('start')
#   max=30
#   x1=detectFeaturePoints(im1, max)[:,:2]
#   x2=detectFeaturePoints(im2, max)[:,:2]
#   w=getW(im1,x1, im2,x2)[:,:3]
#   print(w)
#   #for wi in w:
#   # print(f"{wi[0]},{wi[1]},{wi[2]},{wi[3]},")
#   eb.showTime()
