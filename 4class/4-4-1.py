# [4.2 tkfilter]

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
  eb.imshow(Iy[...,None]*[3,-2,0])

  print('※1 x方向偏微分 Ix (赤は正，緑は負)')
  eb.imshow(Ix[...,None]*[3,-2,0])
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
  eb.imshow(IyIy*3)
  eb.stopWatch('(2)')

# (3) 共分散行列
# blurでぼかせる（平滑化）
  Ixx=blur(IxIx)
  Ixy=blur(IxIy)
  Iyy=blur(IyIy)

  print('※3 Iy*Iy の平滑化結果')
  eb.imshow(Iyy*8)
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
  dstf=(Ixx+Iyy-np.sqrt((Ixx-Iyy)*(Ixx-Iyy)-4*Ixy*Ixy))/2

  eb.stopWatch('(4)')

  return dstf


def main():
  src=eb.getImg('0.jpg') # 入力画像

  eb.stopWatch('start')
  dstf=TKfilter(src)
  eb.showTime()

  eb.imshow(dstf*9)


#### UTIL

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

 for v in range(-W,W+1):
  for u in range(-W,W+1):
   out[W:-W,W:-W]+=im[v+W:im.shape[0]+v-W,u+W:u+im.shape[1]-W]
 return out/((2*W+1)*(2*W+1))