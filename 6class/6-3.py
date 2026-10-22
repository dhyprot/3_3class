# [3.1 lsq]

def makeAb(w):
  A=np.zeros((8,8),'d')
  b=np.zeros(8,'d')
  i=np.array([0,1,2,3])
  
  A[i*2+0,0]=w[i, 0]
  A[i*2+0,1]=w[i, 1] 
  A[i*2+0,2]=1
  A[i*2+1,2]=0
  A[i*2+0,3]=0
  A[i*2+1,3]=w[i, 0] 
  A[i*2+0,4]=0
  A[i*2+1,4]=w[i, 1]
  A[i*2+0,5]=0
  A[i*2+1,5]=1
  A[i*2+0,6]=-1 * w[i,0] * w[i, 2]
  A[i*2+1,6]=-1 * w[i,0] * w[i, 3]
  A[i*2+0,7]=-1 * w[i,1] * w[i, 2]
  A[i*2+1,7]=-1 * w[i,1] * w[i, 3]

  b[i*2+0  ]=w[i,2]
  b[i*2+1  ]=w[i,3]
  
  return A,b


def calcHomography(w): # （注）homography = 射影変換

  A,b=makeAb(w) # w から A と b を計算する．

  print(A)  # ※2
  print(b)

  # LeaST-SQuares = 最小2乗法
  tmp=np.linalg.lstsq(A,b,rcond=None)[0]
  print(tmp)

  # ※3 tmp の前に 0 個，後ろに 1 個の 1 を追加する．
  tmp=np.pad(tmp,(0,1),constant_values=1) 
  print(tmp)

  return tmp.reshape(3,3) # 3x3 にする．

# ===========================================================



def main():
  dw,dh=1024,768          # 出力画像の大きさ

  im0=eb.getImg('0.jpg')  # 入力画像0
  im1=eb.getImg('2.jpg')  # 入力画像1

  m0d=np.array([[ 1,0,-100 ],
                [ 0,1,-100 ],
                [ 0,0,1 ]],'f')                 # 画像0 と 出力画像 の関係

  m10=np.array(['''
0.980063 0.155844 98.500361
-0.055756 1.153389 0.503900
-0.000139 0.000316 1.000000'''.split()]).astype('f').reshape(3,3)
                                                # 画像1 と 画像0 の関係
                                                # ***（文字列から行列を作成）
  w=np.array(
     [[119,194,367,209],
      [458,470,715,486],
      [137,456,388,463],
      [10,564,272,560]])
  """
  w=np.array(
     [[113,188,231,201],
      [281,513,398,521],
      [621,218,742,223],
      [573,437,697,448],])
  """
  m10=calcHomography(w)

  m1d=m10 @ m0d                                 # 画像1 と 出力画像 の関係

  dst0=proj(dw,dh, im0, m0d)      # 射影変換後の入力画像0
  dst1=proj(dw,dh, im1, m1d)
  dst=(dst0+dst1)/2               # 画像合成（平均）

  eb.imshow(dst)



def proj(dw,dh, src, a):

 im=np.zeros((dh,dw,3),'f')

 y ,x =np.mgrid[0:dh,0:dw]                   # (i)

 # z_を追加-----
 x_,y_,z_= a[0,0]*x+a[0,1]*y+a[0,2], \
        a[1,0]*x+a[1,1]*y+a[1,2], \
        a[2,0]*x+a[2,1]*y+a[2,2]

 # z_を追加-----
 u,v=x_ / z_, \
    y_ / z_

 m = (0<=u)&(u<src.shape[1])&(0<=v)&(v<src.shape[0]) # (iv)
 # 画像内の座標しか参照しないため，画像を拡大できる

 im[y[m],x[m]]= src[v[m].astype('i'),
		    u[m].astype('i')]        # (iii) int型でやっている

 return im