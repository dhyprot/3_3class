# [5.1 features]


def LocalMax(src,W=7):
# (5) 近傍での最大値検出
  out=np.zeros_like(src)
  t=np.zeros_like(src)
  for v in range(-W,W +1):
    t[W:-W,:]=np.maximum(t[W:-W,:],
                            src[v+W:v+src.shape[0]-W,:])
  for u in range(-W,W+1):
    out[:,W:-W]=np.maximum(out[:,W:-W],
                              t[:,u+W:u+t.shape[1]-W])
  return out

def detectFeaturePoints(src,MAX=100):
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

  eb.imshow(img)

  return ary


def main():
  src=eb.getImg('0.jpg') # 入力画像
  eb.                      stopWatch('start')
  ary=detectFeaturePoints(src)
  print(ary)             # ary を表示
  eb.showTime()
