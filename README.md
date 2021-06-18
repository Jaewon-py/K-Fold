#K-Fold 

데이터를 k개 분할로 나누고, k-1개의 분할에서 훈련이 이루어지며, 나머지 분할에서 평가하는 방법입니다. score의 경우, 각 Fold 별 score들의 평균값을 사용합니다.

![](https://blog.kakaocdn.net/dn/Ac7Cd/btqXXkuYgrM/k2kXdtXSpyoHuBmHVqFGw0/img.png)
(code)
```
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=2021)


<parameters>
-n_splits (int) : Fold의 개수 k 값
-shuffle (bool) : 데이터를 쪼갤 때 섞을지 유무
-random_state (int) : 내부적으로 사용되는 난수값
```


#Stratified K-Fold

기존의 K-Fold의 경우, 분류할 클래스의 비율에 상관없이 데이터를 분할합니다. 하지만 분류할 클래스의 비율이 다를 경우, K-Fold를 적용한 각 Fold가 학습 dataset을 대표한다고 할 수 없습니다. 이러한 문제를 해결하기 위해 Stratified K-Fold가 제안되었습니다.

Stratified K-Fold의 경우, k개의 Fold로 분할한 이후에도 전체 훈련 데이터의 클래스 비율과 각 Fold가 가지고 있는 클래스의 비율을 맞추어 주기 때문에 dataset의 대표성이 보장됩니다. 

![](https://www.researchgate.net/profile/Mohsen-Azimi-2/publication/336889074/figure/fig18/AS:822836264460288@1573190861533/Visualization-of-stratified-k-fold-cross-validation-with-k5.png)
(code)
```
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)

<parameters>
-n_splits (int) : Fold의 개수 k 값
-shuffle (bool) : 데이터를 쪼갤 때 섞을지 유무
-random_state (int) : 내부적으로 사용되는 난수값
```
