1. 
(d) best f1 0.82
go\gu           PER             ORG             LOC             MISC     O
PER             2925.00         49.00           78.00           23.00    74.00
ORG             144.00          1628.00         105.00          83.00    132.00
LOC             39.00           111.00          1853.00         34.00    57.00
MISC            36.00           62.00           40.00           1023.00  107.00
O               33.00           54.00           16.00           31.00    42625.00

a)feature of casing and word using the same embedding matrix, althought they are independent.
=> can be regared as different embedding matrix, independent paths actually
However the embedding size for casing might be to large as 4 case types only.

b)fixed size window which might fail to capture all context informationex
e.g.
```
x : I want to have dinner at Geogre 's home.
y*:
y': O O    O  O    O      O  LOC    O  O
```
where Geogre should be PER or MISC

c)cannot use the predicted result in the context,  which is also important information
```
x : Quarter-final results in the Hong Kong Open on Friday .
y*: 
y': O             O       O  O   LOC  MISC MISC O  O      O
```

2. 
(a)
(i)One more matrix which multiplys the former hidden layer result.
(ii)for the hidden layer, input part O(DH), recursive part O(H^2), intercept and adding O(H). Which is totally O(DH + H2)
for the output layer,  O(HC) for vector-matrix multiplying, C for softmax 
Thus totally should be O(DH+H^2+HC)= O(H\*(D+H+C))
for length T, it should be 
O(H\*T\*(D+H+C))
(b) 
(i) imbalanced dataset
(ii)`$F_1$` is related the category size, which is a global information, thus hard to use SGD. => Might use batch SGD ? Also F_1 is not differentiable
(d)3(i)filtering out the appending part and hence no gradient from that part will be used
(g) only past(left) context infomation will be preserved in hidden layer = > use bi-directional RNN
    gradient explosion/vanish problem => gradient clipping or better model like GRU

    the model doesn't enforce that adjacent tokens have the same tag ("New York State University")
3. 
(a)
i. `$w_h = 1$`,`$u_h = 1$`,`$b = -0.5$`
ii. `$u_z = 1,w_z=1,u_h=1$`
(b)
i.
ii.