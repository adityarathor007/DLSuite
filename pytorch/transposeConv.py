import numpy as np

_input=np.array([
    [3,3,2,1],
    [0,0,1,3],
    [3,1,2,2],
    [2,0,0,2]
])

I=_input.reshape(-1,1)


# kernel was and C is created by placing based on stride steps and padding rest to zero and then flattening each stride step from (4*4) to (1*16)
# [[1,2,0],
#  [2,2,0],
#  [0,1,2]]

C=np.array([
    [0,1,2,0,2,2,0,0,0,1,2,0,0,0,0,0],
    [0,0,1,2,0,2,2,0,0,0,1,2,0,0,0,0],
    [0,0,0,0,0,1,2,0,2,2,0,0,0,1,2,0],
    [0,0,0,0,0,0,1,2,0,2,2,0,0,0,1,2],
])


res=C@I

print(f"res \n {res}")

final=res.reshape(2,-1)
print(f"final result: \n {final}")


# now performing transpose conv
c_transposed=C.T
res_reshaped=res.reshape(4,1)

res=c_transposed@res_reshaped
print(f"res: \n {res}")
print(f"final result: \n {res.reshape(4,-1)}") #even though the kernel is same still the output of transposeConv wont match the input for the Conv 
# As a standard convolution loses some spatial information due to stride and padding.


